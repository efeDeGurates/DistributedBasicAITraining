import socket
import threading
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import pickle

# --- Vocab sınıfı ---
class Vocab:
    def __init__(self, specials=["<pad>", "<sos>", "<eos>", "<unk>"]):
        self.token_to_idx = {}
        self.idx_to_token = []
        for token in specials:
            self.add_token(token)

    def add_token(self, token):
        if token not in self.token_to_idx:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def build_vocab(self, token_lists):
        for tokens in token_lists:
            for token in tokens:
                self.add_token(token)

    def __call__(self, tokens):
        # tokens: iterable of token str
        return [self.token_to_idx.get(token, self.token_to_idx.get("<unk>", 0)) for token in tokens]

    def lookup_token(self, idx):
        return self.idx_to_token[idx] if 0 <= idx < len(self.idx_to_token) else "<unk>"

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.token_to_idx.get("<unk>", 0))

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# --- Transformer Chatbot Model ---
class TransformerChatbot(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt):
        # src: (batch, src_len), tgt: (batch, tgt_len)
        src_emb = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float, device=src.device))
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float, device=tgt.device))
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        output = self.fc_out(output)
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, device='cpu') * float('-inf'), diagonal=1)
        return mask

# --- Veri ve vocab hazırlama ---
def load_data_and_prepare_vocab(csv_path):
    df = pd.read_csv(csv_path)
    inputs = df["input"].astype(str).tolist()
    outputs = df["output"].astype(str).tolist()

    tokenized_in = [s.lower().split() for s in inputs]
    tokenized_out = [["<sos>"] + s.lower().split() + ["<eos>"] for s in outputs]

    vocab = Vocab()
    vocab.build_vocab(tokenized_in + tokenized_out)

    def encode(sentences):
        return [torch.tensor(vocab(s), dtype=torch.long) for s in sentences]

    enc_in = pad_sequence(encode(tokenized_in), batch_first=True, padding_value=vocab["<pad>"])
    enc_out = pad_sequence(encode(tokenized_out), batch_first=True, padding_value=vocab["<pad>"])
    return enc_in, enc_out, vocab

# --- Eğitim fonksiyonu ---
def train(model, input_tensor, output_tensor, optimizer, loss_fn, vocab, epochs=20):
    model.train()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    output_tensor = output_tensor.to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        tgt_input = output_tensor[:, :-1]
        tgt_output = output_tensor[:, 1:]
        preds = model(input_tensor, tgt_input)
        loss = loss_fn(preds.reshape(-1, preds.shape[-1]), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch} Loss: {loss.item():.4f}")

# --- Vocab kaydetme ---
def save_vocab(vocab, path="vocab_client.pkl"):
    with open(path, "wb") as f:
        pickle.dump(vocab, f)

# --- Vocab gönderme: header+size protokolü ---
def send_vocab(client_socket, vocab_path="vocab_client.pkl"):
    try:
        # Dosya boyutu
        filesize = os.path.getsize(vocab_path)
        filename = os.path.basename(vocab_path)
        header = f"VOCAB_UPLOAD:{filename}:{filesize}\n"
        client_socket.sendall(header.encode())
        # Ardından raw byte gönder
        with open(vocab_path, "rb") as f:
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                client_socket.sendall(chunk)
        print("[✓] Vocab başarıyla gönderildi.")
    except Exception as e:
        print(f"[!] Vocab gönderme hatası: {e}")

# --- Model gönderme: header+size protokolü ---
def send_model(client_socket, model_path):
    try:
        filesize = os.path.getsize(model_path)
        filename = os.path.basename(model_path)
        header = f"MODEL_UPLOAD:{filename}:{filesize}\n"
        client_socket.sendall(header.encode())
        # Ardından raw byte gönder
        with open(model_path, "rb") as f:
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                client_socket.sendall(chunk)
        print("[✓] Model başarıyla gönderildi.")
    except Exception as e:
        print(f"[!] Model gönderme hatası: {e}")

# --- Veri alma ve eğitim thread ---
def receive_data_and_train(client_socket):
    buffer = ""
    receiving = False
    while True:
        try:
            data = client_socket.recv(4096)
            if not data:
                print("[!] Server bağlantısı kapandı.")
                break
            text = data.decode(errors='ignore')
            # DATA_START ve DATA_END işareti
            if "DATA_START" in text:
                receiving = True
                buffer = ""
                # Eğer DATA_START ve aynı pakette CSV verisi de geldiyse, split edebiliriz:
                parts = text.split("DATA_START", 1)[1]
                # parts içinde CSV + belki DATA_END
                if "DATA_END" in parts:
                    # CSV tek pakette gelmiş demek
                    content = parts.split("DATA_END",1)[0]
                    buffer += content
                    receiving = False
                    # Kaydet ve eğitime geç:
                    with open("received_data.csv", "w", encoding="utf-8") as f:
                        f.write(buffer)
                    print("Veri alındı (tek pakette), eğitim başlıyor...")
                    _handle_training_and_send(client_socket)
                else:
                    # CSV parçaları devam edecek
                    buffer += parts
                continue
            elif receiving:
                # CSV parçalanarak geliyor
                if "DATA_END" in text:
                    content = text.split("DATA_END",1)[0]
                    buffer += content
                    receiving = False
                    with open("received_data.csv", "w", encoding="utf-8") as f:
                        f.write(buffer)
                    print("Veri alındı, eğitim başlıyor...")
                    _handle_training_and_send(client_socket)
                else:
                    buffer += text
                continue
            else:
                print(f"[Server]: {text.strip()}")
        except Exception as e:
            print(f"[!] Hata in receive_data_and_train: {e}")
            break

def _handle_training_and_send(client_socket):
    try:
        enc_in, enc_out, vocab = load_data_and_prepare_vocab("received_data.csv")
    except Exception as e:
        print(f"[!] Veri hazırlama hatası: {e}")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerChatbot(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    # Eğitim
    print("[*] Eğitim başlayacak...")
    train(model, enc_in, enc_out, optimizer, loss_fn, vocab, epochs=20)
    # Modeli kaydet
    model_dir = "trained_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "trained_model.pt")
    torch.save(model.state_dict(), model_path)
    print("[✓] Eğitim tamamlandı, model kaydedildi:", model_path)
    # Vocab kaydet ve gönder
    vocab_path = "vocab_client.pkl"
    save_vocab(vocab, vocab_path)
    send_vocab(client_socket, vocab_path)
    # Model gönder
    send_model(client_socket, model_path)

def main():
    client = socket.socket()
    SERVER_IP = '127.0.0.1'
    SERVER_PORT = 12345
    try:
        client.connect((SERVER_IP, SERVER_PORT))
    except Exception as e:
        print(f"[!] Sunucuya bağlanırken hata: {e}")
        return
    print("[✓] Sunucuya bağlanıldı:", SERVER_IP, SERVER_PORT)


    score_value = 10.5 
    header = f"SCORE:{score_value}\n"
    client.sendall(header.encode())

    threading.Thread(target=receive_data_and_train, args=(client,), daemon=True).start()

    while True:
        cmd = input("Komut (exit): ").strip()
        if cmd.lower() == "exit":
            break
        try:
            client.sendall((cmd + "\n").encode())
        except:
            break

    client.close()
    print("Client kapatıldı.")

if __name__ == "__main__":
    main()

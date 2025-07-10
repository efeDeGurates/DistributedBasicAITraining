import socket
import threading
import os
import time
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import pickle
import tkinter as tk
from tkinter import scrolledtext


# --- GUI Class ---
class ClientGUI():
    def __init__(self, root):
        self.root = root
        self.root.title("Distributed AI Client")
        self.root.geometry("850x650")

        tk.Label(root, text="Server IP:").pack()
        self.ip_entry = tk.Entry(root, width=30)
        self.ip_entry.pack()
        self.ip_entry.insert(0, "127.0.0.1")

        tk.Label(root, text="Server Port:").pack()
        self.port_entry = tk.Entry(root, width=10)
        self.port_entry.pack()
        self.port_entry.insert(0, "12345")

        self.connect_btn = tk.Button(root, text="Connect to Server", command=self.connect_to_server)
        self.connect_btn.pack(pady=5)

        # Logs
        self.log_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=25)
        self.log_box.pack(pady=10)

        self.cmd_entry = tk.Entry(root, width=80)
        self.cmd_entry.pack(pady=5)
        self.send_btn = tk.Button(root, text="Send Command", command=self.send_command)
        self.send_btn.pack()

        self.client_socket = None

    def log(self, msg):
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)

    def connect_to_server(self):
        ip = self.ip_entry.get().strip()
        try:
            port = int(self.port_entry.get().strip())
        except ValueError:
            self.log("[!] Invalid port")
            return

        try:
            self.client_socket = socket.socket()
            self.client_socket.connect((ip, port))
            self.log(f"[✓] Connected to {ip}:{port}")

            score = 10.0
            self.client_socket.sendall(f"SCORE:{score}\n".encode())

            threading.Thread(target=receive_data_and_train, args=(self.client_socket, self.log), daemon=True).start()
        except Exception as e:
            self.log(f"[!] Connection error: {e}")

    def send_command(self):
        cmd = self.cmd_entry.get().strip()
        if cmd and self.client_socket:
            try:
                self.client_socket.sendall((cmd + "\n").encode())
                self.cmd_entry.delete(0, tk.END)
            except:
                self.log("[!] Failed to send command")


# --- Vocab and Model Class ---
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
        return [self.token_to_idx.get(token, self.token_to_idx.get("<unk>", 0)) for token in tokens]

    def lookup_token(self, idx):
        return self.idx_to_token[idx] if 0 <= idx < len(self.idx_to_token) else "<unk>"

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.token_to_idx.get("<unk>", 0))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerChatbot(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt):
        src_emb = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float, device=src.device))
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float, device=tgt.device))
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1)) * float('-inf'), diagonal=1).to(tgt.device)
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.fc_out(out)


# --- Training & Communication ---
def train(model, input_tensor, output_tensor, optimizer, loss_fn, vocab, log_func, epochs=20):
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
            log_func(f"[{epoch}] Loss: {loss.item():.4f}")


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


def save_vocab(vocab, path="vocab_client.pkl"):
    with open(path, "wb") as f:
        pickle.dump(vocab, f)


def send_file(client_socket, filepath, tag):
    try:
        filesize = os.path.getsize(filepath)
        filename = os.path.basename(filepath)
        header = f"{tag}:{filename}:{filesize}\n"
        client_socket.sendall(header.encode())
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                client_socket.sendall(chunk)
    except Exception as e:
        print(f"[!] File send error: {e}")


def _handle_training_and_send(client_socket, log_func):
    try:
        enc_in, enc_out, vocab = load_data_and_prepare_vocab("received_data.csv")
    except Exception as e:
        log_func(f"[!] Data error: {e}")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerChatbot(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    log_func("[*] Training started...")
    train(model, enc_in, enc_out, optimizer, loss_fn, vocab, log_func)
    torch.save(model.state_dict(), "trained_model.pt")
    save_vocab(vocab)
    log_func("[✓] Model saved. Sending to server...")
    send_file(client_socket, "trained_model.pt", "MODEL_UPLOAD")
    send_file(client_socket, "vocab_client.pkl", "VOCAB_UPLOAD")


def receive_data_and_train(client_socket, log_func):
    buffer = ""
    receiving = False
    while True:
        try:
            data = client_socket.recv(4096)
            if not data:
                log_func("[!] Server disconnected.")
                break
            text = data.decode(errors='ignore')
            if "DATA_START" in text:
                receiving = True
                buffer = ""
                parts = text.split("DATA_START", 1)[1]
                if "DATA_END" in parts:
                    content = parts.split("DATA_END", 1)[0]
                    buffer += content
                    receiving = False
                    with open("received_data.csv", "w", encoding="utf-8") as f:
                        f.write(buffer)
                    _handle_training_and_send(client_socket, log_func)
                else:
                    buffer += parts
                continue
            elif receiving:
                if "DATA_END" in text:
                    content = text.split("DATA_END", 1)[0]
                    buffer += content
                    receiving = False
                    with open("received_data.csv", "w", encoding="utf-8") as f:
                        f.write(buffer)
                    _handle_training_and_send(client_socket, log_func)
                else:
                    buffer += text
            else:
                log_func(f"[Server]: {text.strip()}")
        except Exception as e:
            log_func(f"[!] Receive error: {e}")
            break


# --- Main ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ClientGUI(root)
    root.mainloop()

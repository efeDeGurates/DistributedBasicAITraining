import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
import os
import csv
import tkinter as tk
from dotenv import load_dotenv

load_dotenv()

serp_api_key = os.getenv("SERP_API_KEY")


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

    def __call__(self, tokens):
        return [self.token_to_idx.get(token, self.token_to_idx["<unk>"]) for token in tokens]

    def lookup_token(self, idx):
        return self.idx_to_token[idx] if idx < len(self.idx_to_token) else "<unk>"

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.token_to_idx["<unk>"])

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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# --- Transformer Model ---
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
        src_mask = None
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        src_emb = self.pos_encoder(self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float, device=src.device)))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float, device=tgt.device)))
        output = self.transformer(src_emb, tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask)
        output = self.fc_out(output)
        return output

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# --- Beam Search ---
def beam_search(model, vocab, src_sentence, beam_width=3, max_len=20, device='cpu'):
    model.eval()
    src_tokens = torch.tensor(vocab(src_sentence.lower().split())).unsqueeze(0).to(device)
    with torch.no_grad():
        memory = model.transformer.encoder(
            model.pos_encoder(model.embedding(src_tokens) * torch.sqrt(torch.tensor(model.d_model, dtype=torch.float, device=device)))
        )
    sequences = [([vocab["<sos>"]], 0.0)]

    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == vocab["<eos>"]:
                all_candidates.append((seq, score))
                continue
            tgt = torch.tensor(seq).unsqueeze(0).to(device)
            tgt_emb = model.pos_encoder(model.embedding(tgt) * torch.sqrt(torch.tensor(model.d_model, dtype=torch.float, device=device)))
            tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(device)
            out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = model.fc_out(out[:, -1, :])
            log_probs = F.log_softmax(logits, dim=-1)
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_width)
            for k in range(beam_width):
                candidate = (seq + [topk_ids[0, k].item()], score + topk_log_probs[0, k].item())
                all_candidates.append(candidate)
        sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_width]

    best_seq = sequences[0][0]
    tokens = [vocab.lookup_token(idx) for idx in best_seq[1:] if idx != vocab["<eos>"]]
    return " ".join(tokens)

def predict(model, vocab, sentence, beam_width=5, max_len=20, device='cpu'):
    return beam_search(model, vocab, sentence, beam_width, max_len, device)

# --- Chat Log ---
def log_chat(user_input, bot_response, logfile="chat_history.csv"):
    file_exists = os.path.isfile(logfile)
    with open(logfile, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["input", "output"])
        writer.writerow([user_input.strip(), bot_response.strip()])

# --- Web Function ---
def get_weather():
    try:
        r = requests.get("https://wttr.in/?format=3", timeout=5)
        return r.text.strip()
    except:
        return "I can't stand the weather right now."

def get_news():
    try:
        r = requests.get("https://edition.cnn.com/", timeout=5)
        soup = BeautifulSoup(r.content, "xml")
        haber = soup.find("item").find("title").text
        return f"Güncel haber: {haber}"
    except:
        return "I can't stand the news right now."

def google_search(query):
    params = {
        "q": query,
        "api_key": serp_api_key,
        "hl": "tr",
        "gl": "tr"
    }
    search = GoogleSearch(params)
    result = search.get_dict()
    try:
        snippet = result["organic_results"][0]["snippet"]
        link = result["organic_results"][0]["link"]
        return f"{snippet}\n{link}"
    except:
        return "I didn't find result"

# --- RunnerGUI ---
class RunnerGUI():
    def __init__(self, model, vocab, device):
        self.model = model
        self.vocab = vocab
        self.device = device

        self.root = tk.Tk()
        self.root.title("Runner")
        self.root.geometry("900x700")
        self.root.configure(bg="#121212")

        self.chat_log = tk.Text(
            self.root,
            state='disabled',
            wrap='word',
            bg="#1e1e1e",
            fg="white",
            font=("Consolas", 16),
            insertbackground="white",
            padx=10,
            pady=10,
            relief=tk.FLAT,
            borderwidth=0
        )
        self.chat_log.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Scrollbar
        scrollBar = tk.Scrollbar(self.chat_log, bg="#2e2e2e", troughcolor="#2e2e2e", activebackground="#3e3e3e")
        scrollBar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_log.config(yscrollcommand=scrollBar.set)
        scrollBar.config(command=self.chat_log.yview)

        # Tag ayarları
        self.chat_log.tag_config("user_right", 
            foreground="#4FC3F7", 
            font=("Consolas", 16, "bold"), 
            justify="right", 
            lmargin1=220, lmargin2=220,
            rmargin=10,
            spacing3=10
        )
        self.chat_log.tag_config("bot_center", 
            foreground="#A5D6A7", 
            font=("Consolas", 16), 
            justify="left",
            lmargin1=100, lmargin2=100,
            rmargin=300,
            spacing3=10
        )
        self.chat_log.tag_config("right_space", font=("Consolas", 4))

        # Entry box
        self.entry = tk.Entry(
            self.root,
            bg="#2c2c2c",
            fg="white",
            font=("Consolas", 16),
            insertbackground="white",
            relief=tk.FLAT
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=10)

        # Send button
        self.sendButton = tk.Button(
            self.root,
            text="Gönder",
            command=self.send_message,
            bg="#4CAF50",
            fg="white",
            font=("Consolas", 14, "bold"),
            activebackground="#66BB6A",
            relief=tk.FLAT,
            padx=20,
            pady=5
        )
        self.sendButton.pack(side=tk.RIGHT, padx=10, pady=10)

        self.entry.bind("<Return>", self.send_message)

    def assistant_response(self, soru):
        soru_lower = soru.lower()
        trigger_keywords = ["What","who", "how"]
        for keyword in trigger_keywords:
            if keyword in soru_lower:
                if "weather" in soru_lower:
                    return get_weather()
                elif "news" in soru_lower:
                    return get_news()

        if soru_lower.startswith("ara") or soru_lower.startswith("search"):
            query = soru.split(" ", 1)[1] if " " in soru else soru
            return google_search(query)

        return predict(self.model, self.vocab, soru, device=self.device)

    def send_message(self, event=None):
        user_msg = self.entry.get()
        if not user_msg.strip():
            return
        self.entry.delete(0, tk.END)

        self.chat_log.config(state='normal')

        self.chat_log.insert(tk.END, "\n", "right_space")
        self.chat_log.insert(tk.END, f"Sen: {user_msg}\n", "user_right")


        bot_reply = self.assistant_response(user_msg)
        self.chat_log.insert(tk.END, f"{bot_reply}\n\n", "bot_center")

        self.chat_log.config(state='disabled')
        self.chat_log.see(tk.END)

        log_chat(user_msg, bot_reply)

    def run(self):
        self.root.mainloop()



# --- Main function ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("received_vocabs/vocab_client.pkl", "rb") as f:
        vocab = pickle.load(f)

    model = TransformerChatbot(len(vocab)).to(device)
    model.load_state_dict(torch.load("received_models/trained_model.pt", map_location=device))
    model.eval()

    print("Chatbot working\n")

    app = RunnerGUI(model, vocab, device)
    app.run()

if __name__ == "__main__":
    main()

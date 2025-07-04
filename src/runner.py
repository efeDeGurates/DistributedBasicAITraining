import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
import os
import csv

SERP_API_KEY = ""

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

# --- Encoding ---
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

# --- Model ---
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
        memory = model.transformer.encoder(model.pos_encoder(model.embedding(src_tokens) * torch.sqrt(torch.tensor(model.d_model, dtype=torch.float, device=device))))
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

def predict(sentence):
    return beam_search(model, vocab, sentence, beam_width=5, max_len=20, device=device)

# --- Chat Log ---
def log_chat(user_input, bot_response, logfile="chat_history.csv"):
    file_exists = os.path.isfile(logfile)
    with open(logfile, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["input", "output"])  # başlıkları ekle
        writer.writerow([user_input.strip(), bot_response.strip()])

# --- Web Fonksiyonları ---
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
        "api_key": SERP_API_KEY,
        "hl": "tr",
        "gl": "tr"
    }
    search = GoogleSearch(params)
    result = search.get_dict()
    try:
        snippet = result["organic_results"][0]["snippet"]
        link = result["organic_results"][0]["link"]
        return f"🔎 {snippet}\n📎 {link}"
    except:
        return "I didn't find result"

# --- Web ---
def assistant_response(soru):
    soru_lower = soru.lower()

    trigger_keywords = ["bugün", "şu an", "şuan", "nedir", "kaç", "hava", "dolar","kimdir","who","how"]
    for keyword in trigger_keywords:
        if keyword in soru_lower:
            if "weather" in soru_lower:
                return get_weather()
            elif "news" in soru_lower:
                return get_news()

    if soru_lower.startswith("ara") or soru_lower.startswith("search"):
        query = soru.split(" ", 1)[1] if " " in soru else soru
        return google_search(query)

    return predict(soru)

# --- Ana ---
def main():
    global vocab, model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("received_vocabs/vocab_client.pkl", "rb") as f:
        vocab = pickle.load(f)

    model = TransformerChatbot(len(vocab)).to(device)
    model.load_state_dict(torch.load("received_models/trained_model.pt", map_location=device))
    model.eval()

    print("🤖 Chatbot working\n")

    while True:
        text = input("Sen: ")
        if text.lower() in ["exit", "quit"]:
            print("Çıkılıyor...")
            break
        cevap = assistant_response(text)
        print("Bot:", cevap)
        log_chat(text, cevap)

if __name__ == "__main__":
    main()

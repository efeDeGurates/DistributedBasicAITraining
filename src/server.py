import socket
import threading
import pandas as pd
import os
import pickle
import torch
import time
import tkinter as tk
from tkinter import scrolledtext

# GUI
import tkinter as tk
from tkinter import scrolledtext, filedialog, ttk

class ServerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Distributed AI Server")
        self.root.geometry("900x700")

        self.log_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Consolas", 10))
        self.log_box.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)

        params_frame = tk.Frame(root)
        params_frame.pack(fill=tk.X, padx=10)

        tk.Label(params_frame, text="Dataset CSV:", font=("Consolas", 10)).grid(row=0, column=0, sticky="w", pady=5)
        self.dataset_path_var = tk.StringVar(value=DATA_FILE)
        self.dataset_entry = tk.Entry(params_frame, textvariable=self.dataset_path_var, font=("Consolas", 10), width=50)
        self.dataset_entry.grid(row=0, column=1, padx=5, sticky="w")
        self.browse_button = tk.Button(params_frame, text="Seç", command=self.browse_dataset)
        self.browse_button.grid(row=0, column=2, padx=5)

        # Epoch
        tk.Label(params_frame, text="Epoch Number:", font=("Consolas", 10)).grid(row=1, column=0, sticky="w", pady=5)
        self.epoch_var = tk.IntVar(value=5)
        self.epoch_spinbox = tk.Spinbox(params_frame, from_=1, to=100, textvariable=self.epoch_var, width=5, font=("Consolas", 10))
        self.epoch_spinbox.grid(row=1, column=1, sticky="w")

        # Batch size
        tk.Label(params_frame, text="Batch Size:", font=("Consolas", 10)).grid(row=2, column=0, sticky="w", pady=5)
        self.batch_var = tk.IntVar(value=32)
        self.batch_spinbox = tk.Spinbox(params_frame, from_=1, to=512, textvariable=self.batch_var, width=5, font=("Consolas", 10))
        self.batch_spinbox.grid(row=2, column=1, sticky="w")

        # Learning rate
        tk.Label(params_frame, text="Learning rate (LR):", font=("Consolas", 10)).grid(row=3, column=0, sticky="w", pady=5)
        self.lr_var = tk.DoubleVar(value=0.001)
        self.lr_spinbox = tk.Spinbox(params_frame, from_=0.00001, to=1.0, increment=0.0001, format="%.5f", textvariable=self.lr_var, width=7, font=("Consolas", 10))
        self.lr_spinbox.grid(row=3, column=1, sticky="w")

        # Start server button
        self.start_button = tk.Button(root, text="Start Server", command=self.start_server, bg="green", fg="white", font=("Consolas", 12, "bold"))
        self.start_button.pack(pady=15)

    def browse_dataset(self):
        filename = filedialog.askopenfilename(
            title="Select dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.dataset_path_var.set(filename)

    def log(self, msg):
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)
        print(msg)

    def start_server(self):
        self.log(f"[i] Sunucu başlatılıyor...")
        self.log(f"[i] Eğitim seti: {self.dataset_path_var.get()}")
        self.log(f"[i] Epoch sayısı: {self.epoch_var.get()}")
        self.log(f"[i] Batch size: {self.batch_var.get()}")
        self.log(f"[i] Öğrenme oranı: {self.lr_var.get()}")

        global DATA_FILE, EPOCHS, BATCH_SIZE, LEARNING_RATE
        DATA_FILE = self.dataset_path_var.get()
        EPOCHS = self.epoch_var.get()
        BATCH_SIZE = self.batch_var.get()
        LEARNING_RATE = self.lr_var.get()

        threading.Thread(target=main_server_logic, args=(self.log,), daemon=True).start()
        self.start_button.config(state=tk.DISABLED)


# --- Server ---

clients = {}
models = {}
vocabs = {}
lock = threading.Lock()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", "datasets", "data.csv"))
MODEL_SAVE_DIR = "received_models"
VOCAB_SAVE_DIR = "received_vocabs"
AGG_MODEL_PATH = "model/aggregated_model.pt"
AGG_VOCAB_PATH = "vocab/aggregated_vocab.pkl"

for path in [MODEL_SAVE_DIR, VOCAB_SAVE_DIR, os.path.dirname(AGG_MODEL_PATH), os.path.dirname(AGG_VOCAB_PATH)]:
    os.makedirs(path, exist_ok=True)

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

def recv_exactly(sock, n):
    data = b""
    while len(data) < n:
        to_read = n - len(data)
        chunk = sock.recv(to_read)
        if not chunk:
            break
        data += chunk
    return data

def send_dataset(client_socket, df_slice, log):
    csv_str = df_slice.to_csv(index=False)
    try:
        client_socket.sendall(b"DATA_START")
        client_socket.sendall(csv_str.encode('utf-8'))
        client_socket.sendall(b"DATA_END")
        log(f"[+] Veri gönderildi: {clients[client_socket]['addr']}")
    except Exception as e:
        log(f"[!] Veri gönderme hatası: {e}")

def broadcast_datasets(log):
    global clients
    with lock:
        if not clients:
            log("[!] Bağlı client yok, veri gönderilemiyor.")
            return

        sorted_clients = sorted(clients.items(), key=lambda x: x[1]["score"], reverse=True)
        total_score = sum([info["score"] for _, info in sorted_clients])
        log(f"[+] Toplam skor: {total_score}, Client sayısı: {len(sorted_clients)}")

        try:
            df = pd.read_csv(DATA_FILE)
        except Exception as e:
            log(f"[!] DATA_FILE okunurken hata: {e}")
            return

        total_len = len(df)
        start_idx = 0

        for idx, (client_socket, info) in enumerate(sorted_clients):
            ratio = info["score"] / total_score if total_score > 0 else 1 / len(sorted_clients)
            length = int(total_len * ratio)
            if idx == len(sorted_clients) - 1:
                length = total_len - start_idx
            df_slice = df.iloc[start_idx:start_idx + length]
            start_idx += length

            clients[client_socket]["data_slice"] = df_slice
            log(f"[>] {info['addr']} veri oranı: %{ratio * 100:.2f}, satır: {len(df_slice)}")
            send_dataset(client_socket, df_slice, log)

def receive_model(client_socket, filename, filesize, log):
    filepath = os.path.join(MODEL_SAVE_DIR, filename)
    log(f"[+] Model alınıyor: {filename} ({filesize} byte)")
    data = recv_exactly(client_socket, filesize)
    try:
        with open(filepath, "wb") as f:
            f.write(data)
    except Exception as e:
        log(f"[!] Model yazma hatası: {e}")
        return False

    try:
        state_dict = torch.load(filepath, map_location=torch.device('cpu'))
        with lock:
            models[client_socket] = state_dict
        log(f"[✓] Model yüklendi: {filepath}")
        return True
    except Exception as e:
        log(f"[!] Model yükleme hatası: {e}")
        return False

def receive_vocab(client_socket, filename, filesize, log):
    filepath = os.path.join(VOCAB_SAVE_DIR, filename)
    log(f"[+] Vocab alınıyor: {filename} ({filesize} byte)")
    data = recv_exactly(client_socket, filesize)
    try:
        with open(filepath, "wb") as f:
            f.write(data)
    except Exception as e:
        log(f"[!] Vocab yazma hatası: {e}")
        return False
    try:
        vocab_obj = pickle.loads(data)
        with lock:
            vocabs[client_socket] = vocab_obj
        log(f"[✓] Vocab yüklendi: {filepath}")
        return True
    except Exception as e:
        log(f"[!] Vocab yükleme hatası: {e}")
        return False

def aggregate_models(log):
    if not models:
        log("[!] Birleştirilecek model yok.")
        return
    log(f"[+] {len(models)} model birleştiriliyor...")
    model_keys = list(next(iter(models.values())).keys())
    aggregated_state_dict = {key: sum(m[key] for m in models.values()) / len(models) for key in model_keys}
    try:
        torch.save(aggregated_state_dict, AGG_MODEL_PATH)
        log(f"[✓] Model kaydedildi: {AGG_MODEL_PATH}")
    except Exception as e:
        log(f"[!] Model kaydedilemedi: {e}")

def aggregate_vocabs(log):
    all_vocabs = []
    for fname in os.listdir(VOCAB_SAVE_DIR):
        try:
            with open(os.path.join(VOCAB_SAVE_DIR, fname), "rb") as f:
                obj = pickle.load(f)
                if isinstance(obj, Vocab):
                    all_vocabs.append(obj)
        except Exception as e:
            log(f"[!] Vocab hatası: {e}")
    if not all_vocabs:
        log("[!] Vocab bulunamadı.")
        return
    merged = Vocab()
    for v in all_vocabs:
        for token in v.idx_to_token:
            merged.add_token(token)
    try:
        with open(AGG_VOCAB_PATH, "wb") as f:
            pickle.dump(merged, f)
        log(f"[✓] Vocab kaydedildi: {AGG_VOCAB_PATH}")
    except Exception as e:
        log(f"[!] Vocab kaydedilemedi: {e}")

def handle_client(client_socket, addr, log):
    peer_id = f"{addr[0]}:{addr[1]}"
    clients[client_socket] = {"addr": peer_id, "score": 0.0, "data_slice": None}
    log(f"[+] Yeni bağlantı: {peer_id}")

    try:
        while True:
            header = b""
            while b"\n" not in header:
                chunk = client_socket.recv(1)
                if not chunk:
                    raise ConnectionError("Client kapandı.")
                header += chunk
            header = header.decode(errors='ignore').strip()

            if header.startswith("SCORE:"):
                try:
                    score = float(header.split(":", 1)[1])
                    with lock:
                        clients[client_socket]["score"] = score
                    log(f"{peer_id} skoru: {score}")
                    broadcast_datasets(log)
                except Exception as e:
                    log(f"[!] SCORE parse hatası: {e}")
                continue

            elif header.startswith("MODEL_UPLOAD:"):
                _, filename, size_str = header.split(":", 2)
                success = receive_model(client_socket, filename, int(size_str), log)
                if success:
                    aggregate_models(log)
                continue

            elif header.startswith("VOCAB_UPLOAD:"):
                _, filename, size_str = header.split(":", 2)
                success = receive_vocab(client_socket, filename, int(size_str), log)
                if success:
                    aggregate_vocabs(log)
                continue

            else:
                log(f"[{peer_id}] {header}")
    except Exception as e:
        log(f"[!] {peer_id} bağlantı hatası: {e}")
    finally:
        with lock:
            clients.pop(client_socket, None)
            models.pop(client_socket, None)
            vocabs.pop(client_socket, None)
        client_socket.close()
        log(f"[-] {peer_id} bağlantısı kesildi.")

def main_server_logic(log):
    server = socket.socket()
    server.bind(('0.0.0.0', 12345))
    server.listen(10)
    log("[✓] Sunucu dinleniyor...")

    def accept_clients():
        while True:
            client_socket, addr = server.accept()
            threading.Thread(target=handle_client, args=(client_socket, addr, log), daemon=True).start()

    threading.Thread(target=accept_clients, daemon=True).start()
    log("[i] Client bağlantıları bekleniyor...")

# --- GUI Start  ---
if __name__ == "__main__":
    root = tk.Tk()
    gui = ServerGUI(root)
    root.mainloop()

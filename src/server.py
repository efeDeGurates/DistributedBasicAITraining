import socket
import threading
import pandas as pd
import os
import pickle
import torch

# Ayarlar
DATA_FILE = "data.csv"
MODEL_SAVE_DIR = "received_models"
VOCAB_SAVE_DIR = "received_vocabs"
AGG_MODEL_PATH = "model/aggregated_model.pt"
AGG_VOCAB_PATH = "vocab/aggregated_vocab.pkl"

# Gerekli dizinler
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)
if not os.path.exists(VOCAB_SAVE_DIR):
    os.makedirs(VOCAB_SAVE_DIR)
# Aggregated output dizinleri
if not os.path.exists(os.path.dirname(AGG_MODEL_PATH)):
    os.makedirs(os.path.dirname(AGG_MODEL_PATH), exist_ok=True)
if not os.path.exists(os.path.dirname(AGG_VOCAB_PATH)):
    os.makedirs(os.path.dirname(AGG_VOCAB_PATH), exist_ok=True)

clients = {}
models = {}
vocabs = {}

lock = threading.Lock()

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
        # tokens: liste veya iterable
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

# --- Dataset gönderme ---
def send_dataset(client_socket, df_slice):
    csv_str = df_slice.to_csv(index=False)
    try:
        client_socket.sendall(b"DATA_START")
        client_socket.sendall(csv_str.encode('utf-8'))
        client_socket.sendall(b"DATA_END")
        print(f"[+] Veri gönderildi: {clients[client_socket]['addr']}")
    except Exception as e:
        print(f"[!] Veri gönderme hatası: {e}")

# --- Broadcast datasets ---
def broadcast_datasets():
    global clients
    with lock:
        if not clients:
            print("[!] Bağlı client yok, veri gönderilemiyor.")
            return

        # Skorlara göre sırala
        sorted_clients = sorted(clients.items(), key=lambda x: x[1]["score"], reverse=True)
        total_score = sum([info["score"] for _, info in sorted_clients])
        print(f"[+] Toplam skor: {total_score}, Client sayısı: {len(sorted_clients)}")

        # CSV'i oku
        try:
            df = pd.read_csv(DATA_FILE)
        except Exception as e:
            print(f"[!] DATA_FILE okunurken hata: {e}")
            return

        total_len = len(df)
        start_idx = 0

        for idx, (client_socket, info) in enumerate(sorted_clients):
            ratio = info["score"] / total_score if total_score > 0 else 1/len(sorted_clients)
            length = int(total_len * ratio)
            if idx == len(sorted_clients) - 1:
                length = total_len - start_idx
            df_slice = df.iloc[start_idx:start_idx+length]
            start_idx += length

            clients[client_socket]["data_slice"] = df_slice
            print(f"[>] {info['addr']} skora göre veri oranı: %{ratio*100:.2f}, satır sayısı: {len(df_slice)}")

            send_dataset(client_socket, df_slice)

# --- Model alma ---
def receive_model(client_socket, filename, filesize):
    filepath = os.path.join(MODEL_SAVE_DIR, filename)
    print(f"[+] Model dosyası alımı başlıyor: {filename}, boyut: {filesize} byte")
    # Tam filesize kadar oku
    data = recv_exactly(client_socket, filesize)
    # Dosyaya yaz
    try:
        with open(filepath, "wb") as f:
            f.write(data)
    except Exception as e:
        print(f"[!] Model dosyası yazılırken hata: {e}")
        return False

    print(f"[✓] Model dosyası kaydedildi: {filepath} (okunan: {len(data)} byte)")
    # Torch load
    try:
        state_dict = torch.load(filepath, map_location=torch.device('cpu'))
        with lock:
            models[client_socket] = state_dict
        print(f"[✓] Model state_dict başarıyla yüklendi sözlüğe.")
        return True
    except Exception as e:
        print(f"[!] Model yükleme hatası: {e}")
        return False

# --- Vocab alma: tam size bazlı ---
def receive_vocab(client_socket, filename, filesize):
    filepath = os.path.join(VOCAB_SAVE_DIR, filename)
    print(f"[+] Vocab dosyası alınıyor: {filename}, boyut: {filesize} byte")
    data = recv_exactly(client_socket, filesize)
    try:
        with open(filepath, "wb") as f:
            f.write(data)
    except Exception as e:
        print(f"[!] Vocab dosyası yazılırken hata: {e}")
        return False
    print(f"[✓] Vocab dosyası kaydedildi: {filepath} (okunan: {len(data)} byte)")
    try:
        vocab_obj = pickle.loads(data)
        with lock:
            vocabs[client_socket] = vocab_obj
        print(f"[✓] Vocab nesnesi başarıyla yüklendi sözlüğe.")
    except Exception as e:
        print(f"[!] Vocab pickle yükleme hatası: {e} (ham veri saklandı)")
    return True

# --- Modelleri birleştir ---
def aggregate_models():
    global models
    if not models:
        print("[!] Birleştirilecek model yok.")
        return

    print(f"[+] {len(models)} model birleştiriliyor...")
    model_keys = list(next(iter(models.values())).keys())
    aggregated_state_dict = {}

    for key in model_keys:
        weights = [m[key] for m in models.values()]
        # Tensor toplanıp ortalama
        aggregated_state_dict[key] = sum(weights) / len(weights)

    try:
        torch.save(aggregated_state_dict, AGG_MODEL_PATH)
        print(f"[✓] Modeller birleştirildi ve '{AGG_MODEL_PATH}' olarak kaydedildi.")
    except Exception as e:
        print(f"[!] Birleştirilmiş model kaydedilirken hata: {e}")

# --- Vocabları birleştir ---
def aggregate_vocabs():
    print("[+] Vocablar birleştiriliyor...")
    all_vocabs = []
    for fname in os.listdir(VOCAB_SAVE_DIR):
        path = os.path.join(VOCAB_SAVE_DIR, fname)
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
                if isinstance(obj, Vocab):
                    all_vocabs.append(obj)
                else:
                    pass
        except Exception as e:
            print(f"[!] '{fname}' yüklenirken hata: {e}")
    if not all_vocabs:
        print("[!] Birleştirilecek vocab yok.")
        return
    merged = Vocab()
    for v in all_vocabs:
        for token in v.idx_to_token:
            merged.add_token(token)
    try:
        with open(AGG_VOCAB_PATH, "wb") as f:
            pickle.dump(merged, f)
        print(f"[✓] Vocablar birleştirildi ve '{AGG_VOCAB_PATH}' olarak kaydedildi.")
    except Exception as e:
        print(f"[!] Birleştirilmiş vocab kaydedilirken hata: {e}")

# --- Client handler ---
def handle_client(client_socket, addr):
    peer_id = f"{addr[0]}:{addr[1]}"
    print(f"[+] Bağlantı geldi: {peer_id}")
    clients[client_socket] = {"addr": peer_id, "score": 0.0, "data_slice": None}

    try:
        buffer = b""
        while True:
            header_bytes = b""
            while b"\n" not in header_bytes:
                chunk = client_socket.recv(1)
                if not chunk:
                    # bağlantı kapanmış
                    raise ConnectionError("Bağlantı kapandı")
                header_bytes += chunk
            header_line, sep, rest = header_bytes.partition(b"\n")
            header = header_line.decode(errors='ignore').strip()

            # İşle: header kontrolü
            if header.startswith("SCORE:"):
                # SCORE:<float>
                try:
                    score = float(header.split(":",1)[1])
                    with lock:
                        clients[client_socket]["score"] = score
                    print(f"{peer_id} skoru: {score}")
                    # Veri bölüştürmeyi tetikle
                    broadcast_datasets()
                except Exception as e:
                    print(f"[!] SCORE parse hatası: {e}")
                # rest tamamını text değilse ihmal et
                continue

            elif header.startswith("MODEL_UPLOAD:"):
                # MODEL_UPLOAD:filename:size
                parts = header.split(":", 2)
                if len(parts) != 3:
                    print(f"[!] Geçersiz MODEL_UPLOAD header: {header}")
                    continue
                _, filename, size_str = parts
                try:
                    filesize = int(size_str)
                except:
                    print(f"[!] MODEL_UPLOAD size parse edilemedi: {size_str}")
                    continue
                success = receive_model(client_socket, filename, filesize)
                if success:
                    aggregate_models()
                continue

            elif header.startswith("VOCAB_UPLOAD:"):
                parts = header.split(":", 2)
                if len(parts) != 3:
                    print(f"[!] Geçersiz VOCAB_UPLOAD header: {header}")
                    continue
                _, filename, size_str = parts
                try:
                    filesize = int(size_str)
                except:
                    print(f"[!] VOCAB_UPLOAD size parse edilemedi: {size_str}")
                    continue
                success = receive_vocab(client_socket, filename, filesize)
                if success:
                    aggregate_vocabs()
                continue

            else:
                print(f"{peer_id} MSG: {header}")
                continue

    except ConnectionError:
        print(f"[-] {peer_id} bağlantısı kapandı.")
    except Exception as e:
        print(f"[!] {peer_id} bağlantı hatası: {e}")
    finally:
        with lock:
            clients.pop(client_socket, None)
            models.pop(client_socket, None)
            vocabs.pop(client_socket, None)
        client_socket.close()
        print(f"[-] {peer_id} handler sonlandı.")

# --- Main ---
def main():
    server = socket.socket()
    server.bind(('0.0.0.0', 12345))
    server.listen(10)
    print("[✓] Sunucu dinliyor...")

    connected_clients = []

    def accept_clients():
        while True:
            client_socket, addr = server.accept()
            print(f"[+] Yeni client bağlandı: {addr}")
            connected_clients.append((client_socket, addr))

    accept_thread = threading.Thread(target=accept_clients, daemon=True)
    accept_thread.start()

    print("[i] Client bağlantıları için 5 saniye bekleniyor...")
    threading.Event().wait(5.0)

    print(f"[✓] {len(connected_clients)} client bağlandı. Eğitim başlatılıyor...")

    for client_socket, addr in connected_clients:
        threading.Thread(target=handle_client, args=(client_socket, addr), daemon=True).start()


if __name__ == "__main__":
    main()

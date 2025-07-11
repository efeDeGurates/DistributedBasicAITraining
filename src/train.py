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
from vocab import Vocab
from model import PositionalEncoding,TransformerChatbot


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
            self.log("Invalid port")
            return

        try:
            self.client_socket = socket.socket()
            self.client_socket.connect((ip, port))
            self.log(f"Connected to {ip}:{port}")

            score = 10.0
            self.client_socket.sendall(f"SCORE:{score}\n".encode())

            threading.Thread(target=receive_data_and_train, args=(self.client_socket, self.log), daemon=True).start()
        except Exception as e:
            self.log(f"Connection error: {e}")

    def send_command(self):
        cmd = self.cmd_entry.get().strip()
        if cmd and self.client_socket:
            try:
                self.client_socket.sendall((cmd + "\n").encode())
                self.cmd_entry.delete(0, tk.END)
            except:
                self.log("Failed to send command")

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
        print(f"File send error: {e}")


def _handle_training_and_send(client_socket, log_func):
    try:
        enc_in, enc_out, vocab = load_data_and_prepare_vocab("received_data.csv")
    except Exception as e:
        log_func(f"Data error: {e}")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerChatbot(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    log_func("Training started...")
    train(model, enc_in, enc_out, optimizer, loss_fn, vocab, log_func)
    torch.save(model.state_dict(), "trained_model.pt")
    save_vocab(vocab)
    log_func("Model saved. Sending to server...")
    send_file(client_socket, "trained_model.pt", "MODEL_UPLOAD")
    send_file(client_socket, "vocab_client.pkl", "VOCAB_UPLOAD")


def receive_data_and_train(client_socket, log_func):
    buffer = ""
    receiving = False
    while True:
        try:
            data = client_socket.recv(4096)
            if not data:
                log_func("Server disconnected.")
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
            log_func(f"Receive error: {e}")
            break


# --- Main ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ClientGUI(root)
    root.mainloop()

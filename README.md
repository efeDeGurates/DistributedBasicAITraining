﻿# Distributed Chatbot Training over P2P

Train your own AI chatbot with friends — no data sharing, just local training and model merging over the network.

---

## 🔧 What is this?

- Each client trains a Transformer chatbot on a private dataset.
- The server sends data slices, collects models & vocab files.
- It aggregates all models into a single smart brain

---

## Run in 3 Steps

### Start the Server

```bash
python server.py
```

### Start One or More Clients

```bash
python train.py
```

Each client trains locally and sends back a model.

### Chat with the AI

```bash
python runner.py
```

Ask anything, get smart replies. Internet features included.

---

## 🛠️ Requirements

```txt
torch
pandas
requests
beautifulsoup4
serpapi
```

Install:

```bash
pip install -r requirements.txt
```

---

## 🔍 Cool Features

- Transformer-based chatbot (PyTorch)
- Weighted model merging (like federated learning)
- Web integration: weather, news, search
- Beam Search decoding
- Auto-logs chat to CSV

---

## License

MIT — use freely and improve it!

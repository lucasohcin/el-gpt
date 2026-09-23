# El GPT 1.0 — Custom Neural Network Engine & ChatGPT Interface

Welcome to **El GPT 1.0** — an end-to-end, locally trained AI language model and ChatGPT-grade web application built for **Apple Silicon (M1/M2/M3/M4) Macs** using PyTorch Metal Performance Shaders (MPS).

---

## 🌟 Features

- **Modern Decoder-Only Transformer Architecture**:
  - **RoPE (Rotary Position Embeddings)**: Dynamic relative position encodings.
  - **RMSNorm**: Stable Root Mean Square layer normalization.
  - **SwiGLU Feed-Forward Networks**: High parameter efficiency activation.
  - **Flash/Metal SDPA Attention**: Causal multi-head attention accelerated on Apple Silicon GPU/MPS.
- **Custom Tokenizer**:
  - Byte-level fallback (256 byte tokens) ensuring 100% lossless encoding with zero out-of-vocabulary crashes.
  - Specialized tokens for roles (`<|system|>`, `<|user|>`, `<|assistant|>`), code, and math syntax.
- **Domain Specialization**:
  - **Math**: Step-by-step arithmetic, linear/quadratic equations, geometry, calculus, and LaTeX formulas.
  - **Coding**: Python, JavaScript, data structures, algorithms (binary search, sorting, palindromes), and async APIs.
  - **Daily Chat**: Conversational banter, persona self-identification as "El GPT 1.0", facts, and productivity.
- **ChatGPT-Style UI**:
  - Sleek dark theme matching ChatGPT's design system.
  - Real-time token streaming typewriter effect via Server-Sent Events (SSE).
  - Code blocks with syntax highlighting and one-click "Copy Code".
  - LaTeX math rendering with KaTeX (`$...$` and `$$...$$`).
  - Conversation history stored locally in browser.
- **Interactive Neural Training Studio**:
  - Live SVG loss curve visualization updating in real time.
  - Real-time telemetry: steps, epochs, tokens/sec, and GPU/MPS utilization.
  - Hyperparameter tuning: customize epochs, batch size, and learning rate directly from the UI.

---

## 🚀 Quick Start

### 1. Launch Server and UI
Run the one-click startup script:
```bash
./run.sh
```
This script will automatically prepare the environment, synthesize the dataset curriculum, and launch the server at:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 🏋️ Training El GPT 1.0

### Option A: From the ChatGPT Interface (Training Studio)
1. Open [http://localhost:8000](http://localhost:8000) in your browser.
2. Click **"Train Engine"** in the top right or **"Training Studio"** in the sidebar.
3. Configure your epochs (e.g. 5–10) and learning rate.
4. Click **"Launch Training Run"** and watch the loss curve drop in real time!

### Option B: From the Terminal
Train the model directly on your Apple M1 GPU via CLI:
```bash
.venv/bin/python engine/train.py --epochs 5 --batch-size 4 --lr 0.0005
```
Checkpoints will be saved automatically to `checkpoints/el_gpt_1.0_latest.pt`.

---

## 📂 Project Structure

```
El GPT/
├── .venv/                      # Python 3.11 virtual environment
├── requirements.txt            # PyTorch, FastAPI, Uvicorn, etc.
├── run.sh                      # One-click startup script
├── engine/                     # PyTorch Neural Network
│   ├── __init__.py
│   ├── model.py                # ElGPTDecoder (RoPE + RMSNorm + SwiGLU)
│   ├── tokenizer.py            # Universal Byte-Level Tokenizer
│   ├── dataset.py              # Loss-masked dialogue dataset loader
│   ├── inference.py            # Autoregressive generation & SSE streamer
│   └── train.py                # AdamW + Cosine LR training loop with MPS
├── data/                       # Training Curriculum
│   ├── prepare_datasets.py     # Math, Code, Chat dataset synthesizer
│   ├── train.jsonl             # Multi-domain training samples
│   └── val.jsonl               # Validation set
├── server/                     # Backend API
│   ├── __init__.py
│   ├── app.py                  # FastAPI app & SSE streaming route
│   └── training_runner.py      # Background training worker & telemetry
├── frontend/                   # ChatGPT-grade Web UI
│   ├── index.html              # ChatGPT layout & Training Studio modal
│   ├── styles.css              # Dark theme, modern spacing & typography
│   └── app.js                  # SSE stream consumer, KaTeX, Highlight.js
└── checkpoints/                # Trained weights and state dicts
    └── el_gpt_1.0_latest.pt
```

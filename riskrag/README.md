# 🏦 BFSI Risk Register AI — Local Query System

A fully **offline, local LLM-powered** query engine for your BFSI Risk Register.
Ask questions in plain English and get intelligent answers drawn from your 213-row risk database.

---

## Architecture

```
Your Question
     │
     ▼
[ Embedding Model ]  ←── runs locally via sentence-transformers
     │  (converts question to vector)
     ▼
[ FAISS Vector Store ] ←── your 213 risk records, pre-indexed
     │  (finds most relevant risks)
     ▼
[ Ollama LLM ]  ←── runs locally (Llama 3 / Mistral / Phi-3)
     │  (generates answer from matched context)
     ▼
Your Answer  (+ source Risk IDs cited)
```

**Everything runs on your PC — no internet required after setup.**

---

## Requirements

| Component | What it does | Size |
|-----------|-------------|------|
| Python 3.10+ | Runtime | — |
| Ollama | Runs the LLM locally | ~200 MB app |
| LLM model (e.g. `llama3.2:3b`) | Answers questions | ~2 GB |
| sentence-transformers | Embeds text for search | ~500 MB |
| FAISS / other libs | Vector similarity search | ~50 MB |

**Minimum hardware:** 8 GB RAM, any modern CPU (GPU optional but faster)

---

## Quick Setup (5 steps)

### Step 1 — Install Ollama
Download from **https://ollama.com** and install for your OS (Windows / Mac / Linux).

### Step 2 — Pull a local LLM
Open a terminal and run:
```bash
# Recommended: fast, good quality, runs on 8GB RAM
ollama pull llama3.2:3b

# Alternative: slightly larger but excellent at structured data Q&A
ollama pull mistral:7b

# Lightweight option for low-RAM PCs (4GB RAM)
ollama pull phi3:mini
```

### Step 3 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Build the vector index (one-time, ~30 seconds)
```bash
python build_index.py
```
This reads `risk_data.json`, embeds all 213 risks, and saves `risk_index.faiss` + `risk_chunks.pkl`.

### Step 5 — Start querying!
```bash
# Interactive chat mode
python query.py

# Web UI (recommended — runs at http://localhost:7860)
python app.py
```

---

## Example Questions You Can Ask

```
> What are the CRITICAL risks in my register?
> Show all compliance risks related to PMLA
> Which risks involve the treasury function?
> What cyber risks are escalated to BRC?
> List all risks where control effectiveness is Weak
> What is the total financial exposure for credit risks?
> Which risks have a risk appetite breach?
> Show me all incidents (Is Incident = Yes)
> What action plans are overdue?
> Summarise AML/KYC failures and their regulatory references
> Which risks are owned by the Compliance Officer?
> What are the top 5 highest inherent risk score items?
> Show risks related to RBI KYC Master Directions
> What are the liquidity risks in the Treasury function?
> Give me a Board-level risk summary
```

---

## File Structure

```
risk_register_ai/
├── README.md              ← This file
├── requirements.txt       ← Python dependencies
├── risk_data.json         ← Your 213 risk records (structured)
├── build_index.py         ← One-time: builds FAISS vector index
├── query.py               ← Command-line interactive chat
├── app.py                 ← Web UI (Gradio) — http://localhost:7860
├── rag_engine.py          ← Core RAG logic (shared by query.py + app.py)
└── risk_index/            ← Created by build_index.py
    ├── risk_index.faiss
    └── risk_chunks.pkl
```

---

## Changing the LLM Model

Edit `rag_engine.py` and change `MODEL_NAME`:
```python
MODEL_NAME = "llama3.2:3b"   # default
MODEL_NAME = "mistral:7b"    # better quality
MODEL_NAME = "phi3:mini"     # fastest / lowest RAM
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ollama: command not found` | Install Ollama from https://ollama.com |
| `model not found` | Run `ollama pull llama3.2:3b` |
| Out of memory | Use `phi3:mini` instead |
| Slow answers | Normal for CPU-only; GPU speeds it up 10× |
| `No module named faiss` | Run `pip install faiss-cpu` |

---

## Privacy

✅ All data stays on your PC  
✅ No API keys needed  
✅ No internet connection after setup  
✅ Your risk register never leaves your machine  

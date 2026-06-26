"""
rag_engine.py
=============
Core Retrieval-Augmented Generation engine.
Shared by query.py (CLI) and app.py (Web UI).
"""

import json, os, pickle, re
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME   = "llama3.2:3b"   # Change to "mistral:7b" or "phi3:mini" as needed
EMBED_MODEL  = "all-MiniLM-L6-v2"
TOP_K        = 6               # How many risk records to retrieve per query
INDEX_DIR    = "risk_index"

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior BFSI Risk Analyst AI assistant for a Small Bank / NBFC / Cooperative Bank in India.
You have been given a structured risk register containing identified risks with categories, scores, regulatory references, controls, action plans, financial impacts, and statuses.

Your job:
1. Answer questions about the risks in the register accurately and concisely.
2. Always cite the Risk IDs you are drawing from (e.g. "per NBFC-OPS-2025-001").
3. When summarising multiple risks, structure your answer with bullet points or a table.
4. For regulatory questions, cite the specific regulation referenced in the risk record.
5. If the question asks for aggregates (counts, totals), compute them from the provided context.
6. If the answer is not in the provided context, say so clearly — do not fabricate risk data.
7. Maintain a professional, board-ready tone appropriate for risk reporting.

Context risks are formatted as structured text blocks. Extract information precisely from them.
"""

# ── Load index (lazy, cached) ──────────────────────────────────────────────────
_index   = None
_chunks  = None
_risks   = None
_embed_model = None

def _load_resources():
    global _index, _chunks, _risks, _embed_model

    if _index is not None:
        return  # already loaded

    if not os.path.exists(f"{INDEX_DIR}/risk_index.faiss"):
        raise FileNotFoundError(
            "Index not found. Please run:  python build_index.py  first."
        )

    import faiss
    from sentence_transformers import SentenceTransformer

    _index = faiss.read_index(f"{INDEX_DIR}/risk_index.faiss")
    with open(f"{INDEX_DIR}/risk_chunks.pkl", "rb") as f:
        data = pickle.load(f)
    _chunks = data["chunks"]
    _risks  = data["risks"]
    _embed_model = SentenceTransformer(EMBED_MODEL)


def retrieve(query: str, top_k: int = TOP_K):
    """Embed query and retrieve top-k most relevant risk chunks."""
    _load_resources()

    import faiss
    q_emb = _embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_emb)
    scores, indices = _index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({
                "score":  float(score),
                "chunk":  _chunks[idx],
                "risk":   _risks[idx],
            })
    return results


def build_prompt(query: str, retrieved: list) -> list:
    """Build the messages list for the Ollama chat API."""
    context_blocks = []
    for i, r in enumerate(retrieved, 1):
        context_blocks.append(f"--- Risk Record {i} ---\n{r['chunk']}")

    context_text = "\n\n".join(context_blocks)

    user_message = f"""Use the following risk register records to answer the question.

RISK REGISTER CONTEXT:
{context_text}

QUESTION: {query}

Answer:"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]


def answer(query: str, top_k: int = TOP_K, model: str = MODEL_NAME) -> dict:
    """
    Full RAG pipeline: retrieve → build prompt → call Ollama → return result.

    Returns dict with keys:
        answer   : str
        sources  : list of risk dicts
        scores   : list of floats
    """
    import ollama

    retrieved = retrieve(query, top_k)
    if not retrieved:
        return {
            "answer": "No relevant risks found in the register for your query.",
            "sources": [],
            "scores": [],
        }

    messages = build_prompt(query, retrieved)

    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": 0.1, "num_predict": 1024},
    )

    return {
        "answer":  response["message"]["content"],
        "sources": [r["risk"]  for r in retrieved],
        "scores":  [r["score"] for r in retrieved],
    }


def answer_stream(query: str, top_k: int = TOP_K, model: str = MODEL_NAME):
    """
    Streaming version — yields text tokens as they arrive from Ollama.
    Also yields a final dict with sources when stream ends.
    """
    import ollama

    retrieved = retrieve(query, top_k)
    if not retrieved:
        yield "No relevant risks found in the register for your query."
        return

    messages = build_prompt(query, retrieved)

    stream = ollama.chat(
        model=model,
        messages=messages,
        stream=True,
        options={"temperature": 0.1, "num_predict": 1024},
    )

    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            yield token

    # Yield source metadata as a sentinel dict at the end
    yield {
        "__sources__": [r["risk"]  for r in retrieved],
        "__scores__":  [r["score"] for r in retrieved],
    }


# ── Analytical helpers (fast, no LLM needed) ──────────────────────────────────
def get_all_risks():
    _load_resources()
    return _risks

def stats_summary() -> str:
    """Quick statistics about the register — no LLM call needed."""
    _load_resources()
    risks = _risks

    total = len(risks)
    by_cat  = {}
    by_rating = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
    incidents = 0
    breaches  = 0
    total_fin = 0.0

    for r in risks:
        c = r.get("category", "Unknown")
        by_cat[c] = by_cat.get(c, 0) + 1
        rat = r.get("inherent_rating", "UNKNOWN")
        by_rating[rat] = by_rating.get(rat, 0) + 1
        if str(r.get("is_incident","")).strip().lower() == "yes":
            incidents += 1
        if str(r.get("risk_appetite_breach","")).strip().lower() == "yes":
            breaches += 1
        try:
            total_fin += float(r.get("financial_impact_lakhs", 0) or 0)
        except:
            pass

    lines = [
        f"📊 Risk Register Statistics",
        f"─────────────────────────────",
        f"Total Risks       : {total}",
        f"Incidents         : {incidents}",
        f"Appetite Breaches : {breaches}",
        f"Total Financial Exposure : ₹{total_fin:,.1f} Lakhs",
        f"",
        f"By Category:",
    ]
    for cat, cnt in sorted(by_cat.items(), key=lambda x: -x[1]):
        lines.append(f"  {cat:<30} {cnt:>4}")
    lines += [
        f"",
        f"By Inherent Risk Rating:",
        f"  CRITICAL : {by_rating['CRITICAL']}",
        f"  HIGH     : {by_rating['HIGH']}",
        f"  MEDIUM   : {by_rating['MEDIUM']}",
        f"  LOW      : {by_rating['LOW']}",
    ]
    return "\n".join(lines)

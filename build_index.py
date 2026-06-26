"""
build_index.py
==============
One-time script: reads risk_data.json, creates rich text chunks from each
risk record, embeds them with sentence-transformers, and saves a FAISS index.

Run once:
    python build_index.py
"""

import json, os, pickle, time
import numpy as np

def load_risks(path="risk_data.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_chunk(r: dict) -> str:
    """Convert a risk record into a rich text chunk for embedding."""
    fin = f"₹{r['financial_impact_lakhs']} Lakhs" if r.get('financial_impact_lakhs') else "Not estimated"
    icaap = f"₹{r['icaap_capital_lakhs']} Lakhs" if r.get('icaap_capital_lakhs') else "N/A"

    return f"""Risk ID: {r['id']}
Title: {r['title']}
Category: {r['category']}
Sub-Category: {r['sub_category']}
Business Function: {r['business_function']}
Institution Type: {r['institution_type']}
Branch / Unit: {r['branch']}
Description: {r['description']}
Risk Owner: {r['risk_owner']}
Regulatory Reference: {r['regulatory_reference']}
Is Incident: {r['is_incident']}
Inherent Likelihood: {r['inherent_likelihood']} | Inherent Impact: {r['inherent_impact']} | Inherent Score: {r['inherent_score']} | Inherent Rating: {r['inherent_rating']}
Control Effectiveness: {r['control_effectiveness']}
Residual Likelihood: {r['residual_likelihood']} | Residual Impact: {r['residual_impact']} | Residual Score: {r['residual_score']} | Residual Rating: {r['residual_rating']}
Risk Appetite Breach: {r['risk_appetite_breach']}
Risk Treatment: {r['risk_treatment']}
Action Plan: {r['treatment_action_plan']}
Action Owner: {r['action_owner']}
Target Completion Date: {r['target_date']}
Financial Impact: {fin}
ICAAP Capital Charge: {icaap}
Status: {r['status']}
CRO Comment: {r['cro_comment']}
Last Reviewed: {r['last_reviewed']} | Next Review Due: {r['next_review_due']}"""

def build_index():
    print("=" * 60)
    print("  BFSI Risk Register AI — Building Vector Index")
    print("=" * 60)

    # ── Load risk data ───────────────────────────────────────
    print("\n[1/4] Loading risk_data.json ...", end=" ")
    risks = load_risks()
    print(f"{len(risks)} records loaded.")

    # ── Build text chunks ────────────────────────────────────
    print("[2/4] Building text chunks ...", end=" ")
    chunks = [make_chunk(r) for r in risks]
    print(f"{len(chunks)} chunks created.")

    # ── Embed ────────────────────────────────────────────────
    print("[3/4] Loading embedding model (all-MiniLM-L6-v2) ...")
    print("      (first run downloads ~90 MB — subsequent runs are instant)")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    t0 = time.time()
    print("      Embedding all chunks ...", end=" ", flush=True)
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    print(f"done in {time.time()-t0:.1f}s  shape={embeddings.shape}")

    # ── Build FAISS index ────────────────────────────────────
    print("[4/4] Building FAISS index ...", end=" ")
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)   # inner-product (cosine after normalise)
    index.add(embeddings.astype(np.float32))

    os.makedirs("risk_index", exist_ok=True)
    faiss.write_index(index, "risk_index/risk_index.faiss")

    with open("risk_index/risk_chunks.pkl", "wb") as f:
        pickle.dump({"chunks": chunks, "risks": risks}, f)

    print(f"done.  {index.ntotal} vectors stored.")
    print("\n✅  Index built successfully!")
    print("    Files saved to: risk_index/")
    print("\nNext step: run  python app.py  to open the web UI")
    print("       or: run  python query.py  for command-line chat\n")

if __name__ == "__main__":
    build_index()

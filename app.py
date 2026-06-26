"""
app.py
======
Web UI for the BFSI Risk Register AI.
Runs at http://localhost:7860

Usage:
    python app.py
    python app.py --model mistral:7b --port 7860
"""

import argparse, json
import gradio as gr
from rag_engine import answer, stats_summary, get_all_risks, MODEL_NAME, TOP_K

# ── Preset example questions ───────────────────────────────────────────────────
EXAMPLES = [
    "What are all the CRITICAL inherent risk items in the register?",
    "Show all AML/KYC failures and their regulatory references",
    "Which risks involve the Treasury function and are rated HIGH or CRITICAL?",
    "List all cyber security risks that are escalated to BRC",
    "Which risks have control effectiveness marked as Weak and risk appetite breach as Yes?",
    "Summarise the top 5 financial exposure risks by estimated impact in ₹ Lakhs",
    "What action plans are assigned to the Compliance Officer?",
    "Show all incidents (Is Incident = Yes) with their status",
    "What are the key credit concentration risks in MSME lending?",
    "Give me a board-level executive summary of the risk landscape",
    "Which risks are related to RBI KYC Master Directions or PMLA?",
    "List all liquidity risks and their residual ratings",
    "What fraud risks exist in IT & Cyber and what controls are in place?",
    "Show operational risks with status Under Assessment or Escalated to BRC",
    "What are the climate/ESG risks and their financial implications?",
]

RATING_COLORS = {
    "CRITICAL": "#dc2626",
    "HIGH":     "#ea580c",
    "MEDIUM":   "#ca8a04",
    "LOW":      "#16a34a",
    "UNKNOWN":  "#6b7280",
}


def format_sources_html(sources, scores):
    if not sources:
        return ""
    rows = ""
    for src, sc in zip(sources, scores):
        rating = src.get("inherent_rating", "UNKNOWN")
        color  = RATING_COLORS.get(rating, "#6b7280")
        fin    = f"₹{src.get('financial_impact_lakhs', 0)} L"
        rows += f"""
        <tr>
          <td style="font-family:monospace;font-size:12px;white-space:nowrap">{src.get('id','')}</td>
          <td style="font-size:12px">{src.get('title','')[:55]}</td>
          <td style="font-size:12px">{src.get('category','')}</td>
          <td style="text-align:center">
            <span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:bold">{rating}</span>
          </td>
          <td style="font-size:12px;text-align:right">{fin}</td>
          <td style="font-size:11px;color:#666;text-align:right">{sc:.3f}</td>
        </tr>"""

    return f"""
    <div style="margin-top:16px">
      <p style="font-size:13px;font-weight:600;color:#374151;margin-bottom:6px">
        📌 Source Risk Records Retrieved
      </p>
      <table style="width:100%;border-collapse:collapse;font-family:sans-serif">
        <thead>
          <tr style="background:#f3f4f6;font-size:11px;color:#6b7280;text-transform:uppercase">
            <th style="text-align:left;padding:6px 8px">Risk ID</th>
            <th style="text-align:left;padding:6px 8px">Title</th>
            <th style="text-align:left;padding:6px 8px">Category</th>
            <th style="text-align:center;padding:6px 8px">Rating</th>
            <th style="text-align:right;padding:6px 8px">Fin. Impact</th>
            <th style="text-align:right;padding:6px 8px">Similarity</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


def chat_fn(query, history, model_name, top_k):
    if not query.strip():
        return history, "", ""

    try:
        result = answer(query, top_k=int(top_k), model=model_name)
        ans    = result["answer"]
        sources_html = format_sources_html(result["sources"], result["scores"])
    except Exception as e:
        ans = f"⚠️ Error: {str(e)}\n\nMake sure:\n1. Ollama is running (`ollama serve`)\n2. Model is pulled (`ollama pull {model_name}`)\n3. Index is built (`python build_index.py`)"
        sources_html = ""

    history = history or []
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": ans})
    return history, "", sources_html


def stats_fn():
    return stats_summary()


def build_app(default_model=MODEL_NAME, default_topk=TOP_K):

    css = """
    #chatbot { font-family: 'Inter', sans-serif; }
    #chatbot .message { font-size: 14px; line-height: 1.6; }
    .stat-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; font-family: monospace; font-size: 13px; }
    footer { display: none !important; }
    """

    with gr.Blocks(
        title="BFSI Risk Register AI",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=css,
    ) as demo:

        # ── Header ────────────────────────────────────────────
        gr.HTML("""
        <div style="background:linear-gradient(135deg,#1e3a5f 0%,#1e5799 100%);
                    padding:24px 32px;border-radius:12px;margin-bottom:20px;color:white">
          <h1 style="margin:0;font-size:24px;font-weight:700;letter-spacing:-0.5px">
            🏦 BFSI Risk Register AI
          </h1>
          <p style="margin:6px 0 0;font-size:14px;opacity:0.85">
            Ask questions about your 213-item risk register · Powered by local LLM (Ollama) · 100% offline
          </p>
        </div>
        """)

        with gr.Row():
            # ── Left column: chat ──────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Risk Register Chat",
                    elem_id="chatbot",
                    height=480,
                    
                )
                with gr.Row():
                    query_box = gr.Textbox(
                        placeholder="Ask about risks, regulations, owners, ratings, financial impact ...",
                        label="Your Question",
                        lines=2,
                        scale=5,
                    )
                    submit_btn = gr.Button("Ask ➤", variant="primary", scale=1)

                sources_box = gr.HTML(label="Sources")

                gr.HTML("<hr style='margin:12px 0;border-color:#e5e7eb'>")
                gr.Markdown("**Example questions — click to ask:**")
                gr.Examples(
                    examples=[[e] for e in EXAMPLES],
                    inputs=[query_box],
                    label="",
                )

            # ── Right column: settings + stats ────────────────
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Settings")
                model_dd = gr.Dropdown(
                    choices=["llama3.2:3b", "mistral:7b", "phi3:mini", "llama3.1:8b", "gemma2:2b"],
                    value=default_model,
                    label="LLM Model (Ollama)",
                    info="Must be pulled via: ollama pull <model>",
                )
                topk_sl = gr.Slider(
                    minimum=3, maximum=12, value=default_topk, step=1,
                    label="Records retrieved per query (Top-K)",
                    info="Higher = more context, slower",
                )
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

                gr.Markdown("### 📊 Register Stats")
                stats_out = gr.Textbox(
                    label="",
                    lines=22,
                    interactive=False,
                    elem_classes=["stat-box"],
                )
                stats_btn = gr.Button("Refresh Stats", variant="secondary")

        # ── Events ────────────────────────────────────────────
        history_state = gr.State([])

        def submit(q, hist, model, topk):
            return chat_fn(q, hist, model, topk)

        submit_btn.click(
            fn=submit,
            inputs=[query_box, history_state, model_dd, topk_sl],
            outputs=[chatbot, query_box, sources_box],
        ).then(
            fn=lambda h: h,
            inputs=[chatbot],
            outputs=[history_state],
        )

        query_box.submit(
            fn=submit,
            inputs=[query_box, history_state, model_dd, topk_sl],
            outputs=[chatbot, query_box, sources_box],
        ).then(
            fn=lambda h: h,
            inputs=[chatbot],
            outputs=[history_state],
        )

        clear_btn.click(
            fn=lambda: ([], [], ""),
            outputs=[chatbot, history_state, sources_box],
        )

        stats_btn.click(fn=stats_fn, outputs=[stats_out])

        # Auto-load stats on start
        demo.load(fn=stats_fn, outputs=[stats_out])

    return demo


def main():
    parser = argparse.ArgumentParser(description="BFSI Risk Register AI — Web UI")
    parser.add_argument("--model",  default=MODEL_NAME)
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--share",  action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  🏦 BFSI Risk Register AI — Web UI")
    print("="*60)
    print(f"  Model  : {args.model}")
    print(f"  URL    : http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop\n")

    app = build_app(default_model=args.model)
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()

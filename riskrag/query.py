"""
query.py
========
Command-line interactive chat for the BFSI Risk Register AI.

Usage:
    python query.py
    python query.py --model mistral:7b
    python query.py --top-k 8
"""

import argparse, sys, textwrap
from rag_engine import answer_stream, stats_summary, MODEL_NAME

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich import print as rprint
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None


BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║       🏦  BFSI Risk Register AI — Local Query System        ║
║          Powered by Ollama + FAISS + sentence-transformers  ║
╚══════════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Commands:
  /stats     — Show register statistics (no LLM call)
  /model X   — Switch LLM model (e.g. /model mistral:7b)
  /topk N    — Change number of retrieved records (default 6)
  /help      — Show this help
  /quit      — Exit

Example questions:
  > What are all CRITICAL risks?
  > Show AML/KYC failures with weak controls
  > Which risks involve the Treasury function?
  > List risks with financial impact above ₹100 Lakhs
  > Summarise cyber risks escalated to BRC
  > What are the top action items for the Compliance Officer?
"""


def print_sources(sources, scores):
    if not sources:
        return
    if RICH:
        table = Table(title="📌 Source Risk Records Retrieved", show_lines=True)
        table.add_column("Risk ID",   style="cyan",    no_wrap=True)
        table.add_column("Title",     style="white",   max_width=45)
        table.add_column("Category",  style="yellow",  no_wrap=True)
        table.add_column("Rating",    style="red",     no_wrap=True)
        table.add_column("Score",     style="dim",     no_wrap=True)
        for src, sc in zip(sources, scores):
            rating = src.get("inherent_rating", "")
            color  = {"CRITICAL": "red", "HIGH": "orange3", "MEDIUM": "yellow", "LOW": "green"}.get(rating, "white")
            table.add_row(
                src.get("id",""),
                src.get("title","")[:60],
                src.get("category",""),
                f"[{color}]{rating}[/{color}]",
                f"{sc:.3f}",
            )
        console.print(table)
    else:
        print("\n── Source Records ──")
        for src, sc in zip(sources, scores):
            print(f"  {src.get('id','')}  {src.get('title','')[:55]}  [{src.get('inherent_rating','')}]  sim={sc:.3f}")


def run_query(query: str, model: str, top_k: int):
    if RICH:
        console.print(f"\n[dim]🔍 Retrieving top-{top_k} records and generating answer...[/dim]\n")
        console.print("[bold green]Answer:[/bold green] ", end="")
    else:
        print(f"\n[Retrieving top-{top_k} records...]\n")
        print("Answer: ", end="", flush=True)

    sources = []
    scores  = []
    full_answer = []

    for token in answer_stream(query, top_k=top_k, model=model):
        if isinstance(token, dict) and "__sources__" in token:
            sources = token["__sources__"]
            scores  = token["__scores__"]
        else:
            print(token, end="", flush=True)
            full_answer.append(str(token))

    print("\n")
    print_sources(sources, scores)
    print()


def main():
    parser = argparse.ArgumentParser(description="BFSI Risk Register AI — CLI")
    parser.add_argument("--model",  default=MODEL_NAME, help="Ollama model name")
    parser.add_argument("--top-k",  type=int, default=6, help="Number of records to retrieve")
    args = parser.parse_args()

    model  = args.model
    top_k  = args.top_k

    print(BANNER)
    print(f"  Model : {model}   |   Top-K : {top_k}")
    print(f"  Type /help for commands, /quit to exit\n")

    while True:
        try:
            if RICH:
                query = console.input("[bold cyan]You ▶[/bold cyan] ").strip()
            else:
                query = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not query:
            continue

        if query.lower() in ("/quit", "/exit", "exit", "quit"):
            print("\nGoodbye! 👋\n")
            break
        elif query.lower() == "/help":
            print(HELP_TEXT)
        elif query.lower() == "/stats":
            print(stats_summary())
        elif query.lower().startswith("/model "):
            model = query[7:].strip()
            print(f"  Model switched to: {model}")
        elif query.lower().startswith("/topk "):
            try:
                top_k = int(query[6:].strip())
                print(f"  Top-K set to: {top_k}")
            except ValueError:
                print("  Invalid number.")
        else:
            try:
                run_query(query, model=model, top_k=top_k)
            except Exception as e:
                print(f"\n⚠️  Error: {e}")
                print("   Make sure Ollama is running and the model is pulled.")
                print(f"   Try: ollama pull {model}\n")


if __name__ == "__main__":
    main()

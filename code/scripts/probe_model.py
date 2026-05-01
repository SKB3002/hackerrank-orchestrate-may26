"""Quick model-swap probe: run the answerer with a candidate model on one row.

Usage:
    python -m code.scripts.probe_model --model openai/gpt-oss-20b --row 28
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from code.agent.answerer import _build_user_message, AnswererOutput, SYSTEM_PROMPT
from code.agent.gatekeeper import gate_sync
from code.agent.retrieval import HybridRetriever
from code.agent.schemas import TicketInput
from code.llm.groq_client import GroqClient
from code.llm.structured import complete_structured

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--input", type=Path, default=ROOT / "support_tickets" / "support_tickets.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.input.open(encoding="utf-8")))
    r = rows[args.row]
    ticket = TicketInput(
        issue=r.get("Issue", ""), subject=r.get("Subject", ""),
        company=r.get("Company", "None") or "None",
    )
    print(f"=== Row {args.row} ===")
    print(f"  Company: {ticket.normalized_company}")
    print(f"  Subject: {ticket.subject!r}")
    print(f"  Issue:   {ticket.issue!r}")

    print("\n[1] retrieving ...")
    retriever = HybridRetriever.load()
    hits = retriever.search(ticket.joined_text, top_k=5, vendor=ticket.normalized_company)
    for h in hits:
        print(f"   - rrf={h.rrf_score:.4f} [{h.chunk.vendor}/{h.chunk.topic}] {h.chunk.title[:80]}")

    print("\n[2] gatekeeper ...")
    client = GroqClient()
    gate, gtrace = gate_sync(client, ticket, hits)
    print(f"   should_escalate={gate.should_escalate} confidence={gate.confidence} request_type={gate.request_type.value} product_area={gate.product_area!r}")
    print(f"   reason: {gate.reason}")

    print(f"\n[3] answerer with {args.model} ...")
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(ticket, gate, hits)},
    ]
    obj, llm = complete_structured(client, args.model, msgs, AnswererOutput, max_tokens=1500)
    print(f"   model={llm.model} cached={llm.cached} latency_ms={llm.latency_ms:.0f}")
    print(f"   product_area: {obj.product_area!r}")
    print(f"   request_type: {obj.request_type.value}")
    print(f"   should_escalate: {obj.should_escalate}")
    print(f"   justification: {obj.justification}")
    print(f"\n   response:\n{obj.response}")


if __name__ == "__main__":
    main()

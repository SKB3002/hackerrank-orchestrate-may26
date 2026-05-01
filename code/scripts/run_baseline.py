"""Run Approach A baseline: input CSV → output CSV.

Usage:
    python -m code.scripts.run_baseline --input <in.csv> --output <out.csv>

Used in Phase 2 to establish floor scores against the sample CSV.
Also used as the fallback path if the LLM lane fails by h21.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from code.agent.baseline import BaselineAgent
from code.agent.schemas import FULL_OUTPUT_HEADER, TicketInput


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=1.5,
                    help="BM25 score below this triggers polite-refusal fallback")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.input.open(encoding="utf-8")))
    print(f"[baseline] loading corpus + building BM25 indexes...")
    agent = BaselineAgent(threshold=args.threshold)
    print(f"[baseline] indexed {len(agent.corpus)} docs across {len(agent._vendor_indexes)} vendors")

    out_rows = []
    for i, r in enumerate(rows):
        ticket = TicketInput(
            issue=r.get("Issue", ""),
            subject=r.get("Subject", ""),
            company=r.get("Company", "None") or "None",
        )
        result = agent.predict(ticket)
        out_rows.append(result.to_csv_row())
        if i < 3 or i % 10 == 0:
            print(f"  [{i:02d}] {ticket.normalized_company:10s} | {result.status.value:9s} | {result.request_type.value:15s} | {result.product_area!r}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FULL_OUTPUT_HEADER)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n[baseline] wrote {len(out_rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()

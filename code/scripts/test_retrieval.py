"""Smoke retrieval against the 10 sample tickets — prints top hits per row."""

from __future__ import annotations

import csv
from pathlib import Path

from code.agent.retrieval import HybridRetriever

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "support_tickets" / "sample_support_tickets.csv"


def main() -> None:
    print("[retrieval-smoke] loading index...")
    r = HybridRetriever.load()
    print(f"[retrieval-smoke] {len(r.chunks)} chunks loaded")

    rows = list(csv.DictReader(SAMPLE.open(encoding="utf-8")))
    for i, row in enumerate(rows):
        query = (row["Subject"].strip() + "\n\n" + row["Issue"].strip()).strip()
        company = (row["Company"] or "None").strip()
        gold_pa = row["Product Area"]
        hits = r.search(query, top_k=5, vendor=company)
        print(f"\n=== ROW {i} | company={company!r} | gold_PA={gold_pa!r} ===")
        print(f"   Q: {query[:120]!r}")
        for h in hits:
            print(f"   - rrf={h.rrf_score:.4f}  bm25_r={h.bm25_rank}  dense_r={h.dense_rank}  "
                  f"[{h.chunk.vendor}/{h.chunk.topic}] {h.chunk.title[:80]}")


if __name__ == "__main__":
    main()

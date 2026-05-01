"""One-shot inspector for the sample CSV. Phase 1 scaffolding."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "support_tickets" / "sample_support_tickets.csv"
TEST = ROOT / "support_tickets" / "support_tickets.csv"


def main() -> None:
    rows = list(csv.DictReader(SAMPLE.open(encoding="utf-8")))
    print(f"sample rows: {len(rows)}")
    print(f"columns: {list(rows[0].keys())}")
    print()
    print("Status:", Counter(r["Status"] for r in rows))
    print("Request Type:", Counter(r["Request Type"] for r in rows))
    print("Company:", Counter(r["Company"] for r in rows))
    print("Product Area:", Counter(r["Product Area"] for r in rows))
    print()
    for i, r in enumerate(rows):
        print(f"--- ROW {i} ---")
        print(f"  Company:      {r['Company']!r}")
        print(f"  Subject:      {r['Subject'][:120]!r}")
        print(f"  Issue:        {r['Issue'][:300]!r}")
        print(f"  Status:       {r['Status']!r}")
        print(f"  Product Area: {r['Product Area']!r}")
        print(f"  Request Type: {r['Request Type']!r}")
        print(f"  Response:     {r['Response'][:200]!r}")
        print()

    test_rows = list(csv.DictReader(TEST.open(encoding="utf-8")))
    print(f"=== test rows (unlabeled): {len(test_rows)} ===")
    print(f"columns: {list(test_rows[0].keys())}")
    print(f"Company: {Counter(r['Company'] for r in test_rows)}")


if __name__ == "__main__":
    main()

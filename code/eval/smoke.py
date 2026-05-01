"""Phase 1 smoke: prove the eval harness scores correctly.

Tests:
  1. Gold-vs-gold: every column should be 1.0 (or near it, justification depends on length).
  2. Empty-vs-gold: every column near zero.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from code.agent.schemas import FULL_OUTPUT_HEADER, JUSTIFICATION_COL
from code.eval.metrics import evaluate

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "support_tickets" / "sample_support_tickets.csv"


def _add_justification(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        nr = dict(r)
        # Synthesize a 60-char justification so the length-band check passes
        nr[JUSTIFICATION_COL] = "Auto: matches sample gold for harness self-test."
        out.append(nr)
    return out


def _write(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FULL_OUTPUT_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FULL_OUTPUT_HEADER})


def main() -> None:
    rows = list(csv.DictReader(GOLD.open(encoding="utf-8")))
    rows_with_just = _add_justification(rows)

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        # Test 1 — perfect predictions
        gold_path = td_p / "gold.csv"
        pred_path = td_p / "pred_perfect.csv"
        _write(rows_with_just, gold_path)
        _write(rows_with_just, pred_path)
        report = evaluate(gold_path, pred_path)
        print("=== TEST 1: gold-vs-gold (expect ~1.0 across the board) ===")
        print(report.summary())
        assert report.columns["Status"].score == 1.0, f"Status not 1.0: {report.columns['Status']}"
        assert report.columns["Request Type"].score == 1.0, f"Request Type not 1.0"
        assert report.columns["Product Area"].score >= 0.95, f"Product Area below 0.95: {report.columns['Product Area']}"
        print("[OK] Test 1 passed\n")

        # Test 2 — empty predictions
        empty_rows = []
        for r in rows_with_just:
            er = {k: r.get(k, "") for k in ["Issue", "Subject", "Company"]}
            er.update({"Response": "", "Product Area": "", "Status": "Replied", "Request Type": "invalid", JUSTIFICATION_COL: ""})
            empty_rows.append(er)
        empty_path = td_p / "pred_empty.csv"
        _write(empty_rows, empty_path)
        report2 = evaluate(gold_path, empty_path)
        print("=== TEST 2: empty-vs-gold (expect mostly low) ===")
        print(report2.summary())
        # Status: 9/10 are 'Replied', so always-replied gets 0.9 accuracy. That's fine.
        # Request Type: 7/10 product_issue, always-invalid gets 2/10 = 0.2.
        assert report2.columns["Request Type"].score < 0.3, f"Request Type too high on empty: {report2.columns['Request Type']}"
        assert report2.columns["Response"].score < 0.05, f"Response too high on empty: {report2.columns['Response']}"
        print("[OK] Test 2 passed\n")

    print("[ALL OK] Harness smoke tests pass.")


if __name__ == "__main__":
    main()

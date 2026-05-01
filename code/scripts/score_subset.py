"""Score a prediction CSV against the matching subset of the gold CSV (head N rows)."""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

from code.agent.schemas import FULL_OUTPUT_HEADER
from code.eval.metrics import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--gold", required=True, type=Path)
    args = ap.parse_args()

    pred = list(csv.DictReader(args.predictions.open(encoding="utf-8")))
    gold_full = list(csv.DictReader(args.gold.open(encoding="utf-8")))
    n = len(pred)
    gold_subset = gold_full[:n]

    # Add a Justification field to gold rows so the harness header check passes
    for r in gold_subset:
        r.setdefault("Justification", "")

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", suffix=".csv", delete=False) as f:
        w = csv.DictWriter(f, fieldnames=FULL_OUTPUT_HEADER)
        w.writeheader()
        for r in gold_subset:
            w.writerow({k: r.get(k, "") for k in FULL_OUTPUT_HEADER})
        gold_path = Path(f.name)

    report = evaluate(gold_path, args.predictions)
    print(report.summary())


if __name__ == "__main__":
    main()

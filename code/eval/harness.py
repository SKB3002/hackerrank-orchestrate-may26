"""Eval harness CLI.

Usage:
    python -m code.eval.harness --predictions <pred.csv> --gold <gold.csv> [--report <md>]

Prints a summary; optionally writes a markdown diff report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from code.eval import diff_report
from code.eval.metrics import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    report = evaluate(args.gold, args.predictions)
    print(report.summary())

    if args.report:
        diff_report.write(report, args.report)
        print(f"\nWrote diff report -> {args.report}")


if __name__ == "__main__":
    main()

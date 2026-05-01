"""One-off: dump a CSV's logical rows with concise per-row status."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    args = ap.parse_args()
    rows = list(csv.DictReader(args.path.open(encoding="utf-8")))
    print(f"logical rows: {len(rows)}")
    for i, r in enumerate(rows):
        comp = r.get("Company", "")
        st = r.get("Status", "")
        rt = r.get("Request Type", "")
        pa = r.get("Product Area", "")
        resp = r.get("Response", "")
        print(f"  [{i:02d}] {comp:10s} {st:9s} | {rt:15s} | PA={pa!r:25s} | resp_len={len(resp):4d}")


if __name__ == "__main__":
    main()

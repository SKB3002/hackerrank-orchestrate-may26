"""Dump a CSV row in full, including multi-line Response."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Force UTF-8 on Windows so en/em-dashes, vertical-ellipsis, etc. don't crash cp1252
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("indices", type=int, nargs="+")
    args = ap.parse_args()
    rows = list(csv.DictReader(args.path.open(encoding="utf-8")))
    for i in args.indices:
        if i >= len(rows):
            print(f"\n[row {i} out of range]")
            continue
        r = rows[i]
        print(f"\n{'=' * 70}\nROW {i}\n{'=' * 70}")
        print(f"Issue:        {r.get('Issue', '')[:300]}")
        print(f"Subject:      {r.get('Subject', '')[:300]}")
        print(f"Company:      {r.get('Company', '')}")
        print(f"Status:       {r.get('Status', '')}")
        print(f"Request Type: {r.get('Request Type', '')}")
        print(f"Product Area: {r.get('Product Area', '')!r}")
        print(f"Justification:")
        print(f"  {r.get('Justification', '')}")
        print(f"Response:")
        for line in (r.get("Response", "") or "").splitlines():
            print(f"  {line}")


if __name__ == "__main__":
    main()

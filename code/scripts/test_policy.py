"""Verify policy hits/misses on the 10 sample tickets.

Expected:
  Row 1 ('site is down') → outage_site_down (hard_escalate)
  Row 6 ('actor in Iron Man') → off_topic_trivia (force_invalid)
  Row 9 ('Thank you for helping me') → bare_greeting (force_invalid)
  All other rows: NO hit (passed through to LLM lane).
"""

from __future__ import annotations

import csv
from pathlib import Path

from code.agent.policy import evaluate, load_policy

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    rules = load_policy()
    print(f"[policy] {len(rules)} rules loaded\n")
    rows = list(csv.DictReader((ROOT / "support_tickets" / "sample_support_tickets.csv").open(encoding="utf-8")))
    for i, r in enumerate(rows):
        text = (r["Subject"] + "\n" + r["Issue"]).strip()
        hit = evaluate(text, rules)
        marker = f"[HIT: {hit.rule_id} -> {hit.action.value}]" if hit else "[no hit]"
        print(f"  [{i:02d}] {marker:55s} {text[:80]!r}")


if __name__ == "__main__":
    main()

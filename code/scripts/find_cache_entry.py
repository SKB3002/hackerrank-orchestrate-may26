"""Search the LLM cache for an entry matching a substring (debug helper)."""

from __future__ import annotations

import argparse
import json

from code.agent.cache import LLMCache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("substring")
    ap.add_argument("--max", type=int, default=3)
    args = ap.parse_args()

    c = LLMCache()
    n = 0
    for k in c.dc:
        v = c.dc[k]
        content = v.get("content", "") if isinstance(v, dict) else ""
        if args.substring in content:
            print(f"\n--- match {n} ---")
            print(content)
            n += 1
            if n >= args.max:
                break
    print(f"\n[total matches: {n}]")


if __name__ == "__main__":
    main()

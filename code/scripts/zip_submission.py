"""Build code.zip for HackerRank submission.

Includes:
  - all *.py under code/
  - code/policies/*.yaml + *.txt
  - code/llm/prompts/*.txt
  - code/index/manifest.json
  - code/README.md, code/requirements.txt
  - .env.example (in repo root, copied into the zip as code/.env.example)

Excludes:
  - data/, support_tickets/*.csv (corpus + inputs supplied by evaluator)
  - .venv/, __pycache__/, *.pyc
  - .env (local key)
  - code/index/{chunks.jsonl, bm25.pkl, dense.npy} (rebuilt from data/)
  - code/index/.llm_cache/ (per-run cache)
  - .git/, runs/

Usage: python -m code.scripts.zip_submission [--out PATH]
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"


# Patterns to skip when walking code/
EXCLUDE_DIRS = {"__pycache__", ".llm_cache", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
# Things we never ship (build artifacts that the evaluator will rebuild)
EXCLUDE_INDEX_FILES = {"chunks.jsonl", "bm25.pkl", "dense.npy"}


def _should_skip(p: Path) -> bool:
    parts = set(p.parts)
    if parts & EXCLUDE_DIRS:
        return True
    if p.suffix in EXCLUDE_SUFFIXES:
        return True
    if p.parent.name == "index" and p.name in EXCLUDE_INDEX_FILES:
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "submission" / "code.zip")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(CODE_DIR.rglob("*")):
            if not p.is_file():
                continue
            if _should_skip(p):
                continue
            arcname = p.relative_to(ROOT).as_posix()
            zf.write(p, arcname=arcname)
            n += 1
        # Also include .env.example so the evaluator can see what env vars we expect
        env_ex = ROOT / ".env.example"
        if env_ex.exists():
            zf.write(env_ex, arcname="code/.env.example")
            n += 1

    print(f"[zip] wrote {n} files -> {args.out}")
    # Show what's inside
    with zipfile.ZipFile(args.out) as zf:
        for name in zf.namelist():
            sz = zf.getinfo(name).file_size
            print(f"  {sz:>9d}  {name}")


if __name__ == "__main__":
    main()

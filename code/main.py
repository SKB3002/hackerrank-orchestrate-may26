"""CLI entry point — `python -m code.main`.

Thin wrapper around code.scripts.run_pipeline so the canonical entry point
is the one named in AGENTS.md / problem_statement.md.
"""

from __future__ import annotations

from code.scripts.run_pipeline import main

if __name__ == "__main__":
    main()

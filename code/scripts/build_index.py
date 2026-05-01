"""Build the hybrid retrieval index from data/. One-time on each corpus update.

Usage: python -m code.scripts.build_index
"""

from __future__ import annotations

from code.agent.retrieval import HybridRetriever


def main() -> None:
    HybridRetriever.build()


if __name__ == "__main__":
    main()

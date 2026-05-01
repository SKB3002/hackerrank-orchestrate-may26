"""Disk cache for LLM responses. Phase 4 deliverable.

Key = sha256(model + prompt_repr + temp + seed + response_format_repr).
Repeat calls during dev iteration are free.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import diskcache

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "code" / "index" / ".llm_cache"


def _key(model: str, messages: list[dict], temperature: float, seed: int,
         response_format: dict | None, max_tokens: int | None) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
        "response_format": response_format,
        "max_tokens": max_tokens,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class LLMCache:
    """Thin wrapper. Open one per process; re-open is fine, diskcache is process-safe."""

    def __init__(self, dir: Path = CACHE_DIR):
        dir.mkdir(parents=True, exist_ok=True)
        self.dc = diskcache.Cache(str(dir))

    def make_key(self, **kwargs: Any) -> str:
        return _key(**kwargs)

    def get(self, key: str) -> dict | None:
        return self.dc.get(key)  # returns None if missing

    def set(self, key: str, value: dict) -> None:
        self.dc.set(key, value)

    def close(self) -> None:
        self.dc.close()

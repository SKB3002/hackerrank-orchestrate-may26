"""Phase 4 verification: async Groq client, cache, structured output.

Tests:
  1. Sync structured: ask 8b for JSON {answer: str, confidence: float}, verify Pydantic parse.
  2. Cache hit: same call again should be cached=True with latency_ms=0.
  3. Async fan-out: 5 concurrent calls — verify all succeed under semaphore.
"""

from __future__ import annotations

import asyncio
import time

from pydantic import BaseModel, Field

from code.llm.groq_client import GATEKEEPER_MODEL, GroqClient
from code.llm.structured import acomplete_structured, complete_structured


class Probe(BaseModel):
    answer: str = Field(..., description="A one-word answer")
    confidence: float = Field(..., ge=0.0, le=1.0)


def test_sync_structured(c: GroqClient) -> None:
    print("\n[1] sync structured output ...")
    msgs = [
        {"role": "system", "content": "You answer questions and emit a single JSON object only. Schema: {\"answer\": string, \"confidence\": float in [0,1]}."},
        {"role": "user", "content": "What color is the sky on a clear day? Answer in one word."},
    ]
    obj, r = complete_structured(c, GATEKEEPER_MODEL, msgs, Probe, max_tokens=128)
    print(f"   parsed: {obj}")
    print(f"   model={r.model} cached={r.cached} latency_ms={r.latency_ms:.0f}")
    assert obj.confidence >= 0.0 and obj.confidence <= 1.0
    assert obj.answer.strip(), "answer empty"


def test_cache_hit(c: GroqClient) -> None:
    print("\n[2] cache hit on identical call ...")
    msgs = [
        {"role": "system", "content": "You answer questions and emit a single JSON object only. Schema: {\"answer\": string, \"confidence\": float in [0,1]}."},
        {"role": "user", "content": "What color is the sky on a clear day? Answer in one word."},
    ]
    obj, r = complete_structured(c, GATEKEEPER_MODEL, msgs, Probe, max_tokens=128)
    print(f"   parsed: {obj}")
    print(f"   cached={r.cached} latency_ms={r.latency_ms:.0f}")
    assert r.cached is True, f"expected cache hit, got cached={r.cached}"


async def test_async_fanout(c: GroqClient) -> None:
    print("\n[3] async fan-out (5 concurrent gatekeeper calls) ...")
    queries = [
        "Reply with one word: name a fruit.",
        "Reply with one word: name a color.",
        "Reply with one word: name a planet.",
        "Reply with one word: name a metal.",
        "Reply with one word: name an animal.",
    ]
    tasks = []
    for q in queries:
        msgs = [
            {"role": "system", "content": "You emit only a JSON object. Schema: {\"answer\": string, \"confidence\": float}"},
            {"role": "user", "content": q},
        ]
        tasks.append(acomplete_structured(c, GATEKEEPER_MODEL, msgs, Probe, max_tokens=64))
    t0 = time.perf_counter()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.perf_counter() - t0
    n_ok = sum(1 for r in results if not isinstance(r, Exception))
    n_fail = sum(1 for r in results if isinstance(r, Exception))
    print(f"   {n_ok}/5 ok, {n_fail}/5 failed, total {elapsed:.1f}s")
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"   [fail {i}] {type(r).__name__}: {str(r)[:160]}")
        else:
            obj, llm = r
            print(f"   [{i}] {obj.answer!r:20s} conf={obj.confidence:.2f}  cached={llm.cached}  lat={llm.latency_ms:.0f}ms")
    assert n_ok == 5, f"expected 5/5, got {n_ok}/5"


def main() -> None:
    c = GroqClient()
    test_sync_structured(c)
    test_cache_hit(c)
    asyncio.run(test_async_fanout(c))
    print("\n[ALL OK] Phase 4 client + cache + structured output verified.")


if __name__ == "__main__":
    main()

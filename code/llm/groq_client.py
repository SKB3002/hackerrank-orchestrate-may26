"""Groq client: sync + async, rate-limit-aware, retry on 429/5xx, cached.

Phase 4 final form. Phase 0 had a bare smoke-test stub; this replaces it.

Design:
  - One AsyncGroq per model (semaphore-limited concurrency per model)
  - tenacity retries: exp backoff (2..60s), max 6 attempts, on 429/5xx
  - Cache on top: sha256(model + messages + temp + seed + response_format)
  - Determinism: temp=0.2, seed=42 by default; both knobs forwarded to Groq.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from groq import AsyncGroq, Groq, RateLimitError, APIStatusError, APIConnectionError, APITimeoutError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from code.agent.cache import LLMCache

GATEKEEPER_MODEL = "llama-3.1-8b-instant"
ANSWERER_MODEL = "llama-3.3-70b-versatile"
# Fallback answerer when the primary 70b model hits its per-model TPD ceiling.
# Different model family on Groq → fresh quota pool. gpt-oss-20b is faster and
# a strong substitute on grounded-response tasks (smoke-tested on Row 28).
FALLBACK_ANSWERER_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = "qwen-2.5-32b"

DEFAULT_TEMPERATURE = 0.2
DEFAULT_SEED = 42

# Per-model concurrency caps. Groq free-tier RPM/TPM varies by model;
# we set conservative semaphores and let tenacity absorb 429s.
_PER_MODEL_LIMITS: dict[str, int] = {
    GATEKEEPER_MODEL: 10,
    ANSWERER_MODEL: 6,
    FALLBACK_ANSWERER_MODEL: 6,
    JUDGE_MODEL: 4,
}

_RETRY_EXC = (RateLimitError, APIStatusError, APIConnectionError, APITimeoutError)


@dataclass
class LLMResult:
    content: str
    model: str
    cached: bool
    latency_ms: float
    finish_reason: str | None
    raw: dict | None  # response.model_dump() for trace logs (excluded if cached)


def _client_sync() -> Groq:
    load_dotenv()
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    return Groq(api_key=key)


class GroqClient:
    """Async-first; sync wrapper provided for non-async callers."""

    def __init__(self, cache: LLMCache | None = None):
        load_dotenv()
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set")
        self._async = AsyncGroq(api_key=key)
        self._sync = Groq(api_key=key)
        self.cache = cache or LLMCache()
        self._semaphores: dict[str, asyncio.Semaphore] = {
            m: asyncio.Semaphore(n) for m, n in _PER_MODEL_LIMITS.items()
        }

    def _sem(self, model: str) -> asyncio.Semaphore:
        if model not in self._semaphores:
            self._semaphores[model] = asyncio.Semaphore(4)
        return self._semaphores[model]

    async def acomplete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = DEFAULT_TEMPERATURE,
        seed: int = DEFAULT_SEED,
        response_format: dict | None = None,
        max_tokens: int | None = 1024,
    ) -> LLMResult:
        # Cache check first — never enter the rate-limit lane on a hit
        key = self.cache.make_key(
            model=model, messages=messages, temperature=temperature, seed=seed,
            response_format=response_format, max_tokens=max_tokens,
        )
        cached = self.cache.get(key)
        if cached is not None:
            return LLMResult(
                content=cached["content"], model=model, cached=True,
                latency_ms=0.0, finish_reason=cached.get("finish_reason"), raw=None,
            )

        async with self._sem(model):
            t0 = time.perf_counter()
            async for attempt in AsyncRetrying(
                wait=wait_exponential_jitter(initial=2, max=60, jitter=2),
                stop=stop_after_attempt(6),
                retry=retry_if_exception_type(_RETRY_EXC),
                reraise=True,
            ):
                with attempt:
                    kwargs: dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "seed": seed,
                    }
                    if max_tokens is not None:
                        kwargs["max_tokens"] = max_tokens
                    if response_format is not None:
                        kwargs["response_format"] = response_format
                    resp = await self._async.chat.completions.create(**kwargs)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            choice = resp.choices[0]
            content = choice.message.content or ""
            finish_reason = choice.finish_reason
            self.cache.set(key, {"content": content, "finish_reason": finish_reason})
            return LLMResult(
                content=content, model=model, cached=False, latency_ms=latency_ms,
                finish_reason=finish_reason, raw=resp.model_dump(),
            )

    def complete(self, model: str, messages: list[dict], **kw: Any) -> LLMResult:
        """Sync wrapper. Useful for one-off scripts and unit tests.

        Cache check still happens; the network call is sync (no semaphore needed
        for a single call). Concurrency requires acomplete().
        """
        key = self.cache.make_key(
            model=model, messages=messages,
            temperature=kw.get("temperature", DEFAULT_TEMPERATURE),
            seed=kw.get("seed", DEFAULT_SEED),
            response_format=kw.get("response_format"),
            max_tokens=kw.get("max_tokens", 1024),
        )
        cached = self.cache.get(key)
        if cached is not None:
            return LLMResult(content=cached["content"], model=model, cached=True,
                             latency_ms=0.0, finish_reason=cached.get("finish_reason"), raw=None)

        t0 = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": model, "messages": messages,
            "temperature": kw.get("temperature", DEFAULT_TEMPERATURE),
            "seed": kw.get("seed", DEFAULT_SEED),
        }
        max_tokens = kw.get("max_tokens", 1024)
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        rf = kw.get("response_format")
        if rf is not None:
            kwargs["response_format"] = rf
        resp = self._sync.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        choice = resp.choices[0]
        content = choice.message.content or ""
        self.cache.set(key, {"content": content, "finish_reason": choice.finish_reason})
        return LLMResult(content=content, model=model, cached=False, latency_ms=latency_ms,
                         finish_reason=choice.finish_reason, raw=resp.model_dump())


# ---- legacy smoke-test entry point (Phase 0 verification, kept for compat) -----


def smoke_test() -> str:
    resp = _client_sync().chat.completions.create(
        model=GATEKEEPER_MODEL,
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        temperature=DEFAULT_TEMPERATURE,
        seed=DEFAULT_SEED,
        max_tokens=16,
    )
    return resp.choices[0].message.content or ""


if __name__ == "__main__":
    import sys
    try:
        out = smoke_test()
        print(f"[OK] Groq smoke test: {out!r}")
    except Exception as exc:
        print(f"[FAIL] Groq smoke test: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

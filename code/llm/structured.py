"""Structured output helper: ask Groq for JSON, validate with Pydantic, retry on failure.

Manual implementation — no `instructor` dependency. Cleaner debug path on Groq's
OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
import re
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from code.llm.groq_client import GroqClient, LLMResult

T = TypeVar("T", bound=BaseModel)

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_obj(text: str) -> str:
    """Best-effort extraction of the outermost JSON object from a free-form response.

    Groq usually obeys response_format={"type":"json_object"} but we belt-and-suspender.
    """
    text = text.strip()
    # Strip ```json fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    if text.startswith("{") and text.endswith("}"):
        return text
    m = _JSON_OBJ_RE.search(text)
    return m.group(0) if m else text


async def acomplete_structured(
    client: GroqClient,
    model: str,
    messages: list[dict],
    schema: Type[T],
    temperature: float = 0.2,
    seed: int = 42,
    max_tokens: int = 1024,
    max_retries: int = 2,
) -> tuple[T, LLMResult]:
    """Async: emit JSON, parse to Pydantic. Retries on parse/validation failure
    by appending a corrective system message.
    """
    msgs = list(messages)
    last_err: Exception | None = None
    last_result: LLMResult | None = None

    for attempt in range(max_retries + 1):
        result = await client.acomplete(
            model=model, messages=msgs, temperature=temperature, seed=seed,
            response_format={"type": "json_object"}, max_tokens=max_tokens,
        )
        last_result = result
        try:
            obj = _extract_json_obj(result.content)
            data = json.loads(obj)
            return schema.model_validate(data), result
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e
            if attempt >= max_retries:
                break
            # Add a corrective turn so the model can repair its output
            msgs = list(messages) + [
                {"role": "assistant", "content": result.content},
                {"role": "user", "content": (
                    f"Your previous reply did not parse as valid JSON for the required schema. "
                    f"Error: {type(e).__name__}: {str(e)[:300]}\n"
                    f"Reply with ONLY a single valid JSON object that matches the schema. No prose."
                )},
            ]

    # Out of retries — raise with the last error and last raw content for debug
    raise ValueError(f"structured output failed after {max_retries+1} attempts: "
                     f"{type(last_err).__name__}: {last_err}\nLast content: {last_result.content[:500] if last_result else '<no result>'}")


def complete_structured(
    client: GroqClient,
    model: str,
    messages: list[dict],
    schema: Type[T],
    temperature: float = 0.2,
    seed: int = 42,
    max_tokens: int = 1024,
    max_retries: int = 2,
) -> tuple[T, LLMResult]:
    """Sync version of acomplete_structured."""
    msgs = list(messages)
    last_err: Exception | None = None
    last_result: LLMResult | None = None

    for attempt in range(max_retries + 1):
        result = client.complete(
            model=model, messages=msgs, temperature=temperature, seed=seed,
            response_format={"type": "json_object"}, max_tokens=max_tokens,
        )
        last_result = result
        try:
            obj = _extract_json_obj(result.content)
            data = json.loads(obj)
            return schema.model_validate(data), result
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e
            if attempt >= max_retries:
                break
            msgs = list(messages) + [
                {"role": "assistant", "content": result.content},
                {"role": "user", "content": (
                    f"Your previous reply did not parse as valid JSON for the required schema. "
                    f"Error: {type(e).__name__}: {str(e)[:300]}\n"
                    f"Reply with ONLY a single valid JSON object that matches the schema. No prose."
                )},
            ]
    raise ValueError(f"structured output failed after {max_retries+1} attempts: "
                     f"{type(last_err).__name__}: {last_err}\nLast content: {last_result.content[:500] if last_result else '<no result>'}")

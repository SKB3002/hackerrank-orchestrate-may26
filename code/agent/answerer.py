"""Answerer LLM (70b). Phase 6.

Receives the ticket + gate verdict + top-5 retrieval hits WITH chunk bodies.
Emits a grounded response with mandatory [doc_id] citations + the final
TicketOutput fields (response, product_area, request_type, justification,
should_escalate).

The validator (next module) post-processes: enforces that every sentence
making a factual claim carries a [doc_id]; drops uncited sentences; if too
many sentences are dropped, force escalation.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from code.agent.retrieval import Hit
from code.agent.schemas import GateVerdict, RequestType, TicketInput
from code.llm.groq_client import ANSWERER_MODEL, FALLBACK_ANSWERER_MODEL, GroqClient
from code.llm.structured import acomplete_structured, complete_structured

PROMPT_PATH = Path(__file__).resolve().parent.parent / "llm" / "prompts" / "answerer.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

# Cap on context per chunk (chars, not tokens — quick approximation).
# 70b TPM on Groq free tier is the bottleneck; 5 chunks * 2400 chars ≈ 3000 tokens of context.
CHUNK_CHAR_BUDGET = 2400
MAX_HITS = 5


class AnswererOutput(BaseModel):
    response: str
    product_area: str = Field(default="")
    request_type: RequestType
    justification: str = Field(default="")
    should_escalate: bool = Field(default=False)


def _format_passages(hits: list[Hit]) -> str:
    if not hits:
        return "(NO PASSAGES — the corpus has no good match for this query)"
    blocks = []
    for h in hits[:MAX_HITS]:
        body = h.chunk.text
        if len(body) > CHUNK_CHAR_BUDGET:
            body = body[:CHUNK_CHAR_BUDGET] + " ..."
        crumbs = " > ".join(h.chunk.breadcrumbs) if h.chunk.breadcrumbs else ""
        blocks.append(
            f"--- doc_id: {h.chunk.doc_id}\n"
            f"vendor: {h.chunk.vendor}  |  topic: {h.chunk.topic}\n"
            f"title: {h.chunk.title}\n"
            f"breadcrumbs: {crumbs}\n"
            f"passage:\n{body}\n"
        )
    return "\n".join(blocks)


def _build_user_message(ticket: TicketInput, gate: GateVerdict, hits: list[Hit]) -> str:
    return (
        f"COMPANY: {ticket.normalized_company}\n"
        f"SUBJECT: {ticket.subject.strip() or '(blank)'}\n"
        f"ISSUE:\n{ticket.issue.strip()}\n\n"
        f"GATEKEEPER ROUTING:\n"
        f"  request_type: {gate.request_type.value}\n"
        f"  product_area: {gate.product_area or ''}\n"
        f"  reason: {gate.reason}\n"
        f"  confidence: {gate.confidence:.2f}\n\n"
        f"RETRIEVED PASSAGES (cite by doc_id):\n{_format_passages(hits)}\n\n"
        f"Emit a single JSON object per the schema in the system message. "
        f"Cite [doc_id] after each factual sentence. No prose outside the JSON."
    )


def answer_sync(client: GroqClient, ticket: TicketInput, gate: GateVerdict,
                hits: list[Hit], model: str | None = None) -> tuple[AnswererOutput, dict]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(ticket, gate, hits)},
    ]
    use_model = model or ANSWERER_MODEL
    try:
        obj, llm_result = complete_structured(
            client=client, model=use_model, messages=msgs, schema=AnswererOutput,
            max_tokens=1500,
        )
    except Exception as exc:
        # On rate-limit (or any other failure with the primary model), fall
        # back to a different family. This matters when the per-model TPD
        # ceiling on llama-3.3-70b-versatile is hit but we still want to
        # serve the row with high-quality output. Caller still sees a clean
        # AnswererOutput; only the model name in the trace changes.
        from groq import RateLimitError
        if isinstance(exc, RateLimitError) and model is None:
            obj, llm_result = complete_structured(
                client=client, model=FALLBACK_ANSWERER_MODEL, messages=msgs,
                schema=AnswererOutput, max_tokens=1500,
            )
        else:
            raise
    trace = {"model": llm_result.model, "cached": llm_result.cached,
             "latency_ms": llm_result.latency_ms, "finish_reason": llm_result.finish_reason}
    return obj, trace


async def answer_async(client: GroqClient, ticket: TicketInput, gate: GateVerdict, hits: list[Hit]) -> tuple[AnswererOutput, dict]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(ticket, gate, hits)},
    ]
    obj, llm_result = await acomplete_structured(
        client=client, model=ANSWERER_MODEL, messages=msgs, schema=AnswererOutput,
        max_tokens=1500,
    )
    trace = {"model": llm_result.model, "cached": llm_result.cached,
             "latency_ms": llm_result.latency_ms, "finish_reason": llm_result.finish_reason}
    return obj, trace

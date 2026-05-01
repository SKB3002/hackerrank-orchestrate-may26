"""Gatekeeper LLM (8b). Phase 5.

Sees: ticket text + top-5 retrieval hits (titles + topics, NOT full chunk bodies).
Emits: GateVerdict {should_escalate, confidence, request_type, product_area, reason}.

The gatekeeper does NOT see the corpus content — it only routes. The answerer
does the content work. This separation is the architectural defense in the
interview ('why two agents'): the gatekeeper is cheap, fast, and replicable;
the answerer is expensive but only runs on tickets that survive the gate.
"""

from __future__ import annotations

from pathlib import Path

from code.agent.retrieval import Hit
from code.agent.schemas import GateVerdict, RequestType, TicketInput
from code.llm.groq_client import GATEKEEPER_MODEL, GroqClient
from code.llm.structured import acomplete_structured, complete_structured

PROMPT_PATH = Path(__file__).resolve().parent.parent / "llm" / "prompts" / "gatekeeper.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")


def _format_hits(hits: list[Hit]) -> str:
    if not hits:
        return "(no retrieval hits — corpus has nothing relevant)"
    lines = []
    for i, h in enumerate(hits):
        lines.append(f"  {i+1}. [{h.chunk.vendor}/{h.chunk.topic}] {h.chunk.title}  "
                     f"(rrf={h.rrf_score:.4f}, bm25_r={h.bm25_rank}, dense_r={h.dense_rank})")
    return "\n".join(lines)


def _build_user_message(ticket: TicketInput, hits: list[Hit]) -> str:
    company = ticket.normalized_company
    subj = ticket.subject.strip() or "(blank)"
    issue = ticket.issue.strip()
    return (
        f"COMPANY: {company}\n"
        f"SUBJECT: {subj}\n"
        f"ISSUE:\n{issue}\n\n"
        f"TOP RETRIEVAL HITS:\n{_format_hits(hits)}\n\n"
        f"Emit a single JSON object per the schema described in the system message. "
        f"No prose."
    )


_SELF_SERVICE_RE = __import__("re").compile(
    r"self[-\s]?service|self[-\s]?serve|user\s+can\s+(do|perform|complete)|"
    r"steps\s+(available|provided|are)\s|user\s+can\s+follow|can\s+be\s+done\s+via",
    __import__("re").IGNORECASE,
)


def _normalize_verdict(v: GateVerdict) -> GateVerdict:
    """Light cleanup:
      - strip vendor prefix from product_area
      - if the gate's `reason` says self-service exists, override should_escalate→False
        (the 8b model sometimes contradicts its own reasoning)
    """
    pa = (v.product_area or "").strip().lower()
    for vendor in ("hackerrank_", "claude_", "visa_"):
        if pa.startswith(vendor):
            pa = pa[len(vendor):]

    should_escalate = v.should_escalate
    if should_escalate and _SELF_SERVICE_RE.search(v.reason or ""):
        # Reason contradicts action — trust the reasoning, demote to reply.
        should_escalate = False

    return v.model_copy(update={"product_area": pa, "should_escalate": should_escalate})


def gate_sync(client: GroqClient, ticket: TicketInput, hits: list[Hit]) -> tuple[GateVerdict, dict]:
    """Sync wrapper. Returns (verdict, trace_info)."""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(ticket, hits)},
    ]
    verdict, llm_result = complete_structured(
        client=client, model=GATEKEEPER_MODEL, messages=msgs, schema=GateVerdict,
        max_tokens=512,
    )
    verdict = _normalize_verdict(verdict)
    trace = {
        "model": llm_result.model, "cached": llm_result.cached,
        "latency_ms": llm_result.latency_ms, "finish_reason": llm_result.finish_reason,
    }
    return verdict, trace


async def gate_async(client: GroqClient, ticket: TicketInput, hits: list[Hit]) -> tuple[GateVerdict, dict]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(ticket, hits)},
    ]
    verdict, llm_result = await acomplete_structured(
        client=client, model=GATEKEEPER_MODEL, messages=msgs, schema=GateVerdict,
        max_tokens=512,
    )
    verdict = _normalize_verdict(verdict)
    trace = {
        "model": llm_result.model, "cached": llm_result.cached,
        "latency_ms": llm_result.latency_ms, "finish_reason": llm_result.finish_reason,
    }
    return verdict, trace

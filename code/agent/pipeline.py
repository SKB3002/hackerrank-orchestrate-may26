"""End-to-end pipeline: policy → retrieve → gate → answer → validate → trace.

Phase 8 module. Pre-policy hook is added in Phase 7; for now policy is None.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from code.agent.answerer import answer_async, answer_sync
from code.agent.gatekeeper import gate_async, gate_sync
from code.agent.policy import CompiledRule, apply_policy, evaluate as evaluate_policy, load_policy
from code.agent.retrieval import HybridRetriever, Hit
from code.agent.schemas import (
    GateVerdict,
    JUSTIFICATION_COL,
    RequestType,
    Status,
    TicketInput,
    TicketOutput,
)
from code.agent.validator import ValidationResult, validate
from code.llm.groq_client import GroqClient


@dataclass
class PipelineTrace:
    ticket_idx: int
    company: str
    issue_preview: str
    policy_hits: list[str] = field(default_factory=list)
    n_retrieved: int = 0
    retrieved_doc_ids: list[str] = field(default_factory=list)
    gate: dict | None = None
    answer: dict | None = None
    validation: dict | None = None
    final: dict = field(default_factory=dict)
    latency_ms: dict = field(default_factory=dict)
    cache_hits: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    output: TicketOutput
    trace: PipelineTrace


# Tunable: low gate confidence still answers but with reduced expectations.
# Pure escalate-on-low-confidence was rejected during planning because the gold
# escalation rate is ~10% — the gatekeeper's escalation logic comes from rules,
# not threshold. We DO use the threshold to widen the retrieval-fallback policy.
LOW_GATE_CONF = 0.30


class Pipeline:
    def __init__(self, retriever: HybridRetriever, client: GroqClient,
                 policy_rules: list[CompiledRule] | None = None):
        self.retriever = retriever
        self.client = client
        self.policy_rules = policy_rules if policy_rules is not None else load_policy()

    def _retrieve(self, ticket: TicketInput) -> list[Hit]:
        query = ticket.joined_text
        return self.retriever.search(query, top_k=5, vendor=ticket.normalized_company)

    @staticmethod
    def _make_polite_refusal(req_type: RequestType, text: str) -> str:
        t = (text or "").lower()
        if req_type == RequestType.INVALID:
            if any(w in t for w in ("thank", "thanks", "appreciate", "grateful")):
                return "Happy to help"
            return "I am sorry, this is out of scope from my capabilities."
        return ("I couldn't find a confident answer for your question in our support documentation. "
                "Please reach out to our support team for personalized help.")

    @staticmethod
    def _build_output_from_escalate(ticket: TicketInput, gate: GateVerdict | None,
                                     reason: str = "") -> TicketOutput:
        return TicketOutput(
            issue=ticket.issue, subject=ticket.subject, company=ticket.company,
            response="Escalate to a human",
            product_area="" if not gate else (gate.product_area or ""),
            status=Status.ESCALATED,
            request_type=gate.request_type if gate else RequestType.BUG,
            justification=reason or (gate.reason if gate else "Escalated by policy or pipeline."),
        )

    def run_sync(self, ticket: TicketInput, idx: int = 0) -> PipelineResult:
        trace = PipelineTrace(
            ticket_idx=idx, company=ticket.normalized_company,
            issue_preview=ticket.issue[:140],
        )
        latency: dict[str, float] = {}
        cache_hits: dict[str, bool] = {}

        # ---- 0. policy pre-gate ------------------------------------------
        t0 = time.perf_counter()
        policy_hit = evaluate_policy(ticket.joined_text, self.policy_rules)
        latency["policy_ms"] = (time.perf_counter() - t0) * 1000.0
        if policy_hit is not None:
            trace.policy_hits.append(f"{policy_hit.rule_id}:{policy_hit.action.value}")
            out = apply_policy(ticket, policy_hit, self.policy_rules)
            trace.final = out.to_csv_row() | {"_short_circuit": f"policy:{policy_hit.rule_id}"}
            trace.latency_ms = latency
            trace.cache_hits = cache_hits
            return PipelineResult(output=out, trace=trace)

        # ---- 1. retrieval -------------------------------------------------
        t0 = time.perf_counter()
        hits = self._retrieve(ticket)
        latency["retrieve_ms"] = (time.perf_counter() - t0) * 1000.0
        trace.n_retrieved = len(hits)
        trace.retrieved_doc_ids = [h.chunk.doc_id for h in hits]

        # ---- 2. gatekeeper ------------------------------------------------
        t0 = time.perf_counter()
        gate, gtrace = gate_sync(self.client, ticket, hits)
        latency["gate_ms"] = (time.perf_counter() - t0) * 1000.0
        cache_hits["gate"] = gtrace.get("cached", False)
        trace.gate = {
            "should_escalate": gate.should_escalate, "confidence": gate.confidence,
            "request_type": gate.request_type.value, "product_area": gate.product_area,
            "reason": gate.reason,
        }

        # ---- 2b. gatekeeper-driven escalate -----------------------------
        # Gold escalation rate is ~10% so we treat the gate's escalate signal
        # cautiously: only honor it if the gate is reasonably confident OR if
        # retrieval came up empty (no docs to ground a reply on).
        if gate.should_escalate and (gate.confidence >= 0.5 or not hits):
            out = self._build_output_from_escalate(ticket, gate)
            trace.final = out.to_csv_row() | {"_escalated_by": "gatekeeper"}
            trace.latency_ms = latency
            trace.cache_hits = cache_hits
            return PipelineResult(output=out, trace=trace)
        # If gate said escalate but with low confidence and we have hits,
        # demote to "let the answerer try" — a low-confidence flag is a hint,
        # not a verdict. Track in trace for interview defense.
        if gate.should_escalate:
            trace.policy_hits.append("low_conf_escalate_demoted_to_answer")

        # If retrieval found nothing AND gate confidence is low → polite refusal
        if not hits or gate.confidence < LOW_GATE_CONF:
            response = self._make_polite_refusal(gate.request_type, ticket.joined_text)
            out = TicketOutput(
                issue=ticket.issue, subject=ticket.subject, company=ticket.company,
                response=response,
                product_area=gate.product_area if gate.request_type != RequestType.INVALID else "",
                status=Status.REPLIED,
                request_type=gate.request_type,
                justification=f"Low retrieval/confidence (hits={len(hits)}, gate_conf={gate.confidence:.2f}). Returned polite refusal.",
            )
            trace.final = out.to_csv_row() | {"_short_circuit": "low_confidence"}
            trace.latency_ms = latency
            trace.cache_hits = cache_hits
            return PipelineResult(output=out, trace=trace)

        # ---- 3. answerer --------------------------------------------------
        t0 = time.perf_counter()
        ans, atrace = answer_sync(self.client, ticket, gate, hits)
        latency["answer_ms"] = (time.perf_counter() - t0) * 1000.0
        cache_hits["answer"] = atrace.get("cached", False)
        trace.answer = {
            "response_len": len(ans.response), "product_area": ans.product_area,
            "request_type": ans.request_type.value, "should_escalate": ans.should_escalate,
        }

        # If the answerer itself wants to escalate
        if ans.should_escalate:
            out = self._build_output_from_escalate(ticket, gate,
                                                    reason=ans.justification or "Answerer flagged unanswerable.")
            trace.final = out.to_csv_row() | {"_escalated_by": "answerer"}
            trace.latency_ms = latency
            trace.cache_hits = cache_hits
            return PipelineResult(output=out, trace=trace)

        # Normalize "(none)" / "none" / "n/a" to empty
        ans_pa = (ans.product_area or "").strip()
        if ans_pa.lower() in {"(none)", "none", "n/a", "null"}:
            ans = ans.model_copy(update={"product_area": ""})

        # ---- 4. validator -------------------------------------------------
        valid_ids = set(h.chunk.doc_id for h in hits)
        t0 = time.perf_counter()
        v: ValidationResult = validate(ans.response, valid_ids)
        latency["validate_ms"] = (time.perf_counter() - t0) * 1000.0
        trace.validation = {
            "n_sentences": v.n_sentences, "n_cited": v.n_cited, "n_dropped": v.n_dropped,
            "drop_rate": round(v.drop_rate, 2), "cited_doc_ids": v.cited_doc_ids,
            "forced_escalate": v.forced_escalate,
        }

        if v.forced_escalate or not v.cleaned_response:
            # Validator forced an escalation due to too many uncited claims
            response = self._make_polite_refusal(ans.request_type, ticket.joined_text)
            out = TicketOutput(
                issue=ticket.issue, subject=ticket.subject, company=ticket.company,
                response=response,
                product_area=ans.product_area or gate.product_area or "",
                status=Status.REPLIED,
                request_type=ans.request_type,
                justification=f"Answer dropped {v.n_dropped}/{v.n_sentences} sentences (uncited/hallucinated); fell back to polite refusal.",
            )
        else:
            out = TicketOutput(
                issue=ticket.issue, subject=ticket.subject, company=ticket.company,
                response=v.cleaned_response,
                product_area=ans.product_area or gate.product_area or "",
                status=Status.REPLIED,
                request_type=ans.request_type,
                justification=(ans.justification or "")[:400],
            )

        trace.final = out.to_csv_row()
        trace.latency_ms = latency
        trace.cache_hits = cache_hits
        return PipelineResult(output=out, trace=trace)

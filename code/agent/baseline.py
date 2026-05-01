"""Approach A — pure deterministic baseline. No LLM.

Pipeline:
  1. Read ticket. Normalize company.
  2. Filter corpus by vendor (None → all).
  3. BM25 over (title + breadcrumbs) of remaining docs. Top doc wins.
  4. product_area = top doc's topic dir; status = Replied unless an
     outage rule fires; request_type = keyword classifier.
  5. response = top doc's first paragraph + corpus citation; if BM25
     score is below threshold, fall back to a polite refusal template.

Wins points by:
  - Beating the dumb-baseline floor on Response and Product Area
  - Demonstrating the corpus is useful (interview defense)
  - Acting as the floor + fallback if the LLM lane fails by h21
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from code.agent.corpus import Doc, load_corpus, vendor_filter
from code.agent.schemas import (
    JUSTIFICATION_COL,
    RequestType,
    Status,
    TicketInput,
    TicketOutput,
)

# Outage / "site down" patterns → escalate as bug. Single gold-Escalated example
# in the sample is "site is down & none of the pages are accessible" (bug, no Product Area).
_OUTAGE_PATTERNS = re.compile(
    r"\b(site\s+(is\s+)?down|outage|cannot\s+access|none\s+of\s+the\s+pages|"
    r"server\s+down|all\s+pages\s+(are\s+)?broken|website\s+(is\s+)?down)\b",
    re.IGNORECASE,
)

# Bug language (without being an outage)
_BUG_PATTERNS = re.compile(
    r"\b(error|crash|broken|not\s+working|won'?t\s+load|stuck|frozen|"
    r"failed|fails|bug|glitch|issue\s+with|doesn'?t\s+work)\b",
    re.IGNORECASE,
)

# Feature-request language
_FEATURE_PATTERNS = re.compile(
    r"\b(could\s+you\s+(add|build|provide)|please\s+add|feature\s+request|"
    r"would\s+like\s+(to\s+see|a\s+feature)|wish\s+(you|there)|"
    r"is\s+there\s+a\s+way\s+to)\b",
    re.IGNORECASE,
)

# Invalid / off-topic / greeting. We only treat a message as `invalid` if
# it is SHORT (greeting / off-topic) — a long ticket that opens with "Hi there"
# is a real question. Compare against word count, not just regex match.
_INVALID_PATTERNS = re.compile(
    r"\b(thank\s+you|thanks(\s+a\s+lot)?|happy\s+to|name\s+of\s+the\s+actor|"
    r"who\s+is\s+the\s+(actor|president)|what'?s\s+the\s+weather|"
    r"tell\s+me\s+a\s+joke)\b",
    re.IGNORECASE,
)
_INVALID_MAX_WORDS = 12  # short messages only — long tickets aren't invalid even if they say 'thanks'

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize_topic(topic: str, vendor: str) -> str:
    """Strip the redundant '<vendor>_' prefix some directories carry.

    e.g. data/hackerrank/hackerrank_community/  →  product_area should be 'community'.
    """
    t = topic.strip().lower()
    pref = f"{vendor.strip().lower()}_"
    if t.startswith(pref):
        return t[len(pref):]
    return topic


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class _IndexedCorpus:
    docs: list[Doc]
    bm25: BM25Okapi


def _build_index(docs: list[Doc]) -> _IndexedCorpus:
    # Index against header_text (title + breadcrumbs) — highest signal per token.
    docs = [d for d in docs if d.title and d.title != "index"]
    tokenized = [_tokenize(d.header_text + " " + d.body[:500]) for d in docs]
    return _IndexedCorpus(docs=docs, bm25=BM25Okapi(tokenized))


def _classify_request_type(text: str) -> RequestType:
    n_words = len(text.split())
    if n_words <= _INVALID_MAX_WORDS and _INVALID_PATTERNS.search(text):
        return RequestType.INVALID
    if _OUTAGE_PATTERNS.search(text):
        return RequestType.BUG
    # Feature requests are checked before generic bug language because
    # a sentence like "could you add ... to fix the broken X" should be a feature request.
    if _FEATURE_PATTERNS.search(text):
        return RequestType.FEATURE_REQUEST
    if _BUG_PATTERNS.search(text):
        return RequestType.BUG
    return RequestType.PRODUCT_ISSUE


def _is_outage(text: str) -> bool:
    return bool(_OUTAGE_PATTERNS.search(text))


_GREETING_THANK_RE = re.compile(r"\b(thank|thanks|appreciate|grateful)\b", re.IGNORECASE)


def _polite_refusal(req_type: RequestType, text: str = "") -> str:
    if req_type == RequestType.INVALID:
        if _GREETING_THANK_RE.search(text):
            return "Happy to help"
        return "I am sorry, this is out of scope from my capabilities."
    return ("I couldn't find a confident answer for your question in our support "
            "documentation. Please reach out to our support team for personalized help.")


def _format_response(top_doc: Doc, score: float, threshold: float) -> str:
    if score < threshold:
        return ""  # caller decides fallback
    para = top_doc.first_paragraph
    if not para:
        return ""
    # Truncate at first paragraph; keep it grounded — no fabrication
    return para


class BaselineAgent:
    def __init__(self, threshold: float = 1.5):
        self.corpus = load_corpus()
        self.index_full = _build_index(self.corpus)
        # Pre-build per-vendor sub-indexes for fast company-routing
        self._vendor_indexes: dict[str, _IndexedCorpus] = {}
        for vendor in {d.vendor for d in self.corpus}:
            sub = vendor_filter(self.corpus, vendor)
            self._vendor_indexes[vendor.lower()] = _build_index(sub)
        self.threshold = threshold

    def _select_index(self, vendor: str) -> _IndexedCorpus:
        v = (vendor or "").strip().lower()
        if v in {"hackerrank", "claude", "visa"}:
            return self._vendor_indexes.get(v, self.index_full)
        return self.index_full

    def predict(self, ticket: TicketInput) -> TicketOutput:
        text = ticket.joined_text
        idx = self._select_index(ticket.normalized_company)
        tokens = _tokenize(text)
        scores = idx.bm25.get_scores(tokens) if tokens else []
        if len(scores) == 0:
            top_doc, top_score = None, 0.0
        else:
            best_i = int(scores.argmax())
            top_doc, top_score = idx.docs[best_i], float(scores[best_i])

        request_type = _classify_request_type(text)

        # Status decision: outage → escalate; everything else → reply (matches gold prior)
        if _is_outage(text) and request_type == RequestType.BUG:
            status = Status.ESCALATED
            response = "Escalate to a human"
            product_area = ""
            justification = "Outage-class report detected ('site down' / 'none of the pages'); routed to human escalation as in gold sample."
        elif request_type == RequestType.INVALID:
            status = Status.REPLIED
            response = _polite_refusal(request_type, text)
            product_area = _normalize_topic(top_doc.topic, top_doc.vendor) if top_doc else ""
            justification = "Off-topic or greeting; replied with polite out-of-scope notice."
        elif top_doc is None or top_score < self.threshold:
            status = Status.REPLIED
            response = _polite_refusal(request_type, text)
            product_area = ""
            justification = f"BM25 top score {top_score:.2f} below threshold {self.threshold}; fell back to polite refusal."
        else:
            response = _format_response(top_doc, top_score, self.threshold)
            if not response:
                response = _polite_refusal(request_type, text)
                product_area = ""
                justification = f"Top doc had empty first paragraph (score {top_score:.2f}); polite refusal."
            else:
                product_area = _normalize_topic(top_doc.topic, top_doc.vendor)
                justification = f"Top BM25 match: {top_doc.title!r} ({top_doc.doc_id}, score={top_score:.2f}). Templated first paragraph as response."
            status = Status.REPLIED

        return TicketOutput(
            issue=ticket.issue,
            subject=ticket.subject,
            company=ticket.company,
            response=response,
            product_area=product_area,
            status=status,
            request_type=request_type,
            justification=justification,
        )

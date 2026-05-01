"""Pydantic schemas + the canonical CSV header contract.

The gold sample CSV uses Title Case headers with spaces:
    Issue, Subject, Company, Response, Product Area, Status, Request Type

We MUST emit `output.csv` with the same header order as the gold so the
evaluator can align rows by index. `INPUT_COLS` are present in the
unlabeled test CSV; `OUTPUT_COLS` are appended.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# ---- Header contract -------------------------------------------------------

INPUT_COLS = ["Issue", "Subject", "Company"]
OUTPUT_COLS = ["Response", "Product Area", "Status", "Request Type"]
GOLD_HEADER = INPUT_COLS + OUTPUT_COLS  # exact column order for output.csv

# Justification is required by the problem statement but the gold sample CSV
# does NOT have a Justification column. We append it as a 5th output column
# at the end so the evaluator can ignore it if it's not graded.
JUSTIFICATION_COL = "Justification"
FULL_OUTPUT_HEADER = GOLD_HEADER + [JUSTIFICATION_COL]


# ---- Enums ----------------------------------------------------------------


class Status(str, Enum):
    REPLIED = "Replied"
    ESCALATED = "Escalated"


class RequestType(str, Enum):
    PRODUCT_ISSUE = "product_issue"
    FEATURE_REQUEST = "feature_request"
    BUG = "bug"
    INVALID = "invalid"


class Vendor(str, Enum):
    HACKERRANK = "HackerRank"
    CLAUDE = "Claude"
    VISA = "Visa"
    NONE = "None"


# ---- Models ---------------------------------------------------------------


class TicketInput(BaseModel):
    """One row from support_tickets.csv (input only)."""

    issue: str = Field(..., description="Ticket body / question")
    subject: str = Field(default="", description="May be blank, partial, or noisy")
    company: str = Field(default="None", description="HackerRank | Claude | Visa | None")

    @property
    def normalized_company(self) -> str:
        c = (self.company or "").strip()
        # Map common typos / casing — the sample CSV has 'None ' with trailing space
        if c.lower() in {"", "none", "n/a", "null"}:
            return "None"
        for v in ("HackerRank", "Claude", "Visa"):
            if c.lower() == v.lower():
                return v
        return c  # unknown — keep as-is, gatekeeper can route

    @property
    def joined_text(self) -> str:
        s = (self.subject or "").strip()
        i = (self.issue or "").strip()
        if s and i:
            return f"{s}\n\n{i}"
        return s or i


class GateVerdict(BaseModel):
    """Output of the 8b gatekeeper."""

    should_escalate: bool
    confidence: float = Field(ge=0.0, le=1.0)
    request_type: RequestType
    product_area: str = Field(..., description="Best-guess area; may be empty string")
    reason: str = Field(..., description="One short sentence")


class TicketOutput(BaseModel):
    """One row of output.csv (5 columns + justification)."""

    issue: str
    subject: str
    company: str
    response: str
    product_area: str
    status: Status
    request_type: RequestType
    justification: str

    def to_csv_row(self) -> dict[str, str]:
        return {
            "Issue": self.issue,
            "Subject": self.subject,
            "Company": self.company,
            "Response": self.response,
            "Product Area": self.product_area,
            "Status": self.status.value,
            "Request Type": self.request_type.value,
            JUSTIFICATION_COL: self.justification,
        }


class TraceLog(BaseModel):
    """JSONL trace record per ticket. One file per run; one line per ticket."""

    ticket_idx: int
    input: TicketInput
    policy_hits: list[str] = Field(default_factory=list)
    retrieved: list[dict] = Field(default_factory=list)  # {doc_id, score, snippet[:120]}
    gate: GateVerdict | None = None
    output: TicketOutput
    latency_ms: dict[str, float] = Field(default_factory=dict)
    cache_hits: dict[str, bool] = Field(default_factory=dict)

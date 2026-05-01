"""YAML policy compiler. Phase 7.

Compiles code/policies/escalation.yaml into a fast matcher. Runs BEFORE
retrieval / gatekeeper / answerer. A hit short-circuits the LLM lane.

Three actions:
  hard_escalate  →  status=Escalated, fixed response, skip LLM
  hard_block     →  status=Replied + invalid + polite refusal, skip LLM
                    (used for prompt-injection / jailbreak attempts)
  force_invalid  →  status=Replied + invalid + given response, skip LLM

The compiler verifies regex compilation at startup; broken rules raise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml

from code.agent.schemas import RequestType, Status, TicketInput, TicketOutput

POLICY_PATH = Path(__file__).resolve().parent.parent / "policies" / "escalation.yaml"


class PolicyAction(str, Enum):
    HARD_ESCALATE = "hard_escalate"
    HARD_BLOCK = "hard_block"
    FORCE_INVALID = "force_invalid"


@dataclass
class CompiledRule:
    id: str
    description: str
    pattern: re.Pattern
    action: PolicyAction
    response: str
    request_type: RequestType
    product_area: str


@dataclass
class PolicyHit:
    rule_id: str
    action: PolicyAction
    matched_text: str  # the substring that triggered the rule


def load_policy(path: Path = POLICY_PATH) -> list[CompiledRule]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    rules: list[CompiledRule] = []
    seen_ids: set[str] = set()
    for r in raw.get("rules", []):
        rid = r["id"]
        if rid in seen_ids:
            raise ValueError(f"duplicate rule id: {rid}")
        seen_ids.add(rid)
        try:
            pattern = re.compile(r["pattern"], re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            raise ValueError(f"rule {rid}: bad regex — {e}") from e
        action = PolicyAction(r["action"])
        rt = RequestType(r["request_type"])
        rules.append(CompiledRule(
            id=rid, description=r.get("description", ""),
            pattern=pattern, action=action,
            response=r["response"].rstrip(),
            request_type=rt, product_area=r.get("product_area", "") or "",
        ))
    return rules


def evaluate(text: str, rules: list[CompiledRule]) -> PolicyHit | None:
    """Return the FIRST matching rule (rules are evaluated in declaration order
    so put hard rules first). None if no rule fires.
    """
    for rule in rules:
        m = rule.pattern.search(text)
        if m:
            return PolicyHit(rule_id=rule.id, action=rule.action, matched_text=m.group(0)[:120])
    return None


def apply_policy(ticket: TicketInput, hit: PolicyHit, rules: list[CompiledRule]) -> TicketOutput:
    """Build the canonical TicketOutput for a policy-driven decision."""
    rule = next(r for r in rules if r.id == hit.rule_id)
    if rule.action == PolicyAction.HARD_ESCALATE:
        status = Status.ESCALATED
    else:
        # hard_block + force_invalid both reply
        status = Status.REPLIED
    just = f"Policy rule '{rule.id}' fired (action={rule.action.value}, matched={hit.matched_text!r})."
    return TicketOutput(
        issue=ticket.issue, subject=ticket.subject, company=ticket.company,
        response=rule.response,
        product_area=rule.product_area,
        status=status,
        request_type=rule.request_type,
        justification=just[:400],
    )


# ---- self-test against the prompt-injection corpus -----------------------


def selftest_injection_corpus() -> tuple[int, int, list[str]]:
    """Verify every line in prompt_injection_corpus.txt hits a hard_block rule.

    Returns (n_total, n_passed, failed_examples).
    """
    rules = load_policy()
    corpus = (Path(__file__).resolve().parent.parent / "policies" / "prompt_injection_corpus.txt").read_text(encoding="utf-8")
    cases = [line.strip() for line in corpus.splitlines()
             if line.strip() and not line.startswith("#")]

    n_pass = 0
    failed: list[str] = []
    for line in cases:
        hit = evaluate(line, rules)
        if hit is not None and hit.action == PolicyAction.HARD_BLOCK:
            n_pass += 1
        else:
            failed.append(f"  MISS: {line!r}  (got: {hit})")
    return len(cases), n_pass, failed


if __name__ == "__main__":
    rules = load_policy()
    print(f"[policy] loaded {len(rules)} rules from {POLICY_PATH.name}")
    for r in rules:
        print(f"  - {r.id:35s}  action={r.action.value:14s}  rt={r.request_type.value}")
    print()
    n, npassed, failed = selftest_injection_corpus()
    print(f"[policy] injection-corpus selftest: {npassed}/{n} hard-blocked")
    if failed:
        print("[policy] failures:")
        for f in failed:
            print(f)

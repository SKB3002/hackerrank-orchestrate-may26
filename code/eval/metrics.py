"""Per-column scorers for the 5 output columns + an aggregate report.

Sample set is small (10 rows). We use simple, explainable metrics that
won't be drowned in noise. Every metric returns a float in [0, 1] and
keeps a short note for the diff report.

Columns scored:
  - Status         (Replied vs Escalated)        precision/recall/F1 + accuracy
  - Request Type   (4-way)                       accuracy + macro-F1
  - Product Area   (free-text)                   fuzzy ratio (rapidfuzz)
  - Response       (free-text)                   ROUGE-L F1 + length-band
  - Justification  (free-text)                   length-band only (no gold col)
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz
from rouge_score import rouge_scorer

from code.agent.schemas import GOLD_HEADER, JUSTIFICATION_COL


@dataclass
class ColumnScore:
    name: str
    score: float
    detail: dict


@dataclass
class RowDiff:
    idx: int
    issue_preview: str
    fields: dict[str, dict]  # field -> {gold, pred, score, note}


@dataclass
class EvalReport:
    n_rows: int
    columns: dict[str, ColumnScore]
    rows: list[RowDiff]
    overall: float

    def summary(self) -> str:
        lines = [f"=== Eval Report ({self.n_rows} rows) ==="]
        for col, cs in self.columns.items():
            lines.append(f"  {col:20s}: {cs.score:.3f}  {cs.detail}")
        lines.append(f"  {'OVERALL':20s}: {self.overall:.3f}")
        return "\n".join(lines)


# ---- helpers --------------------------------------------------------------


def _norm(s: str | None) -> str:
    return (s or "").strip()


def _accuracy(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    return sum(1 for g, p in pairs if g == p) / len(pairs)


def _macro_f1(pairs: list[tuple[str, str]]) -> tuple[float, dict]:
    classes = sorted(set(g for g, _ in pairs) | set(p for _, p in pairs))
    f1s = {}
    for c in classes:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s[c] = round(f1, 3)
    macro = sum(f1s.values()) / len(f1s) if f1s else 0.0
    return macro, f1s


def _binary_pr(pairs: list[tuple[str, str]], positive: str) -> dict:
    tp = sum(1 for g, p in pairs if g == positive and p == positive)
    fp = sum(1 for g, p in pairs if g != positive and p == positive)
    fn = sum(1 for g, p in pairs if g == positive and p != positive)
    tn = sum(1 for g, p in pairs if g != positive and p != positive)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / len(pairs) if pairs else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3), "accuracy": round(acc, 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ---- column scorers -------------------------------------------------------


def score_status(pairs: list[tuple[str, str]]) -> ColumnScore:
    norm = [(_norm(g), _norm(p)) for g, p in pairs]
    detail = _binary_pr(norm, "Escalated")
    detail["accuracy"] = _accuracy(norm)
    return ColumnScore("Status", round(detail["accuracy"], 3), detail)


def score_request_type(pairs: list[tuple[str, str]]) -> ColumnScore:
    norm = [(_norm(g).lower(), _norm(p).lower()) for g, p in pairs]
    acc = _accuracy(norm)
    macro, per_class = _macro_f1(norm)
    return ColumnScore("Request Type", round(acc, 3), {"accuracy": round(acc, 3), "macro_f1": round(macro, 3), "per_class_f1": per_class})


def score_product_area(pairs: list[tuple[str, str]]) -> ColumnScore:
    """Empty-string gold rows are scored 1.0 if pred is empty too, else 0."""
    scores = []
    exact = 0
    for g, p in pairs:
        gn, pn = _norm(g).lower(), _norm(p).lower()
        if gn == pn:
            scores.append(1.0)
            exact += 1
        elif not gn or not pn:
            scores.append(0.0)
        else:
            # token-set ratio is forgiving on word order / extra qualifiers
            scores.append(fuzz.token_set_ratio(gn, pn) / 100.0)
    avg = sum(scores) / len(scores) if scores else 0.0
    return ColumnScore("Product Area", round(avg, 3), {"avg_fuzzy": round(avg, 3), "exact_match": exact, "n": len(scores)})


_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def score_response(pairs: list[tuple[str, str]]) -> ColumnScore:
    """ROUGE-L F1 against gold response. Empty gold → 1.0 if pred also empty."""
    scores = []
    for g, p in pairs:
        gn, pn = _norm(g), _norm(p)
        if not gn and not pn:
            scores.append(1.0)
        elif not gn or not pn:
            scores.append(0.0)
        else:
            scores.append(_ROUGE.score(gn, pn)["rougeL"].fmeasure)
    avg = sum(scores) / len(scores) if scores else 0.0
    return ColumnScore("Response", round(avg, 3), {"avg_rougeL": round(avg, 3), "n": len(scores)})


def score_justification(pairs: list[tuple[str, str]]) -> ColumnScore:
    """No gold for justification — score on length-band quality (40-300 chars)."""
    ok = 0
    for _, p in pairs:
        n = len(_norm(p))
        if 30 <= n <= 400:
            ok += 1
    rate = ok / len(pairs) if pairs else 0.0
    return ColumnScore("Justification", round(rate, 3), {"in_band_rate": round(rate, 3), "n": len(pairs)})


# ---- top-level ------------------------------------------------------------


def evaluate(gold_csv: Path, pred_csv: Path) -> EvalReport:
    gold = list(csv.DictReader(gold_csv.open(encoding="utf-8")))
    pred = list(csv.DictReader(pred_csv.open(encoding="utf-8")))
    if len(gold) != len(pred):
        raise ValueError(f"row count mismatch: gold={len(gold)} pred={len(pred)}")

    cols = {
        "Status": score_status([(g["Status"], p["Status"]) for g, p in zip(gold, pred)]),
        "Request Type": score_request_type([(g["Request Type"], p["Request Type"]) for g, p in zip(gold, pred)]),
        "Product Area": score_product_area([(g["Product Area"], p["Product Area"]) for g, p in zip(gold, pred)]),
        "Response": score_response([(g["Response"], p["Response"]) for g, p in zip(gold, pred)]),
        "Justification": score_justification([(g.get(JUSTIFICATION_COL, ""), p.get(JUSTIFICATION_COL, "")) for g, p in zip(gold, pred)]),
    }

    # Per-row diff for the report
    rows: list[RowDiff] = []
    for i, (g, p) in enumerate(zip(gold, pred)):
        fields = {}
        for col in ["Status", "Request Type", "Product Area"]:
            fields[col] = {"gold": g[col], "pred": p[col], "match": _norm(g[col]).lower() == _norm(p[col]).lower()}
        # For response use a partial-match indicator
        r_gold, r_pred = _norm(g["Response"]), _norm(p["Response"])
        r_score = _ROUGE.score(r_gold, r_pred)["rougeL"].fmeasure if r_gold and r_pred else (1.0 if not r_gold and not r_pred else 0.0)
        fields["Response"] = {"gold": r_gold[:160], "pred": r_pred[:160], "rougeL": round(r_score, 3)}
        rows.append(RowDiff(idx=i, issue_preview=g["Issue"][:140], fields=fields))

    # Overall = weighted average; status & request_type weighted higher because they're discrete + auditable
    weights = {"Status": 0.30, "Request Type": 0.20, "Product Area": 0.15, "Response": 0.30, "Justification": 0.05}
    overall = sum(cols[k].score * w for k, w in weights.items())
    return EvalReport(n_rows=len(gold), columns=cols, rows=rows, overall=round(overall, 3))

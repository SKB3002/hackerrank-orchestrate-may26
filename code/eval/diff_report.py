"""Markdown diff-vs-gold report. Phase 1/9 deliverable."""

from __future__ import annotations

from pathlib import Path

from code.eval.metrics import EvalReport


def render_markdown(report: EvalReport) -> str:
    lines = [
        f"# Eval Report",
        "",
        f"**Rows:** {report.n_rows}  ",
        f"**Overall (weighted):** {report.overall:.3f}",
        "",
        "## Per-column scores",
        "",
        "| Column | Score | Detail |",
        "|---|---|---|",
    ]
    for name, cs in report.columns.items():
        lines.append(f"| {name} | {cs.score:.3f} | `{cs.detail}` |")
    lines += ["", "## Per-row diff", ""]
    for row in report.rows:
        lines.append(f"### Row {row.idx}")
        lines.append(f"> {row.issue_preview}")
        lines.append("")
        lines.append("| Field | Gold | Pred | Match |")
        lines.append("|---|---|---|---|")
        for col in ["Status", "Request Type", "Product Area"]:
            f = row.fields[col]
            mark = "✅" if f["match"] else "❌"
            lines.append(f"| {col} | `{f['gold']}` | `{f['pred']}` | {mark} |")
        rf = row.fields["Response"]
        lines.append(f"| Response (rougeL={rf['rougeL']:.2f}) | {rf['gold']!r} | {rf['pred']!r} | — |")
        lines.append("")
    return "\n".join(lines)


def write(report: EvalReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(report), encoding="utf-8")

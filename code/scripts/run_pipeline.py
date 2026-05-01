"""Run the full pipeline against an input CSV -> output CSV + trace JSONL.

Usage:
    python -m code.scripts.run_pipeline --input <in.csv> --output <out.csv> [--trace <traces.jsonl>] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

from code.agent.pipeline import Pipeline
from code.agent.retrieval import HybridRetriever
from code.agent.schemas import FULL_OUTPUT_HEADER, TicketInput
from code.llm.groq_client import GroqClient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--trace", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N rows (dev/smoke).")
    ap.add_argument("--resume-from", type=Path, default=None,
                    help="Carry over rows from a previous partial CSV. Failed rows (placeholder Escalate with 'Pipeline error' justification) are reprocessed; everything else is kept as-is.")
    args = ap.parse_args()

    print("[pipeline] loading retriever ...")
    retriever = HybridRetriever.load()
    print(f"[pipeline] {len(retriever.chunks)} chunks loaded")

    print("[pipeline] connecting Groq client ...")
    client = GroqClient()

    pipe = Pipeline(retriever=retriever, client=client)

    rows = list(csv.DictReader(args.input.open(encoding="utf-8")))
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"[pipeline] processing {len(rows)} rows ...")

    # Resume support: carry over previously-successful rows from a partial CSV.
    # Only rows whose Justification starts with "Pipeline error" are reprocessed.
    resume: dict[int, dict] = {}
    if args.resume_from is not None and args.resume_from.exists():
        prev = list(csv.DictReader(args.resume_from.open(encoding="utf-8")))
        for i, pr in enumerate(prev[: len(rows)]):
            just = pr.get("Justification", "") or ""
            if not just.startswith("Pipeline error"):
                resume[i] = pr
        print(f"[pipeline] resume: carrying over {len(resume)} rows from {args.resume_from}; reprocessing {len(rows) - len(resume)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)

    # Crash-safe: write each row as it's processed. If we die, we still have
    # the partial CSV + traces and can resume on rerun (the cache makes
    # already-processed rows nearly free).
    out_rows = []
    n_failed = 0
    failed_rows: list[dict] = []
    t_total = time.perf_counter()
    csv_f = args.output.open("w", encoding="utf-8", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=FULL_OUTPUT_HEADER)
    csv_w.writeheader()
    csv_f.flush()
    trace_f = args.trace.open("w", encoding="utf-8") if args.trace is not None else None

    try:
        for i, r in enumerate(rows):
            ticket = TicketInput(
                issue=r.get("Issue", ""),
                subject=r.get("Subject", ""),
                company=r.get("Company", "None") or "None",
            )
            # Resume: keep previously-good rows verbatim, no re-call.
            if i in resume:
                row_out = {k: resume[i].get(k, "") for k in FULL_OUTPUT_HEADER}
                csv_w.writerow(row_out)
                csv_f.flush()
                if trace_f:
                    trace_f.write(json.dumps({"ticket_idx": i, "_resumed": True,
                                               "issue_preview": ticket.issue[:140],
                                               "final": row_out},
                                              ensure_ascii=False, default=str) + "\n")
                    trace_f.flush()
                out_rows.append(row_out)
                print(f"  [{i:02d}] (resumed)  | {row_out['Status']:9s} | {row_out['Request Type']:15s} | PA={row_out['Product Area']!r}")
                continue
            t0 = time.perf_counter()
            try:
                result = pipe.run_sync(ticket, idx=i)
                elapsed = time.perf_counter() - t0
                row_out = result.output.to_csv_row()
                csv_w.writerow(row_out)
                csv_f.flush()
                if trace_f:
                    trace_f.write(json.dumps({**asdict(result.trace), "_total_ms": round(elapsed * 1000, 1)},
                                              ensure_ascii=False, default=str) + "\n")
                    trace_f.flush()
                out_rows.append(row_out)
                gate = result.trace.gate or {}
                print(f"  [{i:02d}] {ticket.normalized_company:10s} | {result.output.status.value:9s} | "
                      f"{result.output.request_type.value:15s} | "
                      f"PA={result.output.product_area!r:25s} | "
                      f"gate_esc={gate.get('should_escalate')} conf={gate.get('confidence', 0):.2f} | "
                      f"{elapsed*1000:.0f}ms")
            except Exception as exc:
                # Don't lose the rest of the run. Log the failure, write a
                # placeholder row, continue. Caller can re-run after fixing
                # the underlying issue (e.g. waiting out a rate-limit).
                n_failed += 1
                err_summary = f"{type(exc).__name__}: {str(exc)[:200]}"
                print(f"  [{i:02d}] !! FAILED: {err_summary}")
                placeholder = {
                    "Issue": ticket.issue, "Subject": ticket.subject, "Company": ticket.company,
                    "Response": "Escalate to a human",
                    "Product Area": "",
                    "Status": "Escalated",
                    "Request Type": "product_issue",
                    "Justification": f"Pipeline error during processing; ticket routed to human escalation. Error: {err_summary[:120]}",
                }
                csv_w.writerow(placeholder)
                csv_f.flush()
                if trace_f:
                    trace_f.write(json.dumps({"ticket_idx": i, "_failed": True, "_error": err_summary,
                                               "issue_preview": ticket.issue[:140]},
                                              ensure_ascii=False, default=str) + "\n")
                    trace_f.flush()
                failed_rows.append({"idx": i, "error": err_summary, "issue": ticket.issue[:80]})
                out_rows.append(placeholder)
    finally:
        csv_f.close()
        if trace_f:
            trace_f.close()

    print(f"\n[pipeline] wrote {len(out_rows)} rows -> {args.output} ({n_failed} failed -> escalated placeholder)")
    print(f"[pipeline] total elapsed: {time.perf_counter() - t_total:.1f}s")
    if failed_rows:
        print("[pipeline] failed rows (re-run with cache to retry):")
        for f in failed_rows:
            print(f"  [{f['idx']:02d}] {f['error'][:100]}  ::  {f['issue']!r}")


if __name__ == "__main__":
    main()

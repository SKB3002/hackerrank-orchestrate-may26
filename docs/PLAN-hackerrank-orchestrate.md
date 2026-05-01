# PLAN — HackerRank Orchestrate (May 2026)

**Project root:** `c:\Suyash_Projects\hackerrank-orchestrate-may26`
**Plan owner:** solo participant
**Time budget:** 24h hard cap → internal target submission at **h23** (1h safety margin)
**Submission deadline:** 2026-05-02T11:00:00+05:30
**Mode:** PLANNING ONLY — no code is written by this document

---

## 1. Executive summary

We are building a deterministic, citation-enforced support-ticket-resolution agent under the **F+C hybrid architecture**: a YAML policy pre-gate that handles the obvious decisions (PII, prompt-injection, low-corpus-overlap, hard escalation triggers) fronts a two-LLM lane — an **8B gatekeeper** (`llama-3.1-8b-instant`) that classifies and decides whether to escalate, and a **70B answerer** (`llama-3.3-70b-versatile`) that produces the final response with mandatory `[doc_id]` citations from a hybrid BM25+BGE+RRF retrieval over the provided corpus. F+C wins over a single-agent design because the policy layer is interview-defensible (deterministic, auditable, fast) and the two-model split keeps the cheap model on the high-volume routing decision while reserving the expensive model for grounded generation. The eval harness is built **first** (h0–h3) so every subsequent change is measurable; an Approach-A BM25+rules baseline is built as a 2-hour side branch to give us a known floor and a fallback if the LLM lane fails. Determinism (temp=0.2, fixed seed, sha256-keyed disk cache, pinned versions, manifest file) is non-negotiable because the AI judge will probe reproducibility.

**What wins:** measurability before model code, policy before LLM, citations or drop, baseline as fallback, ship at h23 not h24.

---

## 2. Repo file layout

Full tree of files we will create. Concrete paths under `code/`.

```
c:\Suyash_Projects\hackerrank-orchestrate-may26\
├── AGENTS.md                                  # already exists, do not modify
├── README.md                                  # already exists, do not modify
├── .env.example                               # already exists; verify GROQ_API_KEY entry
├── .gitignore                                 # already exists; verify excludes index/, .venv, __pycache__, *.pyc, .env
├── docs\
│   └── PLAN-hackerrank-orchestrate.md         # this file
├── support_tickets\
│   ├── sample_support_tickets.csv             # 108 rows, provided
│   ├── support_tickets.csv                    # 56 rows, provided, target of final run
│   └── output.csv                             # GENERATED at h22 — final submission artifact
├── data\                                      # provided corpus
│   ├── visa\
│   ├── hackerrank\
│   └── claude\
└── code\
    ├── README.md                              # design memo + interview cheat-sheet (Phase 10)
    ├── requirements.txt                       # pinned versions
    ├── main.py                                # CLI: python -m code.main --input ... --output ...
    ├── __init__.py
    ├── agent\
    │   ├── __init__.py
    │   ├── pipeline.py                        # orchestrates: policy → retrieve → gate → answer → validate → trace
    │   ├── policy.py                          # YAML policy compiler + matcher (regex + keyword + length + lang)
    │   ├── retrieval.py                       # BM25 + BGE-small + RRF fusion, k=60
    │   ├── gatekeeper.py                      # 8b LLM gate (escalate vs answer)
    │   ├── answerer.py                        # 70b LLM answer w/ enforced citations
    │   ├── validator.py                       # citation enforcer + Pydantic shape validation + cite-or-drop
    │   ├── schemas.py                         # Pydantic: TicketInput, GateVerdict, TicketOutput, TraceLog
    │   ├── taxonomy.py                        # product_area enum derived from data/ subdirs
    │   ├── cache.py                           # diskcache wrapper, sha256(model+prompt+temp+seed)
    │   └── trace.py                           # JSONL trace writer per ticket
    ├── llm\
    │   ├── __init__.py
    │   ├── groq_client.py                     # async client, asyncio.Semaphore, tenacity backoff on 429
    │   └── prompts\
    │       ├── gatekeeper.txt
    │       ├── answerer.txt
    │       └── judge.txt                      # used only by eval harness, never in prod path
    ├── eval\
    │   ├── __init__.py
    │   ├── harness.py                         # main eval entry: scores predictions vs sample
    │   ├── metrics.py                         # per-column scorers (5 columns)
    │   ├── calibrate.py                       # threshold sweep, emits curve data
    │   ├── diff_report.py                     # predicted vs expected → markdown report
    │   └── hand_labels.csv                    # 108 rows, hand-labeled should_escalate (built Phase 1)
    ├── policies\
    │   ├── escalation.yaml                    # ~30 rules — interview centerpiece
    │   └── prompt_injection_corpus.txt        # red-team test cases for the gatekeeper
    ├── index\                                 # build artifacts, gitignored
    │   ├── bm25.pkl
    │   ├── dense.faiss
    │   ├── chunks.jsonl
    │   └── manifest.json                      # determinism manifest: model versions, chunker config, hashes
    └── scripts\
        ├── build_index.py                     # one-time: chunk corpus, build BM25 + FAISS
        └── run.py                             # end-to-end runner: input csv → output csv + traces
```

**Notes on layout:**
- `code/` is the only directory we ship in code.zip.
- `index/` artifacts are rebuilt by the evaluator from `data/` via `scripts/build_index.py`; they are gitignored.
- `policies/escalation.yaml` IS shipped (it's part of the agent, not data).
- `eval/` is shipped because the judge may want to inspect harness/metrics, but `eval/hand_labels.csv` is shipped as evidence of methodology.

---

## 3. Phased task breakdown

Hours are elapsed-from-start. Owner agents listed where the work is naturally specialized.

### Phase 0 — Scaffold + Groq smoke test (h0–h1)

**Owner:** hub:backend-specialist

| Item | Detail |
|---|---|
| Deliverables | `code/requirements.txt` (pinned), `code/__init__.py` + all subpackage `__init__.py`, `code/llm/groq_client.py` (skeleton with one sync call), `.env` populated locally from `.env.example`, corpus discovered (count files per `data/<vendor>/`) |
| Verification | `python -c "from groq import Groq; import os; c=Groq(api_key=os.environ['GROQ_API_KEY']); r=c.chat.completions.create(model='llama-3.1-8b-instant', messages=[{'role':'user','content':'ping'}], temperature=0.2, seed=42); print(r.choices[0].message.content)"` returns a non-empty response. `python -c "import os; print(sum(1 for _,_,fs in os.walk('data') for f in fs))"` prints corpus file count. |
| Depends on | nothing |

### Phase 1 — Eval harness skeleton + hand-labels (h1–h3)

**Owner:** hub:test-engineer (primary), hub:backend-specialist (assist)

| Item | Detail |
|---|---|
| Deliverables | `code/eval/harness.py`, `code/eval/metrics.py`, `code/eval/hand_labels.csv` (108 rows hand-labeled for `should_escalate` boolean and notes), per-column scorers for all 5 output columns (exact-match, F1, citation-coverage, escalation-precision/recall) |
| Verification | `python -m code.eval.harness --predictions <oracle-stub.csv> --gold support_tickets/sample_support_tickets.csv --hand-labels code/eval/hand_labels.csv` prints a metrics dict with non-zero values for every column. Hand-label coverage = 108/108 with two-pass re-read on disagreement-prone rows. |
| Depends on | Phase 0 |

### Phase 2 — Approach A baseline (BM25 + rules) (h3–h5)

**Owner:** hub:backend-specialist

| Item | Detail |
|---|---|
| Deliverables | a 2-hour side-branch baseline: BM25-only retrieval + simple keyword rules → `output.csv`. This is the **floor** that the eval harness scores. If everything else fails by h21, we ship this. |
| Verification | `python -m code.scripts.run --baseline=approach_a --input support_tickets/sample_support_tickets.csv --output /tmp/baseline.csv` produces a complete CSV; `python -m code.eval.harness --predictions /tmp/baseline.csv --gold support_tickets/sample_support_tickets.csv` emits a baseline score we record. |
| Depends on | Phase 1 |

### Phase 3 — Corpus chunking + hybrid retrieval (h5–h7)

**Owner:** hub:backend-specialist

| Item | Detail |
|---|---|
| Deliverables | `code/scripts/build_index.py` (chunk size 600 tokens, overlap 100), `code/agent/retrieval.py` (rank_bm25 + sentence-transformers BGE-small-en-v1.5 + RRF fusion at k=60), `code/agent/taxonomy.py` (product_area enum auto-derived from `data/<vendor>/<topic>/` paths), `code/index/{bm25.pkl,dense.faiss,chunks.jsonl,manifest.json}` |
| Verification | `python -m code.scripts.build_index` completes; `python -c "from code.agent.retrieval import retrieve; print([h.doc_id for h in retrieve('how do I reset my password', k=5)])"` returns 5 plausible doc_ids drawn from at least one vendor. `code/index/manifest.json` records: chunker config, model name+revision, BGE checksum, corpus file count + sha256-of-sha256s. |
| Depends on | Phase 0 (Phase 2 not strictly required but should already be green) |

### Phase 4 — Schemas + structured output + cache + retry (h7–h9)

**Owner:** hub:backend-specialist

| Item | Detail |
|---|---|
| Deliverables | `code/agent/schemas.py` (TicketInput, GateVerdict, TicketOutput, TraceLog Pydantic models), `code/llm/groq_client.py` (async, asyncio.Semaphore: 25 for 8b / 20 for 70b, tenacity exponential backoff on 429/5xx), `code/agent/cache.py` (diskcache keyed on `sha256(model + prompt + temp + seed)`), `instructor` integration sitting on top of Groq's OpenAI-compatible endpoint with retry-on-validation max=2 |
| Verification | a `pytest` (or plain `python -m`) smoke that calls the gated client 30 times concurrently against an instructor-typed dummy schema and observes (a) all 30 succeed, (b) 2nd identical call hits cache (no network), (c) injected 429 triggers backoff. |
| Depends on | Phase 0 |

### Phase 5 — Gatekeeper LLM (8b) (h9–h11)

**Owner:** hub:backend-specialist

| Item | Detail |
|---|---|
| Deliverables | `code/agent/gatekeeper.py`, `code/llm/prompts/gatekeeper.txt` (instruction: classify into product_area enum, decide escalate vs answer, emit confidence ∈ [0,1], one-sentence reason), small unit tests on 10 hand-picked sample tickets covering: clear-answer, clear-escalate, prompt-injection, out-of-corpus, ambiguous |
| Verification | on the 10 unit cases, gatekeeper emits a valid `GateVerdict` for every input (Pydantic parses 10/10) and the should_escalate decision matches hand-label on ≥8/10. |
| Depends on | Phase 3, Phase 4 |

### Phase 6 — Answerer LLM (70b) + citation enforcement (h11–h13)

**Owner:** hub:backend-specialist

| Item | Detail |
|---|---|
| Deliverables | `code/agent/answerer.py`, `code/llm/prompts/answerer.txt` (every sentence must end with one or more `[doc_id]` markers, refuse-with-escalation if context insufficient), `code/agent/validator.py` (post-hoc citation enforcer: drops uncited sentences, computes citation-coverage; rejects answers that drop >50% of sentences and forces escalation) |
| Verification | on 5 sample tickets with retrieved context, every emitted answer contains ≥1 `[doc_id]` per sentence after validator; on 1 deliberately-broken-context test, validator forces an escalation rather than emit hallucination. |
| Depends on | Phase 3, Phase 4 |

### Phase 7 — YAML policy compiler + escalation.yaml + injection corpus (h13–h15)

**Owner:** hub:backend-specialist

| Item | Detail |
|---|---|
| Deliverables | `code/agent/policy.py` (compile YAML → matcher: regex, keyword, length, lang-detect, PII, prompt-injection), `code/policies/escalation.yaml` (~30 rules grouped: hard-escalate, soft-signal, hard-block, route-hint), `code/policies/prompt_injection_corpus.txt` (≥30 known jailbreak / role-override / data-exfil prompts) |
| Verification | `python -m code.agent.policy --test code/policies/prompt_injection_corpus.txt` reports 100% of injection cases hit a hard-block rule; spot-check 5 sample tickets and confirm hard-escalate rules don't false-positive on benign cases. |
| Depends on | Phase 1 (we need the hand-labels to know what hard-escalate looks like) |

### Phase 8 — Pipeline integration + first end-to-end run (h15–h17)

**Owner:** hub:backend-specialist

| Item | Detail |
|---|---|
| Deliverables | `code/agent/pipeline.py` (policy → retrieve → gate → answer → validate → trace), `code/agent/trace.py` (JSONL writer, one record per ticket with hashable inputs/outputs), `code/main.py` and `code/scripts/run.py` (CLI orchestration), first end-to-end run on a 5-ticket dev subset |
| Verification | `python -m code.main --input <5-row-subset.csv> --output /tmp/e2e.csv --trace-dir /tmp/traces` emits an `output.csv` with all 5 columns populated for all 5 rows AND a `traces/` directory with 5 JSONL files, each containing every stage's input/output. Re-running with cache hot finishes in <30s. |
| Depends on | Phase 5, Phase 6, Phase 7 |

### Phase 9 — Threshold calibration + diff report + iterate (h17–h19)

**Owner:** hub:test-engineer (primary), hub:backend-specialist (assist)

| Item | Detail |
|---|---|
| Deliverables | `code/eval/calibrate.py` (sweep escalation-confidence threshold over [0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9] — 8 points; cut to 3 if behind schedule), `code/eval/diff_report.py` (markdown report: predicted vs expected with per-row diff and worst-N failures), iterate on prompts / YAML rules to fix worst 10 failures |
| Verification | calibration emits a curve (precision/recall vs threshold) printable in the README; diff_report shows ≥X% improvement over Approach A baseline (X is whatever we achieve, but it must be > baseline or we ship Approach A). |
| Depends on | Phase 8 |

### Phase 10 — Determinism manifest + trace polish + README (h19–h21)

**Owner:** hub:documentation-writer (primary), hub:backend-specialist (assist)

| Item | Detail |
|---|---|
| Deliverables | `code/index/manifest.json` finalized (model IDs, model SHAs where available, chunker config, BGE revision, rank_bm25 version, prompt hashes, calibrated threshold, eval score, timestamp); `code/README.md` design memo (why F+C, why two models, calibration curve embedded, citation policy, how to reproduce, env vars, run command, known limitations) |
| Verification | `python -c "import json; m=json.load(open('code/index/manifest.json')); assert set(m.keys()) >= {'models','chunker','bge_revision','threshold','eval_score','timestamp','prompt_hashes'}"` passes. README opens cleanly in a markdown viewer and the calibration curve renders. |
| Depends on | Phase 9 |

### Phase 11 — Final run + verification + zip (h21–h22.5)

**Owner:** hub:backend-specialist

| Item | Detail |
|---|---|
| Deliverables | `python -m code.main --input support_tickets/support_tickets.csv --output support_tickets/output.csv` → final 56-row CSV; submission `code.zip` excluding `data/`, `support_tickets/*.csv`, `.venv`, `__pycache__`, `code/index/{*.pkl,*.faiss,chunks.jsonl}` (but INCLUDES `code/index/manifest.json` and `code/policies/`) |
| Verification | `python -c "import csv; r=list(csv.DictReader(open('support_tickets/output.csv',encoding='utf-8'))); assert len(r)==56; assert all(all(row[c] for c in ['ticket_id','product_area','should_escalate','response','citations']) for row in r)"` passes. `unzip -l code.zip` shows no excluded artifacts and includes `policies/escalation.yaml` + `index/manifest.json`. |
| Depends on | Phase 10 |

### Phase 12 — Buffer / fix-it (h22.5–h23)

**Owner:** whoever has cycles

| Item | Detail |
|---|---|
| Deliverables | reserved time for whatever broke in Phase 11 verification; if nothing broke, run a final eval on `sample_support_tickets.csv` with the production pipeline and paste numbers into README. |
| Verification | submission uploaded to HackerRank Community Platform by **h23**, log.txt confirmed at `%USERPROFILE%\hackerrank_orchestrate\log.txt`. |
| Depends on | Phase 11 |

---

## 4. Dependency graph

```
Phase 0 ── Phase 1 ── Phase 2 (Approach A baseline, side-branch fallback)
   │          │
   │          └─── Phase 7 (policy needs hand-labels)
   │
   ├── Phase 3 (retrieval) ──┐
   │                          ├── Phase 5 (gatekeeper) ──┐
   ├── Phase 4 (client/cache)─┤                          │
   │                          └── Phase 6 (answerer) ────┤
   │                                                     │
   │                          Phase 7 (policy) ──────────┤
   │                                                     │
   │                                  Phase 8 (pipeline integration)
   │                                                     │
   │                                  Phase 9 (calibrate + iterate)
   │                                                     │
   │                                  Phase 10 (manifest + README)
   │                                                     │
   │                                  Phase 11 (final run + zip)
   │                                                     │
   │                                  Phase 12 (buffer)
```

**Critical path:** 0 → 1 → 3 → {4, 5, 6, 7} → 8 → 9 → 10 → 11.
**Side branch:** 2 (Approach A) is parallelizable after Phase 1; finished by h5 and never touched again unless we need the fallback.
**Phase 4 can run in parallel** with Phase 3 if we have two attention-windows; otherwise serial is fine, the budget tolerates it.

---

## 5. Verification checklist per phase (consolidated)

| Phase | DONE when |
|---|---|
| 0 | Groq smoke call returns text; corpus file count printed |
| 1 | `harness.py` emits non-zero metrics dict for all 5 columns; `hand_labels.csv` has 108/108 rows |
| 2 | Approach A produces a complete CSV and a recorded baseline score |
| 3 | `build_index.py` completes; `retrieve()` returns 5 plausible doc_ids; `manifest.json` populated |
| 4 | 30 concurrent calls succeed; cache hit on repeat; 429 backoff observed |
| 5 | 10/10 unit cases parse to `GateVerdict`; ≥8/10 match hand-label |
| 6 | every emitted answer has ≥1 `[doc_id]` per sentence after validator; broken-context test forces escalation |
| 7 | 100% prompt-injection corpus hits hard-block; benign sample tickets do not false-positive on hard-escalate |
| 8 | end-to-end on 5-row subset emits valid `output.csv` + 5 trace files |
| 9 | calibration curve printed; diff report shows improvement over Approach A |
| 10 | `manifest.json` has all required keys; README renders with curve |
| 11 | final 56-row `output.csv` validated; `code.zip` excludes data/CSVs but includes policies/manifest |
| 12 | submission uploaded to HackerRank by h23 |

---

## 6. Critical-path risks + mitigations

| # | Risk | Probability | Trigger / detection | Mitigation / fallback |
|---|---|---|---|---|
| R1 | Groq rate limit hits during dev iteration | High | 429s in tenacity logs | diskcache keyed on prompt hash → repeat calls free; narrow dev to 20-row subset; add per-model semaphore limits (8b=25, 70b=20) |
| R2 | BGE-small download fails / very slow on hackathon network | Medium | timeout in `build_index.py` | fallback to `all-MiniLM-L6-v2` (smaller, more cached); ultimate fallback: BM25-only retrieval with k=20 |
| R3 | `instructor` library is incompatible with Groq's OpenAI-compatible endpoint | Medium | TypeError on first call | drop instructor; manual JSON output with `response_format={'type':'json_object'}` + `pydantic.TypeAdapter.validate_json`; retry-on-validate loop max=2 |
| R4 | YAML policy false-positives over-escalate (hurts recall on answerable tickets) | Medium | calibration curve shows precision-on-answer drop | downgrade hard-escalate rules to soft-signal mode (only adjust threshold by +/- delta, never force-escalate from policy); keep hard-block only for prompt-injection + PII |
| R5 | Citation enforcer drops too many sentences (answers become empty) | Medium | validator drop-rate > 50% on >20% of tickets | tune retriever k upward; relax citation rule from "every sentence" to "every claim sentence" (skip greetings/closings); ultimate fallback: cite-at-paragraph granularity |
| R6 | Hour 18 and we're behind | Medium | Phase 9 not started by h18 | cut calibration sweep from 8 points to 3 ({0.5, 0.65, 0.8}); skip Phase 10 polish; ship with default threshold 0.65 |
| R7 | Hour 21 and the LLM lane is still failing eval | Low-Medium | diff report worse than Approach A baseline | ship Approach A (Phase 2 baseline). It runs, it produces a complete CSV, it has a known floor score. Better than a broken submission. |
| R8 | output.csv schema mismatch with evaluator expectations | Medium | Phase 11 verification fails | keep `support_tickets/sample_support_tickets.csv` schema as the contract; `pipeline.py` must emit exactly that header order |
| R9 | Determinism breaks (different runs → different outputs) | Low | re-run produces non-identical hash | enforce temp=0.2, fixed seed=42, cache forced-hit, log nondeterministic call sites; note in manifest if any model call is observed nondeterministic |
| R10 | log.txt missing or polluted with secrets | Low | grep for "GROQ_API_KEY" in log | every prompt logged via `[REDACTED]` substitution; `.env` never read into log path |

---

## 7. Submission artifacts checklist

At h23 we hand in:

- [ ] **`code.zip`** — built from `code/` directory.
  - **Includes:** all `*.py`, `code/policies/*.yaml`, `code/policies/*.txt`, `code/llm/prompts/*.txt`, `code/eval/hand_labels.csv`, `code/index/manifest.json`, `code/README.md`, `code/requirements.txt`, `code/.env.example`
  - **Excludes:** `data/`, `support_tickets/*.csv`, `.venv/`, `__pycache__/`, `*.pyc`, `.env`, `code/index/*.pkl`, `code/index/*.faiss`, `code/index/chunks.jsonl`, `.git/`
- [ ] **`support_tickets/output.csv`** — exactly 56 rows, all 5 columns populated, header matches sample
- [ ] **`%USERPROFILE%\hackerrank_orchestrate\log.txt`** — full chat transcript, no secrets, AGREEMENT RECORDED present
- [ ] **HackerRank Community Platform** — submission link from the hackathon email, files uploaded, confirmation screenshot

---

## 8. AI Judge interview prep

Likely question → artifact / number we point to.

| Question | Pointer |
|---|---|
| "Why two agents instead of one?" | `policies/escalation.yaml` shows the deterministic decisions a single LLM would have to re-derive every call. Gatekeeper precision/recall on `eval/hand_labels.csv` shows the 8b is good enough for routing — wasting 70B on routing is 5–10× cost for no quality gain. |
| "Why temp=0.2 and not 0.0?" | Groq's seed support is best-effort; pure temp=0 still drifts on long generations. 0.2 with fixed seed gave us best reproducibility in `eval/calibrate.py` runs (show the variance numbers). |
| "How do you prevent hallucination?" | `agent/validator.py` enforces `[doc_id]` per sentence; drop-then-escalate when coverage < 50%. Show one trace where validator forced an escalation. |
| "What if the corpus doesn't have the answer?" | Two layers: (1) policy YAML's low-confidence rules; (2) gatekeeper's confidence threshold (calibrated, see curve in README). Both routes lead to escalation, not hallucination. |
| "Why F+C and not pure agentic / DSPy?" | Time budget. Pure-agent (Approach D) has unbounded loop risk; DSPy compile (Approach E) needs labeled training data we don't have. F+C is interview-defensible because every decision is auditable. |
| "How do you know it's deterministic?" | `code/index/manifest.json` records model IDs, prompt hashes, seed, threshold, eval score. Re-run produces identical hashes (modulo the noted Groq-side caveat). |
| "How would you test it in CI?" | `eval/harness.py` against `eval/hand_labels.csv` is exactly that test; run it on every PR with a pinned threshold. |
| "What's your weakest link?" | The 70B answerer's cost and latency; on a higher budget we'd add a cross-encoder reranker (BGE-reranker) before the 70B call to drop k from 60 → 8. |
| "What would you do with another 24 hours?" | (1) BGE-reranker for retrieval precision; (2) DSPy compile-and-optimize on the prompts; (3) multilingual support (currently English-only); (4) online learning from explicit user feedback signal; (5) adversarial test corpus expansion. |
| "Why not fine-tune?" | No labeled training set, no time, and the corpus is small enough that retrieval+citation beats fine-tuning for grounding faithfulness. |

---

## 9. What we are explicitly NOT building (scope defense)

| Excluded | Why |
|---|---|
| Web UI / FastAPI server | Spec is terminal-only, batch CSV |
| Fine-tuning | No labels, no time, no need |
| External knowledge sources (web search, etc.) | Spec says corpus-only |
| Multilingual support | Corpus is English; lang-detect is used only as an escalation signal |
| Real-time streaming | Spec is batch CSV |
| Approach D (three-agent + critic loop) | Unbounded loop risk; latency cost; hard to prove termination in interview |
| Approach E (DSPy compile-and-optimize) | Needs a labeled train/dev split we don't have; tooling risk in 24h |
| Cross-encoder reranker (BGE-reranker) | Stretch goal only — ship if h18 ahead of schedule, else cut |
| Multi-tenant / multi-user features | Single-shot CSV processing, irrelevant |
| Observability stack (OpenTelemetry, Langfuse, etc.) | JSONL traces are sufficient for interview defense |

---

## 10. Open decisions (deferred)

| Decision | Resolves at | Currently |
|---|---|---|
| Exact escalation-confidence threshold | Phase 9 calibration sweep | placeholder 0.65 |
| Final list of escalation YAML rules | grows during Phase 7 + iterates in Phase 9 | starts ~30, may end 25–40 |
| Whether to ship the cross-encoder reranker | only if h18 is ahead of schedule | default: NO |
| Whether to ship Approach A or the LLM lane | Phase 11 verification | default: LLM lane; fallback to A if eval shows it's worse |
| Final answerer prompt wording | last touched in Phase 9 iteration | first-cut by h13 |
| Whether to use FAISS or numpy-cosine for dense | Phase 3 implementation | default FAISS; numpy if FAISS install flakes on Windows |

---

**File created:** `c:\Suyash_Projects\hackerrank-orchestrate-may26\docs\PLAN-hackerrank-orchestrate.md`

# HackerRank Orchestrate — Multi-vendor Support Triage Agent

> Terminal-based agent that resolves support tickets across HackerRank, Claude,
> and Visa using only the provided local corpus. Built for the May 2026
> HackerRank Orchestrate hackathon.

## TL;DR

Architecture is **F+C hybrid**: a deterministic YAML policy pre-gate fronts a
two-LLM lane (8B gatekeeper + 70B answerer) with mandatory retrieval-grounded
citations. Eval-driven build — every change scored against the 10-row sample
set first.

| Stage | Score |
|---|---|
| Dumb baseline (always-Replied + always-invalid) | 0.340 |
| Approach A (BM25 + rules, no LLM) | 0.699 |
| LLM pipeline v1 | 0.780 |
| LLM pipeline + self-service post-fix | 0.821 |
| **+ YAML policy pre-gate (final)** | **0.836** |

## Run it

```
python -m venv .venv
. .venv/Scripts/Activate.ps1            # Windows; use `bin/activate` on Linux/macOS
pip install -r code/requirements.txt
cp .env.example .env                     # then put your GROQ_API_KEY in .env
python -m code.scripts.build_index       # one-time: chunks corpus, builds BM25 + BGE index (~7 min)
python -m code.main \
    --input  support_tickets/support_tickets.csv \
    --output support_tickets/output.csv \
    --trace  runs/traces.jsonl
```

The runner is crash-safe — every row is written to the CSV as it's processed.
A repeat run uses the LLM cache (`code/index/.llm_cache/`) and is essentially
free for tickets we've already answered.

## Architecture

```
ticket
  |
  v
[ POLICY YAML ]  ─────────────► hard_escalate / hard_block / force_invalid
  |                              (no LLM call — deterministic floor)
  v
[ HYBRID RETRIEVAL ]            BM25(rank_bm25) + BGE-small dense + RRF k=60
  |                              chunks=1336, chunk_tokens=600, overlap=100
  v
[ 8B GATEKEEPER ]               sees ticket + top-5 titles only (NOT bodies)
  |                              emits {escalate, confidence, request_type,
  |                                     product_area, reason}
  |  ─── escalate? + conf >= 0.5 ──► Escalate
  v
[ 70B ANSWERER ]                sees ticket + gate verdict + top-5 chunk bodies
  |                              emits {response with [doc_id] citations,
  |                                     product_area, request_type, justification}
  v
[ VALIDATOR ]                   sentence-splits, drops uncited factual claims;
  |                              if drop_rate > 0.6 → polite refusal fallback
  v
output row + JSONL trace
```

## Why F+C (not pure agentic, not pure deterministic)

| Approach | Why we rejected it |
|---|---|
| Pure deterministic (BM25 + rules only) | Ceiling is ~0.70 on the dev set; weak on free-text Response |
| Single-LLM RAG (one prompt does it all) | No safety net for prompt-injection / coercive demands; one bad prompt poisons output |
| Three-agent + self-critique | 4× the LLM calls; Groq free-tier daily TPD limit makes it infeasible |
| DSPy compile-and-optimize | Need labeled train/dev split — only 10 sample rows |
| **F+C (chosen)** | Deterministic pre-gate is interview-defensible; two-model split keeps the cheap model on routing and the expensive model on grounded generation |

The YAML policy is the centerpiece artifact: 9 rules, 14/14 prompt-injection
test corpus caught, 0 false-positives on the 10 dev rows. **20% of unseen test
rows were resolved by the policy with zero LLM tokens** — this saved us when
we hit the Groq TPD ceiling near the end of the run.

## Models (Groq free tier)

| Role | Model | Why |
|---|---|---|
| Gatekeeper (routing) | `llama-3.1-8b-instant` | ~150ms/call, fine for binary verdict + classification |
| Answerer (generation) | `llama-3.3-70b-versatile` | Quality on grounded responses with citations |
| Eval judge (offline only) | `qwen-2.5-32b` | Different family → no same-model bias on eval |

Per-model `asyncio.Semaphore` limits (8b=10, 70b=6) and `tenacity` exponential
backoff on 429s. All calls cached on `sha256(model + messages + temp + seed)`.

## Determinism stance

- `temperature = 0.2`, `seed = 42` everywhere.
- Disk-cached responses → identical reruns return identical CSVs (modulo the
  rare case where Groq's seed support drifts, which we've not observed).
- `code/index/manifest.json` records every load-bearing knob: model versions,
  prompt hashes, policy hashes, requirements hash, threshold values.

## Citation enforcement

Every sentence in the answerer's response that states a corpus-derived fact
must end with `[doc_id]`. The validator post-processes:

1. Split response into sentences.
2. For each sentence, find `[doc_id]` markers.
3. If a sentence is "factual-looking" (long, no greeting language) AND has no
   citation OR cites a doc_id that's not in the retrieved set → drop it.
4. If `drop_rate > 0.60`, force a polite refusal fallback (the answer was too
   ungrounded to ship).

This is the structural defense against hallucination.

## Escalation policy

Gold escalation rate on the 10 sample rows is 1/10 = 10%. We treat escalation
as **rare** — the default is to reply (with a polite refusal if the corpus
can't help). Escalations come from three sources, in order:

1. **YAML policy hard_escalate rules** — outage, account-takeover-by-third-party,
   refund/agent-action demands, score manipulation, payment IDs, infosec forms.
2. **Gatekeeper escalates with confidence ≥ 0.5** AND no retrieval hits would help.
3. **Validator forces escalation** if more than 60% of answer sentences had to
   be dropped (signals the answerer was hallucinating).

Low-confidence gate escalations with hits are **demoted** — the gate's hint
is treated as low-signal and the answerer is allowed to try.

## What's where

```
code/
  README.md                this file
  requirements.txt         pinned versions, all deterministic
  main.py                  CLI: input CSV → output CSV
  agent/
    pipeline.py            orchestrates: policy → retrieve → gate → answer → validate
    policy.py              YAML compiler + matcher (interview centerpiece)
    retrieval.py           BM25 + BGE-small + RRF fusion
    gatekeeper.py          8B routing decision + self-service-override post-process
    answerer.py            70B grounded response with mandatory citations
    validator.py           sentence-level citation enforcer (cite-or-drop)
    schemas.py             Pydantic + canonical CSV header
    corpus.py              frontmatter-aware loader for data/<vendor>/<topic>/*.md
    cache.py               diskcache wrapper, sha256 keys
    baseline.py            Approach A — deterministic floor / fallback path
  llm/
    groq_client.py         async + sync client with semaphore + tenacity
    structured.py          JSON+Pydantic structured output, retry on validation fail
    prompts/
      gatekeeper.txt       routing system prompt
      answerer.txt         grounded-response system prompt
  policies/
    escalation.yaml        9 hand-authored rules
    prompt_injection_corpus.txt   14 jailbreak attempts (selftest 14/14)
  eval/
    harness.py             CLI scorer
    metrics.py             per-column scorers (status / req-type / product-area / response / justification)
    diff_report.py         markdown predicted-vs-gold report
    smoke.py               gold-vs-gold sanity check
  index/                   built artifacts (gitignored except manifest.json)
    chunks.jsonl
    bm25.pkl
    dense.npy
    manifest.json          determinism receipt
  scripts/
    build_index.py         one-time corpus chunking + indexing
    run_baseline.py        CLI for Approach A
    run_pipeline.py        CLI for the full F+C pipeline
    test_retrieval.py      retrieval smoke
    test_groq_client.py    Phase 4 client verification
    test_policy.py         policy sample-set verification
    write_manifest.py      finalize the determinism manifest
```

## Limitations / known gaps

- **Visa corpus is small** (14 docs vs HackerRank's 438). Visa-tagged tickets
  have lower retrieval recall and rely more on the gatekeeper's commonsense
  routing.
- **Greeting-in-middle false-positive risk** is mitigated with `\A...\Z`
  anchors on the `bare_greeting` rule, but a long ticket that ends with a
  pure thanks could in theory still be misclassified (none observed).
- **`product_area` is free-text in the gold** (mix of directory-style names
  like `screen` and descriptive names like `travel_support` and even empty
  strings). We let the LLM emit free-text but bias it toward retrieved-doc
  topics.
- **Groq daily TPD ceiling** (100k tokens for 70B on free tier) is the real
  bottleneck. The YAML policy mitigates this by handling 20% of tickets
  deterministically.

## Reproducibility

```
python -m code.scripts.write_manifest   # build the determinism manifest
python -m code.eval.harness \
    --predictions support_tickets/output.csv \
    --gold        support_tickets/sample_support_tickets.csv \
    --report      runs/diff.md
```

## Scoring methodology (eval harness)

| Column | Metric | Range |
|---|---|---|
| `Status` | Binary precision / recall / F1 + accuracy | [0, 1] |
| `Request Type` | Accuracy + macro-F1 over {product_issue, feature_request, bug, invalid} | [0, 1] |
| `Product Area` | Exact match → 1.0; else `rapidfuzz.token_set_ratio / 100`; empty-vs-empty → 1.0 | [0, 1] |
| `Response` | ROUGE-L F1 vs gold; both-empty → 1.0; one-empty → 0.0 | [0, 1] |
| `Justification` | Length-band (30–400 chars). No gold col available | [0, 1] |
| **Overall** | Weighted average: Status 0.30, Response 0.30, Request Type 0.20, Product Area 0.15, Justification 0.05 | [0, 1] |

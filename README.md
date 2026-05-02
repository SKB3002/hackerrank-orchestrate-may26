# HackerRank Orchestrate

Starter repository for the **HackerRank Orchestrate** 24-hour hackathon (May 1–2, 2026).

Build a terminal-based AI agent that triages real support tickets across three product ecosystems; **HackerRank**, **Claude**, and **Visa** — using only the support corpus shipped in this repo.

Read [`problem_statement.md`](./problem_statement.md) for the full task spec, input/output schema, and allowed values, and [`evalutation_criteria.md`](./evalutation_criteria.md) for how submissions are scored.

---

## Contents

1. [Repository layout](#repository-layout)
2. [What you need to build](#what-you-need-to-build)
3. [Where your code goes](#where-your-code-goes)
4. [Quickstart](#quickstart)
5. [Chat transcript logging](#chat-transcript-logging)
6. [Submission](#submission)
7. [Judge interview](#judge-interview)
8. [Evaluation criteria](#evaluation-criteria)

---

## Repository layout

```
.
├── AGENTS.md                       # Rules for AI coding tools + transcript logging
├── problem_statement.md            # Full task description and I/O schema
├── README.md                       # You are here
├── code/                           # ← Build your agent here
│   └── main.py                     #   Entry point (rename/extend as you like)
├── data/                           # Local-only support corpus (no network needed)
│   ├── hackerrank/                 #   HackerRank help center
│   ├── claude/                     #   Claude Help Center export
│   └── visa/                       #   Visa consumer + small-business support
└── support_tickets/
    ├── sample_support_tickets.csv  # Inputs + expected outputs (for development)
    ├── support_tickets.csv         # Inputs only (run your agent on these)
    └── output.csv                  # Write your agent's predictions here
```

---

## Mental Model 

        ┌──────────────────────┐
        │   USER TICKET        │
        └──────────┬───────────┘
                   │
    ┌──────────────▼──────────────┐
    │  YAML POLICY (10 rules)     │   no LLM
    │  refunds, score-fix, PII…   │   ◄── 20% of tickets handled here
    └──────────────┬──────────────┘
                   │ no hit
    ┌──────────────▼──────────────┐
    │  RETRIEVAL                  │
    │  BM25 + BGE → RRF top-5     │   pulls 5 chunks from
    └──────────────┬──────────────┘   the 774-doc corpus
                   │
    ┌──────────────▼──────────────┐
    │  8B GATEKEEPER              │   ROUTES only
    │  emits: escalate? +         │   does NOT write text
    │  request_type + product_area│
    └──────┬───────────────┬──────┘
           │ escalate      │ reply
           ▼               ▼
    "Escalate to   ┌─────────────────────┐
     a human"      │ 70B ANSWERER        │   WRITES the response
                   │  reads chunks +     │   with [doc_id] citations
                   │  emits cited prose  │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │  VALIDATOR          │
                   │  drop uncited       │
                   │  sentences          │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │  output.csv row     │
                   └─────────────────────┘
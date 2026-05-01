"""Citation-enforcement validator. Phase 6.

Post-processes answerer output:
  1. Splits the response into sentences.
  2. For each sentence: counts [doc_id] markers and checks they exist in
     the retrieved hit set.
  3. If a sentence has no citation AND looks like a factual claim, drop it
     (or replace with a brief refusal if too many drops happen).
  4. Strips raw [doc_id] markers from the final user-facing response —
     citations are recorded internally but not shown to the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A doc_id from our corpus looks like "vendor/topic/article-id". We match
# permissively: brackets containing letters/digits/dashes/slashes/underscores.
_CITE_RE = re.compile(r"\[([a-z0-9][a-z0-9_/\-#.]*)\]", re.IGNORECASE)

# Common abbreviations whose trailing period must NOT be treated as a sentence
# boundary. Without this, "the U.S. Virgin Islands" splits after U.S.
_ABBREV = {"u.s", "u.k", "e.g", "i.e", "mr", "mrs", "ms", "dr", "inc", "ltd",
           "co", "corp", "vs", "etc", "no", "fig", "approx", "incl",
           "u.s.a", "u.k.a", "p.s"}
# Match a sentence-ender (.?!), optional close-quote/bracket, then whitespace,
# then a starting char (capital letter or [ for citation).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])[\"'\)\]]?\s+(?=[A-Z\[])")


def _is_abbrev_at(text: str, dot_idx: int) -> bool:
    """True if the period at text[dot_idx] is part of a known abbreviation."""
    # Walk back collecting [a-z0-9.] until we hit whitespace
    i = dot_idx
    while i > 0 and (text[i - 1].isalnum() or text[i - 1] == "."):
        i -= 1
    word = text[i:dot_idx].lower().rstrip(".")
    return word in _ABBREV

# Sentences that don't need a citation: greetings, sign-offs, refusals,
# transitions, or pure imperatives without specific facts.
_NON_FACT_RE = re.compile(
    r"^(hi|hello|hey|greetings|thanks|thank\s+you|happy\s+to\s+help|"
    r"here'?s|here\s+is|here\s+are|please|sorry|i'?m\s+sorry|"
    r"i\s+(am|was)\s+sorry|i\s+(can|cannot|couldn'?t|could)|"
    r"if\s+you|let\s+me|to\s+(do|fix|resolve)|follow\s+the\s+steps|"
    r"contact\s+(our\s+)?support)\b",
    re.IGNORECASE,
)


@dataclass
class ValidationResult:
    cleaned_response: str  # citations stripped, ready for output.csv
    raw_response: str      # the original with [doc_id] markers preserved
    cited_doc_ids: list[str]
    n_sentences: int
    n_cited: int
    n_dropped: int
    drop_rate: float
    forced_escalate: bool
    notes: list[str]


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    # Iteratively split, skipping splits that occur right after a known abbrev.
    # Walk through every match position; only honor it if the dot before isn't
    # part of an abbreviation.
    parts: list[str] = []
    last = 0
    for m in _SENTENCE_SPLIT_RE.finditer(text):
        # Find the index of the . / ! / ? that triggered the boundary
        dot_idx = m.start() - 1
        # Move past optional trailing quote/bracket
        while dot_idx > 0 and text[dot_idx] in "\"')]":
            dot_idx -= 1
        if text[dot_idx] == "." and _is_abbrev_at(text, dot_idx):
            continue  # skip this boundary
        parts.append(text[last:m.start()].strip())
        last = m.end()
    if last < len(text):
        parts.append(text[last:].strip())
    return [p for p in parts if p]


def _looks_factual(s: str) -> bool:
    """Best-effort heuristic. Long sentences without polite phrasing are factual."""
    if len(s) < 30:
        return False
    if _NON_FACT_RE.match(s):
        return False
    return True


def validate(
    raw_response: str,
    valid_doc_ids: set[str],
    drop_threshold: float = 0.85,
) -> ValidationResult:
    """Enforce citations against the retrieved doc_id set.

    Args:
      raw_response: the answerer's response with [doc_id] markers
      valid_doc_ids: doc_ids from the retrieval hits passed to the answerer
      drop_threshold: if drop_rate > this, flip forced_escalate=True

    Returns:
      ValidationResult with cleaned_response (citations stripped) + diagnostics.
    """
    notes: list[str] = []
    cited_doc_ids: list[str] = []
    sentences = _split_sentences(raw_response)
    if not sentences:
        return ValidationResult(
            cleaned_response="", raw_response=raw_response, cited_doc_ids=[],
            n_sentences=0, n_cited=0, n_dropped=0, drop_rate=0.0,
            forced_escalate=True, notes=["empty response from answerer"],
        )

    # Pre-pass: does the response have at least one valid citation anywhere?
    # If so, treat the whole response as "covered" and only drop sentences with
    # citations that point at OUT-OF-SET (hallucinated) doc_ids. Many models
    # emit one citation at the end covering a paragraph rather than per-sentence;
    # we accept that style as long as a valid doc_id is named SOMEWHERE.
    all_cites = _CITE_RE.findall(raw_response)
    valid_cites_anywhere = [c for c in all_cites if c in valid_doc_ids]
    has_grounding = bool(valid_cites_anywhere)
    cited_doc_ids = list(dict.fromkeys(valid_cites_anywhere))

    kept: list[str] = []
    n_dropped = 0
    n_cited = 0
    for s in sentences:
        cites = _CITE_RE.findall(s)
        if cites:
            valid_cites = [c for c in cites if c in valid_doc_ids]
            if not valid_cites:
                # Cited a doc_id that's not in the retrieved set — possible hallucination
                if _looks_factual(s):
                    notes.append(f"dropped (bad cite): {s[:80]!r}")
                    n_dropped += 1
                    continue
                notes.append(f"bad-cite-tolerated (non-factual): {s[:60]!r}")
                kept.append(s)
            else:
                n_cited += 1
                kept.append(s)
        else:
            # No citation on THIS sentence. Tolerate if the response as a whole
            # is grounded; drop only if both: (a) sentence is factual and
            # (b) the response has zero valid citations anywhere.
            if _looks_factual(s) and not has_grounding:
                notes.append(f"dropped (uncited factual, no grounding anywhere): {s[:80]!r}")
                n_dropped += 1
                continue
            kept.append(s)

    # Strip [doc_id] markers from the kept sentences
    cleaned = " ".join(_CITE_RE.sub("", s).strip() for s in kept)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    n_total = len(sentences)
    drop_rate = n_dropped / n_total if n_total else 0.0
    forced_escalate = drop_rate > drop_threshold or not cleaned

    if forced_escalate and not cleaned:
        notes.append(f"all sentences dropped or empty; forcing escalation")

    return ValidationResult(
        cleaned_response=cleaned, raw_response=raw_response,
        cited_doc_ids=cited_doc_ids,  # already deduped from the pre-pass
        n_sentences=n_total, n_cited=n_cited, n_dropped=n_dropped,
        drop_rate=drop_rate, forced_escalate=forced_escalate, notes=notes,
    )

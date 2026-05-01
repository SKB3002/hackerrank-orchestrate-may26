"""Corpus loader: walks data/<vendor>/<topic>/*.md, parses frontmatter,
exposes Doc objects with title, breadcrumbs, body, vendor, topic.

Used by both the Approach A baseline (BM25 over titles) and Phase 3
hybrid retrieval (BM25 over chunks + dense embeddings).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Doc:
    doc_id: str  # "claude/amazon-bedrock/10280779"
    vendor: str  # claude | hackerrank | visa
    topic: str  # subfolder under vendor (first-level)
    path: Path  # absolute path
    title: str
    breadcrumbs: list[str] = field(default_factory=list)
    body: str = ""

    @property
    def header_text(self) -> str:
        """Title + breadcrumbs — the highest-signal part for BM25."""
        crumbs = " > ".join(self.breadcrumbs) if self.breadcrumbs else ""
        return f"{self.title}\n{crumbs}".strip()

    @property
    def first_paragraph(self) -> str:
        """First non-empty paragraph after the H1, used for templated responses."""
        body = self.body
        # Skip _Last updated_ lines
        body = re.sub(r"^_Last updated:[^\n]*\n", "", body, flags=re.MULTILINE)
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        # Skip empty H1, take first paragraph that isn't all-heading
        for p in paragraphs:
            if p.startswith("#"):
                continue
            return p
        return ""


def _parse_one(path: Path) -> Doc | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
    rel = path.relative_to(DATA_DIR)
    parts = rel.parts
    vendor = parts[0]
    topic = parts[1] if len(parts) > 2 else "uncategorized"
    stem = path.stem
    # doc_id = vendor/topic/stem-prefix-id-only (article id is the leading number)
    article_id = stem.split("-", 1)[0] if "-" in stem else stem
    doc_id = f"{vendor}/{topic}/{article_id}"

    title = stem.replace("-", " ")  # fallback
    breadcrumbs: list[str] = []
    body = text
    m = _FRONTMATTER_RE.match(text)
    if m:
        try:
            fm = yaml.safe_load(m.group(1)) or {}
            title = fm.get("title", title) or title
            breadcrumbs = fm.get("breadcrumbs", []) or []
            body = text[m.end():]
        except yaml.YAMLError:
            pass
    # Strip the leading H1 line if it duplicates the title
    body = re.sub(r"^#\s+[^\n]+\n", "", body.lstrip(), count=1)
    return Doc(doc_id=doc_id, vendor=vendor, topic=topic, path=path, title=title.strip(), breadcrumbs=breadcrumbs, body=body.strip())


def load_corpus(data_dir: Path = DATA_DIR) -> list[Doc]:
    docs: list[Doc] = []
    for p in data_dir.rglob("*.md"):
        d = _parse_one(p)
        if d is not None:
            docs.append(d)
    return docs


def vendor_filter(docs: list[Doc], vendor: str) -> list[Doc]:
    """Filter docs by vendor name. 'None' or unknown returns all docs."""
    v = (vendor or "").strip()
    if v.lower() in {"", "none"}:
        return docs
    target = v.lower()
    return [d for d in docs if d.vendor.lower() == target]

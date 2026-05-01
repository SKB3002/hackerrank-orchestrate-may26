"""Hybrid retrieval: BM25 (lexical) + BGE-small (dense) + RRF fusion.

Phase 3 module. Chunks each doc into ~600-token windows with 100-token
overlap. Indexes both lexically (rank_bm25) and densely (BGE-small-en-v1.5
via sentence-transformers). At query time, combines both rankings with
Reciprocal Rank Fusion (RRF) at k=60.

Determinism: model and chunker config are recorded in index/manifest.json.
"""

from __future__ import annotations

import json
import pickle
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from code.agent.corpus import Doc, load_corpus

ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "code" / "index"

# Chunker config (locked: change → rebuild index → manifest hash bump).
CHUNK_TOKENS = 600
CHUNK_OVERLAP = 100
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384  # BGE-small dimension
RRF_K = 60  # standard RRF constant; higher = flatter fusion

_TOKEN_RE = re.compile(r"\S+")  # whitespace-token approximation; fine for chunking budget
_BM25_TOK_RE = re.compile(r"[a-z0-9]+")


def _tokenize_for_bm25(text: str) -> list[str]:
    return _BM25_TOK_RE.findall(text.lower())


@dataclass
class Chunk:
    chunk_id: str  # doc_id#chunk_idx
    doc_id: str
    vendor: str
    topic: str
    title: str
    text: str  # the actual chunk body
    breadcrumbs: list[str] = field(default_factory=list)


def _chunk_doc(doc: Doc) -> list[Chunk]:
    """Sliding-window chunker on whitespace tokens. Always emits ≥1 chunk per doc."""
    body = doc.body or ""
    # Prepend the title to every chunk so retrieval sees title-as-context
    head = doc.title.strip()
    crumbs = " > ".join(doc.breadcrumbs) if doc.breadcrumbs else ""
    prefix = f"{head}\n{crumbs}\n\n" if crumbs else f"{head}\n\n"

    tokens = _TOKEN_RE.findall(body)
    if not tokens:
        # Title-only doc — still emit one chunk so the doc is searchable
        return [Chunk(chunk_id=f"{doc.doc_id}#0", doc_id=doc.doc_id, vendor=doc.vendor,
                     topic=doc.topic, title=doc.title, text=prefix.strip(),
                     breadcrumbs=list(doc.breadcrumbs))]

    out: list[Chunk] = []
    step = CHUNK_TOKENS - CHUNK_OVERLAP
    idx = 0
    cidx = 0
    while idx < len(tokens):
        window = tokens[idx: idx + CHUNK_TOKENS]
        text = prefix + " ".join(window)
        out.append(Chunk(
            chunk_id=f"{doc.doc_id}#{cidx}",
            doc_id=doc.doc_id,
            vendor=doc.vendor,
            topic=doc.topic,
            title=doc.title,
            text=text,
            breadcrumbs=list(doc.breadcrumbs),
        ))
        if idx + CHUNK_TOKENS >= len(tokens):
            break
        idx += step
        cidx += 1
    return out


@dataclass
class Hit:
    chunk: Chunk
    bm25_rank: int | None = None
    dense_rank: int | None = None
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0


class HybridRetriever:
    """BM25 + BGE-small dense + RRF fusion. Persists index to code/index/.

    Build once via `build()`; load fast via `load()`. The dense path uses
    sentence-transformers — first call may download the model.
    """

    def __init__(self, chunks: list[Chunk], bm25: BM25Okapi, dense_matrix: np.ndarray, embed_model_name: str = EMBED_MODEL_NAME):
        self.chunks = chunks
        self.bm25 = bm25
        self.dense = dense_matrix  # shape (N, D), L2-normalized so dot = cosine
        self.embed_model_name = embed_model_name
        self._embedder = None  # lazy

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer  # heavy import, lazy
            self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder

    def _encode_query(self, query: str) -> np.ndarray:
        # BGE recommends a query prefix for retrieval tasks
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        emb = self.embedder.encode([prefixed], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)[0]

    def search(self, query: str, top_k: int = 5, vendor: str | None = None,
               bm25_top: int = 50, dense_top: int = 50) -> list[Hit]:
        if not query.strip():
            return []
        # Vendor mask: indices we're allowed to consider. If vendor is None or unknown, no mask.
        if vendor and vendor.strip().lower() in {"hackerrank", "claude", "visa"}:
            v = vendor.strip().lower()
            mask = np.array([c.vendor.lower() == v for c in self.chunks], dtype=bool)
        else:
            mask = None

        # ---- BM25 ----
        bm25_scores = self.bm25.get_scores(_tokenize_for_bm25(query))
        if mask is not None:
            bm25_scores = bm25_scores.copy()
            bm25_scores[~mask] = -np.inf
        bm25_order = np.argsort(-bm25_scores)[:bm25_top]
        bm25_rank = {int(i): r + 1 for r, i in enumerate(bm25_order) if bm25_scores[i] > 0}

        # ---- Dense ----
        q_emb = self._encode_query(query)
        dense_scores = self.dense @ q_emb  # cosine via normalized dot
        if mask is not None:
            dense_scores = dense_scores.copy()
            dense_scores[~mask] = -np.inf
        dense_order = np.argsort(-dense_scores)[:dense_top]
        dense_rank = {int(i): r + 1 for r, i in enumerate(dense_order)}

        # ---- RRF fusion ----
        candidates = set(bm25_rank.keys()) | set(dense_rank.keys())
        fused = []
        for i in candidates:
            br = bm25_rank.get(i)
            dr = dense_rank.get(i)
            rrf = 0.0
            if br is not None:
                rrf += 1.0 / (RRF_K + br)
            if dr is not None:
                rrf += 1.0 / (RRF_K + dr)
            fused.append((i, rrf, br, dr, float(bm25_scores[i]) if bm25_scores[i] != -np.inf else 0.0,
                          float(dense_scores[i]) if dense_scores[i] != -np.inf else 0.0))

        fused.sort(key=lambda t: -t[1])
        out: list[Hit] = []
        seen_docs: set[str] = set()
        for i, rrf, br, dr, bs, ds in fused:
            chunk = self.chunks[i]
            # De-dup by doc_id to avoid 5 chunks of the same article dominating top-k
            if chunk.doc_id in seen_docs:
                continue
            seen_docs.add(chunk.doc_id)
            out.append(Hit(chunk=chunk, bm25_rank=br, dense_rank=dr, bm25_score=bs, dense_score=ds, rrf_score=rrf))
            if len(out) >= top_k:
                break
        return out

    # ---- persistence -----------------------------------------------------

    def save(self, dir: Path = INDEX_DIR) -> None:
        dir.mkdir(parents=True, exist_ok=True)
        with (dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        with (dir / "bm25.pkl").open("wb") as f:
            pickle.dump(self.bm25, f)
        np.save(dir / "dense.npy", self.dense)

    @classmethod
    def load(cls, dir: Path = INDEX_DIR) -> "HybridRetriever":
        chunks = []
        with (dir / "chunks.jsonl").open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                chunks.append(Chunk(**d))
        with (dir / "bm25.pkl").open("rb") as f:
            bm25 = pickle.load(f)
        dense = np.load(dir / "dense.npy")
        manifest = json.loads((dir / "manifest.json").read_text(encoding="utf-8"))
        embed_model_name = manifest.get("embed_model", EMBED_MODEL_NAME)
        return cls(chunks=chunks, bm25=bm25, dense_matrix=dense, embed_model_name=embed_model_name)

    @classmethod
    def build(cls, dir: Path = INDEX_DIR, embed_model_name: str = EMBED_MODEL_NAME) -> "HybridRetriever":
        from sentence_transformers import SentenceTransformer  # heavy import, only on build

        t0 = time.time()
        docs = load_corpus()
        # Drop the junk 'index' doc
        docs = [d for d in docs if d.title and d.title.lower() != "index"]
        print(f"[build_index] loaded {len(docs)} docs")

        chunks: list[Chunk] = []
        for d in docs:
            chunks.extend(_chunk_doc(d))
        print(f"[build_index] chunked into {len(chunks)} chunks (target {CHUNK_TOKENS} tokens, overlap {CHUNK_OVERLAP})")

        # BM25 over chunk text
        tokenized = [_tokenize_for_bm25(c.text) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        print(f"[build_index] BM25 ready in {time.time() - t0:.1f}s")

        # Dense embeddings — batch through BGE-small
        t1 = time.time()
        embedder = SentenceTransformer(embed_model_name)
        # BGE recommends NO query prefix on documents (only queries)
        embs = embedder.encode([c.text for c in chunks], batch_size=64, normalize_embeddings=True,
                               show_progress_bar=True, convert_to_numpy=True)
        embs = np.asarray(embs, dtype=np.float32)
        print(f"[build_index] dense embeddings ready in {time.time() - t1:.1f}s shape={embs.shape}")

        retriever = cls(chunks=chunks, bm25=bm25, dense_matrix=embs, embed_model_name=embed_model_name)
        retriever.save(dir)

        # Manifest
        from datetime import datetime, timezone
        manifest = {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "n_docs": len(docs),
            "n_chunks": len(chunks),
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_overlap": CHUNK_OVERLAP,
            "embed_model": embed_model_name,
            "embed_dim": int(embs.shape[1]),
            "rrf_k": RRF_K,
        }
        (dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[build_index] wrote manifest -> {dir/'manifest.json'}")
        print(f"[build_index] total elapsed: {time.time() - t0:.1f}s")
        return retriever

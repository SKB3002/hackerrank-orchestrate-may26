"""Build the full determinism manifest.

Combines retrieval index manifest + LLM/prompt fingerprints + threshold +
eval score (if available) → code/index/manifest.json.

Run after final scoring is done. Idempotent.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "code" / "index"
PROMPTS_DIR = ROOT / "code" / "llm" / "prompts"
POLICY_FILE = ROOT / "code" / "policies" / "escalation.yaml"
INJECTION_CORPUS = ROOT / "code" / "policies" / "prompt_injection_corpus.txt"
REQUIREMENTS = ROOT / "code" / "requirements.txt"


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _sha256_dir_files(d: Path, glob: str = "*") -> dict[str, str]:
    return {p.name: _sha256_file(p) for p in sorted(d.glob(glob)) if p.is_file()}


def main() -> None:
    base = INDEX_DIR / "manifest.json"
    if base.exists():
        manifest = json.loads(base.read_text(encoding="utf-8"))
    else:
        manifest = {}

    manifest.setdefault("built_at", datetime.now(timezone.utc).isoformat())
    manifest["finalized_at"] = datetime.now(timezone.utc).isoformat()

    # Determinism config (locked across runs)
    manifest["determinism"] = {
        "temperature": 0.2,
        "seed": 42,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    # Models
    manifest["models"] = {
        "gatekeeper": "llama-3.1-8b-instant",
        "answerer": "llama-3.3-70b-versatile",
        "judge_offline_only": "qwen-2.5-32b",
        "embedder": manifest.get("embed_model", "BAAI/bge-small-en-v1.5"),
    }

    # Prompt + policy fingerprints
    manifest["prompt_hashes"] = _sha256_dir_files(PROMPTS_DIR, "*.txt")
    manifest["policy_hashes"] = {
        "escalation.yaml": _sha256_file(POLICY_FILE),
        "prompt_injection_corpus.txt": _sha256_file(INJECTION_CORPUS),
    }

    # Requirements lock
    if REQUIREMENTS.exists():
        manifest["requirements_sha256"] = _sha256_file(REQUIREMENTS)

    # Retrieval config (already in base manifest from build_index)
    # Just ensure the key fields exist
    manifest.setdefault("rrf_k", 60)
    manifest.setdefault("chunk_tokens", 600)
    manifest.setdefault("chunk_overlap", 100)

    # Pipeline thresholds
    manifest["thresholds"] = {
        "low_gate_confidence": 0.30,
        "gate_escalate_min_confidence": 0.50,
        "validator_drop_threshold": 0.60,
        "answerer_max_tokens": 1500,
        "gatekeeper_max_tokens": 512,
    }

    # Latest eval scores, if present
    eval_file = ROOT / "runs" / "llm_diff_full.md"
    if eval_file.exists():
        manifest["last_dev_eval_file"] = str(eval_file.relative_to(ROOT))

    base.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[manifest] wrote {base}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

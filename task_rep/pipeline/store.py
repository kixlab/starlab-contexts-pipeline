"""
Versioned result and checkpoint persistence.

A checkpoint is restored only when the complete fingerprint matches: KB content
hash, pipeline config (minus settings that cannot change the outcome, such as
``use_cache``), model identities (already inside the config), rendered-prompt
hash, IRR sampling settings, and serialization version. No mutable facet bank
is shared across prompt or config versions.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from knowledge_items import KnowledgeBase

from optimization import irr
from utils.llm import SEED

from .config import PipelineConfig
from .models import PipelineState, PipelineResult

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "2"


def kb_fingerprint(kb: KnowledgeBase) -> str:
    """Content hash over a canonical serialization of the knowledge base
    (sorted keys, items ordered by id)."""
    items = sorted(kb.to_dict()["items"], key=lambda item: item["id"])
    blob = json.dumps(items, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class PipelineStore:
    """Owns checkpoint/result paths and the exact-run fingerprint."""

    def __init__(
        self,
        cache_dir: str | Path,
        config: PipelineConfig,
        kb_fp: str,
        prompt_fp: str,
    ):
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.kb_fp = kb_fp
        self.prompt_fp = prompt_fp
        self.fingerprint = self._compute_fingerprint()

    def _compute_fingerprint(self) -> str:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kb": self.kb_fp,
            "prompt": self.prompt_fp,
            "config": self.config.semantic_dict(),
            # Code-level sampling settings (the run seed is in the config).
            "sampling": {"irr_temperature": irr.IRR_TEMPERATURE, "seed_base": SEED},
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]

    def checkpoint_path(self) -> Path:
        return self.cache_dir / f"pipeline_ckpt_{self.fingerprint}.json"

    def result_path(self) -> Path:
        return self.cache_dir / f"pipeline_result_{self.fingerprint}.json"

    # ---- checkpoints ----

    def save_checkpoint(self, state: PipelineState) -> None:
        state.fingerprint = self.fingerprint
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": self.fingerprint, "state": state.to_dict()},
                      f, ensure_ascii=False, indent=2)

    def restore_checkpoint(self) -> Optional[PipelineState]:
        path = self.checkpoint_path()
        if not path.exists():
            logger.info("No checkpoint at %s.", path)
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("fingerprint") != self.fingerprint:
            logger.info("Checkpoint fingerprint mismatch — ignoring %s.", path)
            return None
        logger.info("Restored checkpoint from %s.", path)
        return PipelineState.from_dict(data["state"])

    # ---- results ----

    def save_result(self, result: PipelineResult) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.result_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("Saved result to %s.", path)
        return path

    @staticmethod
    def load_result(path: str | Path) -> PipelineResult:
        data = json.loads(Path(path).read_text())
        return PipelineResult.from_dict(data)

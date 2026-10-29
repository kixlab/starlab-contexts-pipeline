"""
Offline demo: run a0/a1/a2 against a real cached knowledge base using fake
(no-VLM) services that replay cached facets, vocabularies, and labels.

Usage:
    python -m pipeline.offline_demo                 # run a0, a1, a2
    python -m pipeline.offline_demo --strategy a1   # run one
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

from config import Config

from construct import ConstructorConfig

from knowledge_items import KnowledgeItemType

from contextual.fakes import (
    load_corpus, PassthroughVocabularyBuilder, ReplayLabeler, HashingEmbedder,
)
from contextual.kb import load_knowledge_base
from contextual.reporting import summarize

from .config import PipelineConfig, ProposalStrategy, A1_CONTEXT, A2_CONTEXT
from .experiments import load_experiment, run_pipeline
from .fakes import ReplayProposer

logger = logging.getLogger(__name__)

TASK_ID = "sushi_roll"
EXPERIMENT_JSON = Path(__file__).parent.parent / "experiments" / "sushi_roll_step_gemini_a0.json"
CORPUS_GLOB = str(
    Path(Config.CACHE_DIR)
    / "sushi_roll__vision_gemini3.1propreview"
    / "run_result_*facets+items*_30_30.json"
)

# Small budgets so the demo exercises both the fill phase and greedy replacement.
DEMO_PROPOSAL_BUDGET = 8
DEMO_POOL_SIZE = 5


def _constructor_config() -> ConstructorConfig:
    return load_experiment(EXPERIMENT_JSON).constructor


def _pipeline_config(constructor: ConstructorConfig, strategy: str) -> PipelineConfig:
    item_scope = load_experiment(EXPERIMENT_JSON).pipeline.item_scope

    strat = ProposalStrategy.BATCH if strategy == "a0" else ProposalStrategy.ITERATIVE
    context = A2_CONTEXT if strategy == "a2" else A1_CONTEXT

    return PipelineConfig(
        proposal_model=constructor.model_config,      # replayed; identity only
        vocab_model=constructor.model_config,
        labeling_model=constructor.model_config,
        embedding_model=constructor.cluster_config.model_config,  # reuse clusterer
        item_scope=item_scope,
        strategy=strat,
        context=context,
        proposal_budget=DEMO_PROPOSAL_BUDGET,
        pool_size=DEMO_POOL_SIZE,
        attempt_limit=50,
        no_improvement_patience=6,
        refusal_patience=3,
        irr_sample_videos=0,
        metric_mode="entropy",
        refinement_threshold=0.2,
        max_schema_facets=1,
        use_cache=False,  # demo: always run fresh so counters are visible
    )


def run_one(strategy: str, corpus_path: str) -> None:
    constructor = _constructor_config()
    task, kb, cache_dir = load_knowledge_base(
        TASK_ID, constructor, KnowledgeItemType.STEP
    )
    facets, annotations = load_corpus(corpus_path)

    config = _pipeline_config(constructor, strategy)
    proposer = ReplayProposer(corpus=facets)
    vocab_builder = PassthroughVocabularyBuilder()
    labeler = ReplayLabeler(annotations)
    embedder = HashingEmbedder()

    # Write demo artifacts to a local scratch dir, never the shared KB cache.
    scratch = Path(__file__).parent.parent / "pipeline_out"
    result = run_pipeline(
        config, task.name, kb, cache_dir,
        proposer=proposer, vocab_builder=vocab_builder,
        labeler=labeler, embedder=embedder,
        output_dir=scratch,
    )
    summary = summarize(result)

    components = "+".join(sorted(c.value for c in config.context))
    print(f"\n===== {strategy.upper()} ({config.strategy.value}"
          f"{'/' + components if strategy != 'a0' else ''}) =====")
    print(json.dumps(summary.to_dict(), indent=2))
    print("ledger:")
    for r in result.ledger:
        tq = r.tree_quality
        extra = f" replaced={r.replaced_facet_id[:8]}" if r.replaced_facet_id else ""
        sim = (f" sim={r.nearest_similarity:.2f}"
               if r.nearest_similarity is not None else "")
        print(f"  attempt {r.attempt_index:>2} | {r.status.value:<16}"
              f" | budget={r.counts_against_budget!s:<5}"
              f" | tree_nodes={tq.tree_nodes if tq else '-'}"
              f" | pool={len(tq.pool_ids) if tq else '-'}{extra}{sim}"
              f" | {r.title[:32] if r.title else ''}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["a0", "a1", "a2", "all"], default="all")
    args = parser.parse_args()

    matches = sorted(glob.glob(CORPUS_GLOB))
    if not matches:
        raise SystemExit(f"No corpus run_result found matching {CORPUS_GLOB}")
    # Pick the richest corpus (most facets) so small demo budgets can fill.
    corpus_path = max(matches, key=lambda p: len(json.loads(Path(p).read_text())["facets"]))
    n_facets = len(json.loads(Path(corpus_path).read_text())["facets"])
    print(f"Corpus: {Path(corpus_path).name} ({n_facets} facets)")

    strategies = ["a0", "a1", "a2"] if args.strategy == "all" else [args.strategy]
    for s in strategies:
        run_one(s, corpus_path)


if __name__ == "__main__":
    main()

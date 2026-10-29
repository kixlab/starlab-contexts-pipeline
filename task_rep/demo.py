"""Run the real pipeline with scripted services over synthetic data, offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from export_bundle import export_bundle
from facets import AnnotationTable, Facet
from knowledge_items import ItemScopeConfig, KnowledgeBase
from pipeline.config import PipelineConfig
from pipeline.runner import PipelineRunner
from utils import ModelConfig

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class DemoServices:
    """Fixed proposals and labels; explicitly simulated, never calls an API."""

    def __init__(self, source):
        self.source = source
        self.facets = [Facet.from_dict(facet) for facet in source["facets"]]
        self.dimensions = {}

    def propose_batch(self, *args, **kwargs):
        return list(self.facets)

    def build_vocabulary(self, task, facets, knowledge_base, **kwargs):
        return facets

    def label_batch(self, task, facets, knowledge_base, **kwargs):
        allowed = {facet.id for facet in facets}
        return AnnotationTable.from_list([row for row in self.source["annotations"] if row[1] in allowed])

    def embed(self, texts):
        vectors = np.zeros((len(texts), 32))
        for row, text in enumerate(texts):
            dimension = self.dimensions.setdefault(text, len(self.dimensions))
            vectors[row, dimension] = 1
        return vectors


def run_demo(output: Path) -> Path:
    source = json.loads((EXAMPLES / "demo-source.json").read_text())
    services = DemoServices(source)
    model = ModelConfig(category="vlm", provider="openai", model="offline-demo")
    config = PipelineConfig(
        proposal_model=model, vocab_model=model, labeling_model=model,
        embedding_model=ModelConfig(category="embedding", provider="openai", model="offline-demo"),
        item_scope=ItemScopeConfig.from_dict({"item_type": "step", "atomic_subtype": "all"}),
        strategy="batch", proposal_budget=2, pool_size=2, irr_sample_videos=0,
        max_schema_facets=2, refinement_threshold=0.55,
    )
    output.mkdir(parents=True, exist_ok=True)
    runner = PipelineRunner(config=config, task="Synthetic paper folding", knowledge_base=KnowledgeBase.from_dict(source["knowledge_base"]),
                            proposer=services, vocab_builder=services, labeler=services, embedder=services,
                            runs_dir=output / "runs", name="synthetic-demo", fresh=True)
    result = runner.run()
    (output / "knowledge-base.json").write_text(json.dumps(source["knowledge_base"], indent=2) + "\n")
    bundle = export_bundle(result.to_dict(), source["knowledge_base"], source["dataset"])
    bundle_path = output / "demo-bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2) + "\n")
    print(f"Offline example: {len(result.pool)} contextual units, {len(bundle['labeled_dataset'])} tutorials")
    print(f"Result: {runner.folder.path / 'result.json'}")
    print(f"UI bundle: {bundle_path}")
    return bundle_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "static" / "demo")
    run_demo(parser.parse_args().output)


if __name__ == "__main__":
    main()

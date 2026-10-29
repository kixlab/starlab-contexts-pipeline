# Task representation pipeline

Start with the [repository README](../README.md) for installation, an offline
example, model-backed experiments, exports, and the web explorer.

`pipeline/` implements batch proposals and iterative feedback using selected
context components: `facets`, `items`, `schemas`, and `history`. The shared
`contextual/` package handles vocabulary construction, labeling, deduplication,
inter-rater checks, candidate selection, checkpoints, and evaluation. It depends
on `facets/`, `knowledge_items/`, `optimization/`, and `utils/`.

From this directory, `python -m pipeline.cli --help` shows commands and
`python -m pytest tests -q` runs the offline suite. The two separate experimental
agent runners are excluded from this distribution. Reports support the public
prompt pipeline without importing either agent.

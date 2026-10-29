# Contextual-unit discovery Pipeline

Start with the [repository README](../README.md) for installation, an offline
example, model-backed experiments, exports, and the design-space explorer.

Knowledge items retain their source tutorials and timestamps; knowledge units
group semantically related items. Contextual units describe items using canonical
values. Candidate organizations combine contextual units for comparison and
selection before TaskTailor configuration.

`pipeline/` implements batch proposals and iterative feedback using selected
context components: `facets`, `items`, `schemas`, and `history`. The shared
`contextual/` package handles vocabulary construction, labeling, deduplication,
inter-rater checks, candidate selection, checkpoints, and evaluation. It depends
on `facets/`, `knowledge_items/`, `optimization/`, and `utils/`.

From this directory, `python -m pipeline.cli --help` shows commands and
`python -m pytest tests -q` runs the offline suite. Reports consume the prompt pipeline's saved results, while `contextual/`
remains independent of runner-specific modules.

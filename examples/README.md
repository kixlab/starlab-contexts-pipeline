# Makeup pipeline example

`demo-dataset.json` contains tutorial metadata and timed step descriptions for
three makeup tutorials. `demo-source.json` contains their 50 knowledge items,
30 contextual units, and saved annotations. `demo-bundle.json` is the matching
bundle for the graph explorer and TaskTailor.

Run `python main.py demo` from the repository root to exercise the pipeline with
saved proposals and labels. The run writes its evaluated result, knowledge base,
and exported UI bundle under `task_rep/static/demo/`.

The full-length demo videos are committed in `explorer/public/media/makeup/`.
See [tutorial sources and media details](../explorer/public/data/README.md).

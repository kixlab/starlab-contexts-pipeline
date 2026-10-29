# Annotated tutorial bundle

The graph explorer and three applications accept one JSON object with these
keys. A contextual unit (CU) is stored as a facet; a schema is a set of CU IDs.
The applications can also consume manually annotated or independently produced
bundles; they do not require this pipeline to run.

| Key | Contents |
| --- | --- |
| `context_schema` | Default contextual-unit selection; TaskTailor initially chooses up to two CUs |
| `facet_candidates` | Available contextual units, stored as facet objects |
| `labeled_dataset` | Source tutorials and their timed knowledge items |

A contextual unit is serialized as a facet containing `id`, `type`, `title`, `title_plural`, `definition`, optional
`guidelines`, and `vocabulary` entries with `label` and `definition` for its canonical values.

A tutorial contains `id`, `url`, `title`, `original_title`, `content`,
`transcript` (`text`, `start`, `end` entries), and `pieces`. A piece contains
`piece_id`, `unit_id`, `item_type`, `content_type`, `start`, `end`, `content`,
and `labels` (a facet-ID-to-string-array map). Times are seconds. `unit_id`
groups semantically related knowledge items into knowledge units across tutorials.
It is distinct from a contextual-unit ID. CU IDs in `labels` must match the
available CU IDs; assigned values must come from that CU's canonical vocabulary.
Every CU in the default selection must also appear among the available CUs.
The arrays normally contain one canonical value per CU, with N/A defined when an
aspect does not apply; older multi-valued bundles remain supported. Field names
such as `facets`, `pieces`, and `context_schema` retain their existing spellings.

Video URLs may be YouTube URLs or local/self-hosted video URLs. Relative URLs
are served from the web app's `public/` directory. The included sample uses
`/media/folding-demo.mp4`; real YouTube playback needs internet access. Empty
URLs are valid for text-only inspection but cannot play a video.

`examples/demo-bundle.json` is the complete synthetic reference fixture.
`python main.py export --help` describes exporting the current pipeline's
separate `result.json` and knowledge-base JSON/JSONL to this format. Its output
contains no run conversations, model request logs, embeddings, or frame paths.
Knowledge text and explicitly supplied video URLs remain part of an export.

The browser normalizer also accepts older step bundles with `facets`,
`knowledge_base`, `annotations`, and `iterations`. Export current run results
before loading them: a raw `result.json` does not contain its knowledge base.

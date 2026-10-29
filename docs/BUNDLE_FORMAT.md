# Shared interface bundle

All public interfaces accept one JSON object with these keys:

| Key | Contents |
| --- | --- |
| `context_schema` | Facets initially available for inspection |
| `facet_candidates` | Candidate facets available for schema selection |
| `labeled_dataset` | Source tutorials and their timed knowledge pieces |

A facet contains `id`, `type`, `title`, `title_plural`, `definition`, optional
`guidelines`, and `vocabulary` entries with `label` and `definition`.

A tutorial contains `id`, `url`, `title`, `original_title`, `content`,
`transcript` (`text`, `start`, `end` entries), and `pieces`. A piece contains
`piece_id`, `unit_id`, `item_type`, `content_type`, `start`, `end`, `content`,
and `labels` (a facet-ID-to-string-array map). Times are seconds. `unit_id`
links related knowledge across tutorials. Facet IDs in `labels` must match the
schema IDs; label values must match that facet's vocabulary.

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

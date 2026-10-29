# Starlab Contexts Pipeline

Build Context Information Maps (CIMs) from tutorial transcripts: extract knowledge
items, cluster shared knowledge across sources, propose contextual factors, label
items, and construct a refinement graph of factor combinations. An included web
explorer lets you inspect the resulting knowledge and design space.

The current implementation is in `task_rep/`. The earlier pipeline remains in
[`legacy/`](legacy/README.md). The separate `starlab-contexts-applications` repo
contains retrieval, segmentation/augmentation, and schema analysis interfaces.
Both repos consume the same JSON bundle and run independently.

## Quick start: offline example

Requirements: Python 3.10 or newer. No GPU, model account, downloaded videos, or
API keys are needed for this example.

```bash
python3 -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python main.py demo
```

The demo uses the real pipeline runner with explicitly scripted proposals,
vocabularies, labels, and embeddings. It exercises evaluation, checkpoints, and
export without pretending to generate model results. It writes a run folder,
`knowledge-base.json`, and `demo-bundle.json` under `task_rep/static/demo/`.
The four tutorials and the video title card are entirely synthetic.

`requirements-lock.txt` records the tested Python environment, including pytest.
For a smaller application install use `requirements.txt`; for video download and
frame extraction also install `requirements-video.txt` and your OS's `ffmpeg`.
Model API calls can run on CPU. Video download uses Node.js as yt-dlp's JS runtime.

## Explore the results

Install Node.js 22 and npm, then:

```bash
cd explorer
npm ci
npm run dev
```

Open <http://localhost:3000>. Choose **Inspect knowledge** or **Explore design
space**. The graph page loads the synthetic dataset; press **Build** to enumerate
and display the graph. The bundled sample also includes a local playable MP4.

To inspect a new demo run, copy `task_rep/static/demo/demo-bundle.json` to
`explorer/public/data/local-bundle.json`, then copy `.env.local.example` to
`.env.local` inside `explorer/` and set:

```dotenv
NEXT_PUBLIC_BUNDLE_URL=/data/local-bundle.json
```

Restart the web server and choose **Local pipeline output**. The same file works
in `starlab-contexts-applications/public/data/local-bundle.json`. These filenames
are ignored by Git. Browser applications require no model API keys.

## Run with model APIs

The public example experiment uses OpenAI for extraction, clustering, proposal,
vocabulary, and labeling. Export `OPENAI_API_KEY` in your shell, or set it through
your environment manager; see `.env.example`. The Python CLI reads environment
variables and does not automatically load dotenv files. Other configurations can
use Gemini (`GEMINI_API_KEY`) or Anthropic through AWS Bedrock (your own AWS
credentials and region). No credentials are bundled.

```bash
# From the repository root, with the virtual environment activated:
python scripts/prepare_demo_data.py
python main.py list
python main.py run paper_folding_demo
python main.py show paper_folding_demo
python main.py tree paper_folding_demo --tau 0.2 0.4 0.6
```

Only `run` above calls model APIs and incurs their usage charges. Edit model IDs,
concurrency, budget, and item scope in
`task_rep/experiments/paper_folding_demo.json` to fit your account. `run` resumes
interrupted runs; `--fresh` starts a new one. The example uses text only and
disables the optional inter-rater check to keep the run small. The `demo` command
always uses scripted services, regardless of environment credentials.

For your own corpus, define a task in `task_rep/data/tasks.json`, put a transcript
dataset in `task_rep/static/task-datasets/<task_id>.json` (see
`examples/demo-dataset.json`), and copy the example experiment with matching
`name` and `task_id`. Keep private task definitions in `task_rep/data/local-tasks.json` and export
`CIM_TASKS_FILE` pointing to it. Use `local-*.json` for private experiment files;
add individual ignore entries for other private filenames. Do not commit a dataset
whose redistribution rights or privacy status you have not reviewed.

Set `CIM_STORAGE_DIR` to keep datasets, cached knowledge, run results, and model
request logs in an external directory. Defaults are resolved relative to the
installed repo, so `python /path/to/repo/main.py ...` also works from another
working directory. Dataset downloads are optional; see
[`task_rep/data_prep/README.md`](task_rep/data_prep/README.md).

## Export a model run

A saved `result.json` contains labels and schema structure; the knowledge base
is stored separately. The run's `run.json` records the relative `sources.units`
JSONL path, under `task_rep/static/cache_2/<task_id>/`.

```bash
python main.py export path/to/result.json \
  --knowledge-base path/to/units.jsonl \
  --dataset examples/demo-dataset.json \
  --output explorer/public/data/local-bundle.json
```

`--dataset` supplies video titles and URLs. Omit it for text-only inspection.
Exports include only schema definitions, labels, knowledge text, and source
timings; model conversations, API usage, embeddings, local frame paths, and
arbitrary metadata are omitted. Review the exported text and URLs before sharing
your own data. See [`docs/BUNDLE_FORMAT.md`](docs/BUNDLE_FORMAT.md).

## Layout and checks

| Path | Purpose |
| --- | --- |
| `main.py` | CLI for the current pipeline, offline demo, and bundle exporter |
| `task_rep/pipeline/` | Batch and iterative proposal runners |
| `task_rep/contextual/` | Candidate lifecycle, caching, evaluation, reporting |
| `task_rep/knowledge_items/`, `facets/`, `optimization/` | Extraction, labels, and schema math |
| `task_rep/reports/` | Offline run reports and re-scoring |
| `explorer/` | Knowledge inspector and design-space graph |
| `examples/` | Synthetic input, labels, and a ready-to-open UI bundle |
| `legacy/` | Earlier CIM construction implementation |
| `docs/history/` | Original-to-filtered commit maps |

```bash
python -m pytest -c pytest.ini task_rep/tests -q
python scripts/check_public.py --history
cd explorer
npm run build
# Once per machine:
npx playwright install chromium
npm run test:smoke
```

CI runs the Python suite, offline demo, privacy scan, production web build, and
browser smoke tests. Live model calls and third-party video downloads are not
part of CI. An automated scan detects common secret patterns and excluded paths;
it cannot determine the privacy or copyright status of arbitrary text.

The import preserves relevant development commits and author metadata, with
rewritten hashes where paths or parentage changed. The two agent implementations
and private study data are excluded. See [`docs/HISTORY.md`](docs/HISTORY.md).

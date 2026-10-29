# Starlab Contexts Pipeline

Construct Context Information Maps (CIMs) from tutorial transcripts: extract
knowledge, discover contextual factors, label items, and evaluate organizing
schemas. The included explorer and the separate applications repo consume the
same JSON bundle.

## Quick start: both repos

Use **Python 3.10+** and **Node.js 22 with npm**. Check `python3 --version`; if it
is older, use a supported interpreter such as `python3.12` in the venv command.
The commands below use a macOS/Linux shell. No GPU or API key is needed.

From a parent folder, clone both repos side by side and set up the pipeline:

For existing clones, skip the two clone commands. Preserve existing `.env.local`
files: set `NEXT_PUBLIC_BUNDLE_URL=/data/local-bundle.json` in place of copying the
example file over them. Stop existing dev servers before running setup.

```bash
git clone https://github.com/kixlab/starlab-contexts-pipeline.git
git clone https://github.com/kixlab/starlab-contexts-applications.git
cd starlab-contexts-pipeline
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python main.py demo

cp task_rep/static/demo/demo-bundle.json explorer/public/data/local-bundle.json
cp explorer/.env.local.example explorer/.env.local
cp task_rep/static/demo/demo-bundle.json ../starlab-contexts-applications/public/data/local-bundle.json
cp ../starlab-contexts-applications/.env.local.example ../starlab-contexts-applications/.env.local
npm --prefix explorer ci
npm --prefix ../starlab-contexts-applications ci
```

Keep these commands running in **two terminals**, both starting from the
pipeline repo root:

```bash
# Terminal 1: explorer
npm --prefix explorer run dev -- --port 3001
```

```bash
# Terminal 2: applications
npm --prefix ../starlab-contexts-applications run dev
```

| URL | Interface | First action |
| --- | --- | --- |
| <http://localhost:3001/inspect> | Knowledge inspector | Select a knowledge unit, then expand its source details |
| <http://localhost:3001/space> | Design-space explorer | Press **Build**, then select a graph node |
| <http://localhost:3000/retrieval> | Retrieval | Select a chapter or another tutorial |
| <http://localhost:3000/segmentation> | Segmentation and augmentation | Seek to a chapter and inspect related information |
| <http://localhost:3000/analysis> | Schema analysis | Select a point on the plot |

Each interface opens **Local pipeline output** automatically. The generated
bundle has four synthetic tutorials, sixteen steps, and two contextual factors.
Both viewers include the matching 20-second video title card. Analysis also adds
a Content Type facet for comparison. To run only the bundled sample, skip the
bundle-copy and `.env.local` configuration steps.

The demo runs the real proposal/evaluation/checkpoint pipeline with scripted
model services. It writes `demo-bundle.json`, `knowledge-base.json`, and a run
folder containing `result.json` under `task_rep/static/demo/`. Extraction with
live model APIs is a separate mode below.

## Run with model APIs

Activate the venv and export your own `OPENAI_API_KEY` in the shell. Python reads
environment variables; it does **not** load `.env` automatically. Then:

```bash
python scripts/prepare_demo_data.py
python main.py run paper_folding_demo
python main.py show paper_folding_demo
```

This uses synthetic transcripts but makes paid model calls. Adjust models and
budgets in `task_rep/experiments/paper_folding_demo.json`. `run` resumes interrupted
runs; `--fresh` starts a new one. For your corpus, add a task to
`task_rep/data/tasks.json` and place its transcripts at
`task_rep/static/task-datasets/<task_id>.json`, following `examples/demo-dataset.json`.
Copy the example experiment under `task_rep/experiments/`, set its `task_id` to
your task, and keep its filename and `name` consistent. See
[task preparation](task_rep/data_prep/README.md) and [.env.example](.env.example)
for optional downloads, providers, and external storage configuration.

## Export another run

Combine its `result.json` with the cached units file recorded in that run's
`run.json` (`sources.units`, relative to `task_rep/static/cache_2/<task_id>/`):

```bash
python main.py export path/to/result.json \
  --knowledge-base path/to/units.jsonl \
  --dataset path/to/task-dataset.json \
  --output explorer/public/data/local-bundle.json
```

Copy that bundle into the applications repo as above. `--dataset` supplies video
titles and URLs; omit it for text-only inspection. For your own local video URLs,
copy the referenced media into each viewer's `public/` directory. Configure
`.env.local` **before a production build** and restart dev servers after changes.
The [bundle contract](docs/BUNDLE_FORMAT.md) describes the shared format.

## Checks and other entry points

Stop dev servers with **Ctrl+C** before building or running these checks.

```bash
python -m pytest -q
python scripts/check_public.py --history
npm --prefix explorer run build
npm --prefix explorer exec -- playwright install chromium
npm --prefix explorer run test:smoke
```

For production, run `npm run build` and then `npm run start` in each web app; use
`-- --port 3001` for the explorer. `python main.py --help` lists pipeline commands.
Core code is in [`task_rep/`](task_rep/README.md); the framework-split variant is
in [`legacy/`](legacy/README.md). CI covers offline execution, exports, and browser
interactions. Live model calls and downloads require your own credentials/data.

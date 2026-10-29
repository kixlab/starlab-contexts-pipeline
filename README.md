# Starlab Contexts Pipeline

Construct and interactively explore the **design space of contextual units** for
tutorial system design. This repo contains the discovery Pipeline and the
design-space explorer. The separate **TaskTailor** repo contains contextual
retrieval, video segmentation, and coverage analysis.

The workflow is **tutorial corpus → knowledge-item extraction → contextual-unit
discovery and annotation → design-space exploration → TaskTailor configuration**.
Knowledge items retain source tutorials, descriptions, and timestamps; semantically
related items can be grouped into knowledge units. Contextual units (CUs) describe
items using named aspects, definitions, and canonical values. A candidate
organization selects CUs whose value signatures group the annotated items.

The explorer presents candidate organizations and refinement relationships in a
directed acyclic graph (DAG). Inspect candidates and **Copy selection** to configure
TaskTailor. TaskTailor can also load annotated corpora directly. Building a graph
or using a TaskTailor feature does not invoke discovery or labeling.

Run the model-backed pipeline when a new corpus needs knowledge extraction,
CU discovery, or labeling. Existing annotations can be loaded immediately.
Changing graph controls or selecting different, already labeled CUs needs no rerun.

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
| <http://localhost:3001/space> | Pipeline: design-space exploration | Press **Build**, inspect a candidate, then **Copy selection** |
| <http://localhost:3000/retrieval> | TaskTailor: contextual retrieval | Play or seek to a knowledge item; inspect matches from other tutorials |
| <http://localhost:3000/segmentation> | TaskTailor: video segmentation | Navigate the chapter timeline and expand other tutorials' chapter lists |
| <http://localhost:3000/analysis> | TaskTailor: coverage analysis | Choose two CUs, or normalized temporal position and a CU; click a bar/bubble |

Each interface opens **My tutorials** automatically. The generated
bundle has four synthetic tutorials, sixteen knowledge items, and two contextual units.
Both viewers include the matching 20-second video title card. To run only the bundled sample, skip the
bundle-copy and `.env.local` configuration steps.

The demo runs the real proposal/evaluation/checkpoint pipeline with scripted
model services. It writes `demo-bundle.json`, `knowledge-base.json`, and a run
folder containing `result.json` under `task_rep/static/demo/`. Extraction with
live model APIs is a separate mode below.

## Use your own corpus

1. Collect tutorials about one task. Prepare a JSON array with a unique `id`,
   `title`, `url`, and `transcript` entries `{text, start, end}` for each tutorial.
   Times are seconds; see [`examples/demo-dataset.json`](examples/demo-dataset.json).
   For videos without transcripts, start with [dataset preparation](task_rep/data_prep/README.md).
2. From the repo root, register the corpus:

   ```bash
   python scripts/register_corpus.py /path/to/tutorials.json --task-id local-my-tutorials --name "My tutorials"
   export CIM_TASKS_FILE=task_rep/data/local-tasks.json
   ```

   This writes an ignored transcript dataset and
   `task_rep/experiments/local-my-tutorials.json`. Review models, budgets, item
   scope, and CU limits there. Registration makes no API calls; its default mode
   uses transcripts without vision. Keep `CIM_TASKS_FILE` set for later commands.
   Registration uses default storage; unset `CIM_STORAGE_DIR` if you changed it.
3. Activate the venv and export your own `OPENAI_API_KEY`. Python reads shell
   variables; it does not load `.env` automatically. Invoke the pipeline:

   ```bash
   python main.py run local-my-tutorials
   python main.py show local-my-tutorials
   python main.py export --experiment local-my-tutorials --output explorer/public/data/local-bundle.json
   ```

   `run` makes paid model calls to extract/cluster knowledge, discover CUs,
   construct vocabularies, and label pieces. It resumes interrupted runs;
   `--fresh` starts another run. `show` and `export` read saved results only.
   Export automatically finds the completed run, knowledge base, and video
   metadata. After changing corpus content, use a new task ID to avoid reusing
   its cached extraction/units.
4. Set `NEXT_PUBLIC_BUNDLE_URL=/data/local-bundle.json` and
   `NEXT_PUBLIC_DATASET_LABEL=My tutorials` in `explorer/.env.local`. Restart the
   explorer; at `/space`, choose the **Tutorial collection** and press **Build**.
   **Max contexts** limits contextual units per candidate; start at 1 or 2 for a large
   pool. Nodes are arranged by vocabulary cost (canonical values excluding N/A).
   Inspect a candidate's contextual units, value distributions, and effectiveness;
   follow refinement links and press **Copy selection** for your selection.
5. Copy the same bundle into the applications repo and configure it there.
   In any TaskTailor feature, choose that collection, open **Paste a saved selection**,
   paste the selection, and press **Apply**. You can also choose context names
   directly. The learner-facing views call contextual units **contexts** and
   knowledge items **clips**. See the
   [applications dataset guide](https://github.com/kixlab/starlab-contexts-applications/blob/main/docs/DATASETS.md).

Already have an annotated bundle? Start at step 4. Each viewer has its own JSON
copy and `.env.local`; there is no shared backend. Multiple named datasets use
`NEXT_PUBLIC_DATASETS` in each viewer, as described in the applications guide.
Set configuration before a production build and restart after changes.

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
copy the referenced media into each viewer's `public/media/local/` directory
(ignored by Git) and use matching `/media/local/` URLs. Configure
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

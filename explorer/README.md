# Pipeline explorer

From this directory, use Node.js 22, `npm ci`, then `npm run dev` and open
<http://localhost:3000>.

`/inspect` inspects knowledge units, source clips, facet labels, and coverage.
`/space` enumerates contextual schemas and displays their refinement graph;
press **Build** after loading a dataset. The original `/analysis_info` and
`/app_tree_2` URLs remain available. The bundled dataset and media are synthetic.

To add output from your pipeline, copy the exported bundle to
`public/data/local-bundle.json`, copy `.env.local.example` to `.env.local`, set
`NEXT_PUBLIC_BUNDLE_URL=/data/local-bundle.json`, and restart the server. Choose
**Local pipeline output**. A raw `result.json` must be combined with its knowledge
base through `python main.py export` in the parent repo first.

`npm run build` makes a production build; `npm run start` serves it. Use
`-- --port 3001` to run beside the applications repo. `npm run test:smoke` tests
the production UI after `npx playwright install chromium`. No study logging or
participant data is included. See the [root README](../README.md) for the full
system setup and export instructions.

# Pipeline explorer

Use Node.js 22. From this directory, run `npm ci`, then
`npm run dev -- --port 3001` and open <http://localhost:3001>.

- `/space`: press **Build**, then select a schema node.

The bundled synthetic dataset and local video work without API keys. To use a
generated bundle, follow the [root quick start](../README.md#quick-start-both-repos).
The configured bundle is selected automatically. Select a graph node and press
**Copy IDs** to use that CU combination in the applications' schema toolbar.
Graph construction only reads existing annotations; it makes no model calls.
For a new raw corpus, follow [corpus setup](../README.md#use-your-own-corpus).
For an existing annotated bundle, copy it into `public/data/local-bundle.json`
and set `NEXT_PUBLIC_BUNDLE_URL=/data/local-bundle.json` and
`NEXT_PUBLIC_DATASET_LABEL=My tutorials` in `.env.local`.
`NEXT_PUBLIC_DATASETS` supports multiple named bundles; see the
[dataset guide](https://github.com/kixlab/starlab-contexts-applications/blob/main/docs/DATASETS.md).
Restart after changing `.env.local` and set it before production builds.

For production, use `npm run build`, then `npm run start -- --port 3001`.
After building, `npx playwright install chromium` and `npm run test:smoke` verify
the graph and dataset switching.

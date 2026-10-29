# Pipeline explorer

Use Node.js 22. From this directory, run `npm ci`, then
`npm run dev -- --port 3001` and open <http://localhost:3001>.

- `/inspect`: select a knowledge unit and expand its source details.
- `/space`: press **Build**, then select a schema node.

The bundled synthetic dataset and local video work without API keys. To use a
generated bundle, follow the [root quick start](../README.md#quick-start-both-repos).
The configured bundle is selected automatically; both interfaces expose dataset
controls. Restart after changing `.env.local` and set it before production builds.

For production, use `npm run build`, then `npm run start -- --port 3001`.
After building, `npx playwright install chromium` and `npm run test:smoke` verify
the inspector, graph, and local data loading.

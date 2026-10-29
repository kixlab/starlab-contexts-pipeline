import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import nextEnv from "@next/env";
import { chromium, expect } from "@playwright/test";

const root = fileURLToPath(new URL("..", import.meta.url));
nextEnv.loadEnvConfig(root);
const port = Number(process.env.CIM_SMOKE_PORT || 3811);
const origin = `http://127.0.0.1:${port}`;
const configured = process.env.NEXT_PUBLIC_BUNDLE_URL;
const cases = [
  ...(configured ? [{label: process.env.NEXT_PUBLIC_DATASET_LABEL || "Local dataset", url: configured}] : []),
  ...JSON.parse(process.env.NEXT_PUBLIC_DATASETS || "[]").map(item => ({label: item.label, url: item.bundleUrl})),
  {label: "Synthetic paper folding", url: "/data/demo-bundle.json"},
];
const server = spawn(process.execPath, ["node_modules/next/dist/bin/next", "start", "--hostname", "127.0.0.1", "--port", String(port)], {
  cwd: root, env: {...process.env, NEXT_TELEMETRY_DISABLED: "1"}, stdio: ["ignore", "pipe", "pipe"],
});
let logs = "", browser;
server.stdout.on("data", data => { logs += data; });
server.stderr.on("data", data => { logs += data; });
try {
  let ready = false;
  for (let attempt = 0; attempt < 120; attempt++) {
    if (server.exitCode !== null) throw new Error("Production server exited early");
    try { if ((await fetch(origin)).ok) { ready = true; break; } } catch {}
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  assert.ok(ready, "Production server did not become ready");
  for (const item of cases) {
    const response = await fetch(new URL(item.url, origin));
    assert.ok(response.ok, `Cannot load ${item.url}`);
    item.bundle = await response.json();
    assert.ok(item.bundle.context_schema.length > 0);
    assert.ok(item.bundle.labeled_dataset.length > 0);
  }
  browser = await chromium.launch({headless: true, ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ? {executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE} : {})});
  const context = await browser.newContext({viewport: {width: 1440, height: 1000}, permissions: ["clipboard-read", "clipboard-write"]});
  const errors = [], external = [], badResponses = [], requests = new Set();
  await context.route("**/*", route => {
    const url = route.request().url();
    if (url.startsWith("http") && new URL(url).origin !== origin) {
      external.push(url); return route.abort();
    }
    return route.continue();
  });
  const page = await context.newPage();
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", request => { if (request.url().startsWith("http")) requests.add(new URL(request.url()).pathname); });
  page.on("response", response => { if (response.status() >= 400) badResponses.push(`${response.status()} ${response.url()}`); });
  const visit = async path => { requests.clear(); await page.goto(origin + path); };
  const fetched = async item => {
    await expect.poll(() => requests.has(new URL(item.url, origin).pathname)).toBeTruthy();
  };
  await visit("/");
  await expect(page.getByRole("heading", {level: 1})).toContainText("CIM Pipeline Explorer");
  await expect(page.getByRole("link", {name: "Inspect knowledge"})).toHaveCount(0);
  for (const path of ["/inspect", "/analysis_info"]) assert.equal((await fetch(origin + path)).status, 404);
  await visit("/space");
  const checkGraph = async item => {
    await expect(page.getByRole("combobox", {name: "Dataset"})).toHaveValue(item.label);
    const build = page.getByRole("button", {name: "Build", exact: true});
    await expect(build).toBeEnabled();
    await build.click();
    await expect(page.getByText(/\d+\/\d+ nodes/)).toBeVisible({timeout: 120000});
    await fetched(item);
    await page.getByRole("button", {name: /^Schema:/}).first().click({force: true});
    await page.getByRole("button", {name: "Copy IDs", exact: true}).first().click();
    const ids = await page.evaluate(() => navigator.clipboard.readText());
    const validIds = new Set(item.bundle.facet_candidates.map(f => f.id));
    assert.ok(ids.split(",").every(id => validIds.has(id.trim())), "Copied schema must use this dataset's CU IDs");
  };
  await checkGraph(cases[0]);
  if (cases.length > 1) for (const item of [...cases.slice(1), cases[0]]) {
    requests.clear();
    await page.getByRole("combobox", {name: "Dataset"}).selectOption(item.label);
    await checkGraph(item);
  }
  if (process.env.CIM_SMOKE_SCREENSHOT) await page.screenshot({path: process.env.CIM_SMOKE_SCREENSHOT, fullPage: true});
  assert.deepEqual(errors, [], "Browser runtime errors");
  assert.deepEqual(external, [], "Local example attempted external requests");
  assert.deepEqual(badResponses, [], "Failed local requests");
  console.log("explorer: dataset switching, graph construction, schema selection, CU ID copying, and inspector removal passed");
} catch (error) {
  console.error(logs.slice(-8000));
  throw error;
} finally {
  if (browser) await browser.close();
  server.kill("SIGTERM");
}

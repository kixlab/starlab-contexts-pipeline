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
  ...(configured ? [{label: "Local pipeline output", url: configured}] : []),
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
  const context = await browser.newContext({viewport: {width: 1440, height: 1000}});
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
  const localVideo = async () => {
    const video = page.locator("video").first();
    await expect(video).toBeVisible();
    await expect.poll(() => video.evaluate(element => element.readyState)).toBeGreaterThanOrEqual(1);
    await video.evaluate(element => { element.currentTime = 7; });
    await expect.poll(() => video.evaluate(element => element.currentTime)).toBeCloseTo(7, 1);
  };
  await visit("/");
  await expect(page.getByRole("heading", {level: 1})).toContainText("CIM Pipeline Explorer");
  await visit("/inspect");
  const checkInspector = async item => {
    await expect(page.getByRole("combobox", {name: "Dataset"})).toHaveValue(item.label);
    await expect(page.getByText(item.bundle.context_schema[0].title, {exact: true}).first()).toBeVisible();
    const text = item.bundle.labeled_dataset[0].pieces[0].content;
    await page.getByRole("button", {name: text, exact: false}).first().click();
    await page.getByRole("button", {name: "Expand details"}).first().click();
    await localVideo();
    await fetched(item);
    await expect(page.getByText("placeholder", {exact: true})).toHaveCount(0);
  };
  await checkInspector(cases[0]);
  if (cases.length > 1) {
    for (const item of [cases[1], cases[0]]) {
      requests.clear();
      await page.getByRole("combobox", {name: "Dataset"}).selectOption(item.label);
      await checkInspector(item);
    }
  }
  await visit("/space");
  await expect(page.getByRole("combobox", {name: "Dataset"})).toHaveValue(cases[0].label);
  const build = page.getByRole("button", {name: "Build", exact: true});
  await expect(build).toBeEnabled();
  await build.click();
  await expect(page.getByText(/\d+\/\d+ nodes/)).toBeVisible();
  await fetched(cases[0]);
  if (process.env.CIM_SMOKE_SCREENSHOT) await page.screenshot({path: process.env.CIM_SMOKE_SCREENSHOT, fullPage: true});
  assert.deepEqual(errors, [], "Browser runtime errors");
  assert.deepEqual(external, [], "Local example attempted external requests");
  assert.deepEqual(badResponses, [], "Failed local requests");
  console.log("explorer: default routes, selected bundle, local media, and dataset switching passed");
} catch (error) {
  console.error(logs.slice(-8000));
  throw error;
} finally {
  if (browser) await browser.close();
  server.kill("SIGTERM");
}

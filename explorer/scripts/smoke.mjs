import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { chromium, expect } from "@playwright/test";

const kind = "explorer";
const port = Number(process.env.CIM_SMOKE_PORT || (kind === "explorer" ? 3811 : 3812));
const origin = `http://127.0.0.1:${port}`;
const root = fileURLToPath(new URL("..", import.meta.url));
const server = spawn(process.execPath, ["node_modules/next/dist/bin/next", "start", "--hostname", "127.0.0.1", "--port", String(port)], {
  cwd: root, env: {...process.env, NEXT_TELEMETRY_DISABLED: "1"}, stdio: ["ignore", "pipe", "pipe"],
});
let logs = "";
server.stdout.on("data", data => { logs += data; });
server.stderr.on("data", data => { logs += data; });
let browser;
try {
  let ready = false;
  for (let attempt = 0; attempt < 120; attempt++) {
    if (server.exitCode !== null) throw new Error("Production server exited early");
    try { if ((await fetch(origin)).ok) { ready = true; break; } } catch {}
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  assert.ok(ready, "Production server did not become ready");
  browser = await chromium.launch({headless: true, ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ? {executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE} : {})});
  const context = await browser.newContext({viewport: {width: 1440, height: 1000}});
  const errors = [], external = [], badResponses = [];
  await context.route("**/*", route => {
    const url = route.request().url();
    if (url.startsWith("http") && new URL(url).origin !== origin) {
      external.push(url); return route.abort();
    }
    return route.continue();
  });
  const page = await context.newPage();
  page.on("pageerror", error => errors.push(error.message));
  page.on("response", response => { if (response.status() >= 400) badResponses.push(`${response.status()} ${response.url()}`); });
  await page.goto(origin);
  await expect(page.getByRole("heading", {level: 1})).toContainText(kind === "explorer" ? "CIM Pipeline Explorer" : "CIM Applications");
  const bundle = await (await fetch(`${origin}/data/demo-bundle.json`)).json();
  assert.equal(bundle.labeled_dataset.length, 4);
  assert.equal(bundle.labeled_dataset.flatMap(video => video.pieces).length, 16);
  const query = "?dataset=Synthetic%20paper%20folding&schema=paper,stage";
  if (kind === "applications") {
    for (const route of ["retrieval", "segmentation"]) {
      await page.goto(`${origin}/${route}${query}`);
      const video = page.locator("video").first();
      await expect(video).toBeVisible();
      await expect.poll(() => video.evaluate(element => element.readyState)).toBeGreaterThanOrEqual(1);
      await video.evaluate(element => { element.currentTime = 7; });
      await expect.poll(() => video.evaluate(element => element.currentTime)).toBeCloseTo(7, 1);
      assert.ok((await page.locator("body").innerText()).includes("Synthetic"));
    }
    await page.goto(`${origin}/analysis`);
    await expect(page.getByText("Paper weight", {exact: true}).first()).toBeVisible();
    await expect.poll(() => page.locator("svg circle").count()).toBeGreaterThan(0);
    await page.locator("svg circle").first().click({force: true});
  } else {
    await page.goto(`${origin}/inspect${query}`);
    await expect(page.getByText("Paper weight", {exact: true}).first()).toBeVisible();
    await page.getByRole("button", {name: /Lay the sheet flat and align its corners/}).click();
    await page.getByRole("button", {name: "Expand details"}).first().click();
    await expect(page.locator("video").first()).toBeVisible();
    await page.goto(`${origin}/space${query}`);
    const build = page.getByRole("button", {name: "Build", exact: true});
    await expect(build).toBeEnabled();
    await build.click();
    await expect(page.getByText(/\d+\/\d+ nodes/)).toBeVisible();
    await expect.poll(() => page.locator("svg circle").count()).toBeGreaterThan(0);
  }
  if (process.env.CIM_SMOKE_SCREENSHOT) await page.screenshot({path: process.env.CIM_SMOKE_SCREENSHOT, fullPage: true});
  assert.deepEqual(errors, [], "Browser runtime errors");
  assert.deepEqual(external, [], "Synthetic demo attempted external requests");
  assert.deepEqual(badResponses, [], "Failed local requests");
  console.log(`${kind}: production routes, example data, and interactions passed without external requests`);
} catch (error) {
  console.error(logs.slice(-8000));
  throw error;
} finally {
  if (browser) await browser.close();
  server.kill("SIGTERM");
}

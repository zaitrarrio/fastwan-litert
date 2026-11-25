// The browser fixture check: serves `web/` + `fixtures/tiny`, drives
// `tests/fixture.html` in headless Chromium through Playwright, prints the
// numbers and exits non-zero when they are off. Options:
//   --accelerator wasm|webgpu   (default wasm; WebGPU needs a GPU and
//                                `--enable-unsafe-webgpu`, not available headless here)
//   --jspi        load the JSPI Wasm build (`loadLiteRt(dir, {jspi: true})`)
//   --threads     load the threaded build (needs COOP/COEP, which serve.mjs sends)
//   --timeout S   give up after S seconds (default 300)
import { createRequire } from "node:module";
import { startServer } from "./serve.mjs";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const args = process.argv.slice(2);
const flag = (n) => args.includes(n);
const opt = (n, d) => (args.includes(n) ? args[args.indexOf(n) + 1] : d);
const accelerator = opt("--accelerator", "wasm");
const jspi = flag("--jspi");
const threads = flag("--threads");
const timeoutMs = Number(opt("--timeout", "300")) * 1000;

import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
if (!existsSync(join(here, "..", "dist", "src", "index.js"))) {
  console.error("web/dist is missing: run `npm run build` first (or `npm run test:fixture`, which builds)");
  process.exit(2);
}

const { server, port } = await startServer({ quiet: true });
const launch = { args: [] };
if (accelerator === "webgpu") launch.args.push("--enable-unsafe-webgpu", "--enable-features=Vulkan");
const browser = await chromium.launch(launch);
let code = 1;
try {
  const page = await browser.newPage();
  const logs = [];
  var pageLogs = logs;
  page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
  page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
  const q = new URLSearchParams({ manual: "1" });
  await page.goto(`http://127.0.0.1:${port}/tests/fixture.html?${q}`);
  await page.waitForFunction(() => typeof window.runFixture === "function", null, { timeout: 60_000 });
  const t0 = Date.now();
  const run = page.evaluate((o) => window.runFixture(o), {
    base: "/fixtures/tiny/",
    wasmDir: "/node_modules/@litertjs/core/wasm/",
    accelerator,
    jspi,
    threads,
  });
  const timer = new Promise((_, reject) => setTimeout(() => reject(new Error(`no result after ${timeoutMs / 1000} s`)), timeoutMs));
  const r = await Promise.race([run, timer]);
  const secs = ((Date.now() - t0) / 1000).toFixed(1);
  for (const e of r.events) console.log("  " + e);
  console.log(
    `fixture ${r.geometry} seed ${r.seed} (${r.tokens} tokens) on ${r.accelerator}${jspi ? "+jspi" : ""}${threads ? "+threads" : ""}: ` +
      `latent rel ${r.latent.rel.toExponential(2)} (max |Δ| ${r.latent.maxAbs.toExponential(2)}, tol ${r.latent.tol.toExponential(0)}), ` +
      `frames ${r.frames.count}/${r.frames.expected} max |Δ| ${r.frames.maxDelta}/255 (mean ${r.frames.meanDelta.toFixed(3)}), ` +
      `${r.downloads} graph downloads, ${secs} s: ${r.ok ? "OK" : "FAIL"}`,
  );
  // LiteRT prints its INFO lines through console.error; keep the real warnings only.
  const warn = logs.filter((l) => /^\[(warning|error|pageerror)\]/.test(l) && !/\] INFO:/.test(l) && !/WebGPU device|No available adapters/.test(l));
  if (warn.length) console.log(warn.join("\n"));
  code = r.ok ? 0 : 1;
} catch (e) {
  console.error("fixture check failed:", e);
  if (typeof pageLogs !== "undefined" && pageLogs.length) console.error("page log:\n" + pageLogs.join("\n"));
  code = 1;
} finally {
  await browser.close();
  server.close();
}
process.exit(code);

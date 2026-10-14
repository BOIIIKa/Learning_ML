// Shared Playwright bootstrap for the verify scripts. Resolves Playwright from the project,
// from $PLAYWRIGHT_MODULE, or from the cloud sandbox's preinstalled copy.
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const candidates = [process.env.PLAYWRIGHT_MODULE, "playwright", "/opt/node-tools/node_modules/playwright"].filter(Boolean);
let pw;
for (const c of candidates) {
  try { pw = require(c); break; } catch { /* try next */ }
}
if (!pw) throw new Error("Playwright not found: npm i -D playwright, or set PLAYWRIGHT_MODULE");

export const BASE = process.env.SITE_URL ?? "http://127.0.0.1:4321";
export const chromium = pw.chromium;

/** Collect console errors/warnings and page errors, ignoring React's devtools notice. */
export function watch(page) {
  const errors = [];
  page.on("console", (m) => {
    if ((m.type() === "error" || m.type() === "warning") && !m.text().includes("Download the React")) errors.push(`${m.type()}: ${m.text().slice(0, 240)}`);
  });
  page.on("pageerror", (e) => errors.push(`pageerror: ${String(e).slice(0, 300)}`));
  return errors;
}

export const arg = (i, fallback) => process.argv[2 + i] ?? fallback;

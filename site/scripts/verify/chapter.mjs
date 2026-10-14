// Notes page check: screenshot every widget, fill + run every code exercise in Pyodide.
//   node scripts/verify/chapter.mjs <slug> [lang=zh] [theme=dark] [outDir=./verify-out]
// Needs the dev server with a local Pyodide (see .claude/skills/verify-site/SKILL.md).
import { mkdirSync } from "node:fs";
import { BASE, arg, chromium, watch } from "./_browser.mjs";

const [slug, lang, theme, out] = [arg(0), arg(1, "zh"), arg(2, "dark"), arg(3, "verify-out")];
if (!slug) throw new Error("usage: chapter.mjs <slug> [lang] [theme] [outDir]");
mkdirSync(out, { recursive: true });
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, colorScheme: theme });
const errors = watch(page);
await page.goto(`${BASE}/${lang}/chapters/${slug}/`, { waitUntil: "networkidle" });
await page.evaluate((t) => document.documentElement.setAttribute("data-theme", t), theme);
let i = 0;
for (const w of await page.locator(".ch-article .widget").all()) {
  await w.scrollIntoViewIfNeeded(); await page.waitForTimeout(1500);
  await w.screenshot({ path: `${out}/${slug}-${lang}-${theme}-w${i++}.png` });
}
await page.locator("#s-practice").scrollIntoViewIfNeeded();
await page.waitForTimeout(1500);
const exs = await page.locator(".code-ex").all();
let failed = 0;
for (const [k, ex] of exs.entries()) {
  await ex.scrollIntoViewIfNeeded(); await page.waitForTimeout(2000);
  await ex.locator("button", { hasText: /填入全部|Fill all/ }).click();
  await page.waitForFunction((k) => document.querySelectorAll(".code-ex")[k].querySelector(".ex-left.done"), k, { timeout: 30000 });
  await ex.locator(".btn-primary").click();
  await page.waitForTimeout(300);
  await page.waitForFunction((k) => { const e = document.querySelectorAll(".code-ex")[k]; return e.querySelector(".ex-summary") && !e.querySelector(".spinner"); }, k, { timeout: 240000 });
  await page.waitForTimeout(2000);
  const fails = await ex.locator(".ex-tests li.fail").allTextContents();
  failed += fails.length;
  console.log(`exercise ${k}:`, await ex.locator(".ex-summary").textContent(), fails.join(" ; "));
  await ex.screenshot({ path: `${out}/${slug}-${lang}-${theme}-ex${k}.png` });
}
console.log(`widgets: ${i}  code exercises: ${exs.length}  failed tests: ${failed}`);
console.log("console:", errors.length ? errors.join("\n") : "clean");
await browser.close();
process.exit(failed || errors.some((e) => !e.startsWith("warning")) ? 1 : 0);

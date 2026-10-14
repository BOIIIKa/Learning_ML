// Visual sweep of the non-chapter pages in both themes, plus a hydration check of a chapter.
//   node scripts/verify/pages.mjs [slug-for-hydration=learning-triad] [outDir]
import { mkdirSync } from "node:fs";
import { BASE, arg, chromium, watch } from "./_browser.mjs";

const slug = arg(0, "learning-triad");
const out = arg(1, "verify-out");
mkdirSync(out, { recursive: true });
const browser = await chromium.launch();
let bad = 0;
for (const theme of ["dark", "light"]) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 860 }, colorScheme: theme });
  const errors = watch(page);
  for (const [name, path] of [["home", "zh/"], ["handbook", "zh/handbook/"], ["chapter", `zh/chapters/${slug}/`], ["en-chapter", `en/chapters/${slug}/`]]) {
    await page.goto(`${BASE}/${path}`, { waitUntil: "networkidle" });
    await page.evaluate((t) => document.documentElement.setAttribute("data-theme", t), theme);
    // Scroll through so every island hydrates (hydration mismatches show up as console errors).
    for (let y = 0; y < 30000; y += 700) { await page.evaluate((yy) => scrollTo(0, yy), y); await page.waitForTimeout(120); }
    await page.evaluate(() => scrollTo(0, 0)); await page.waitForTimeout(500);
    await page.screenshot({ path: `${out}/${name}-${theme}.png` });
  }
  const m = await browser.newPage({ viewport: { width: 390, height: 844 }, colorScheme: theme });
  await m.goto(`${BASE}/zh/chapters/${slug}/`, { waitUntil: "networkidle" });
  const overflow = await m.evaluate(() => document.documentElement.scrollWidth - innerWidth);
  console.log(`${theme}: mobile overflow ${overflow}px;`, errors.length ? errors.join(" | ") : "console clean");
  bad += errors.length + (overflow > 2 ? 1 : 0);
  await page.close(); await m.close();
}
await browser.close();
process.exit(bad ? 1 : 0);

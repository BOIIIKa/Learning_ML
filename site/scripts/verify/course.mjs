// Animated course check.
//   node scripts/verify/course.mjs walk [lang=zh] [slug...]       step through every beat of the decks
//   node scripts/verify/course.mjs film <slug> [lang] [#n] [t1,t2,...] [outDir]   play and screenshot at times
import { mkdirSync } from "node:fs";
import { BASE, arg, chromium, watch } from "./_browser.mjs";

const mode = arg(0, "walk");
const browser = await chromium.launch();
let bad = 0;

if (mode === "walk") {
  const lang = arg(1, "zh");
  const slugs = process.argv.slice(4);
  for (const slug of slugs) {
    const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
    const errors = watch(page);
    await page.goto(`${BASE}/${lang}/course/${slug}/`, { waitUntil: "networkidle" });
    await page.click("[data-act=start]");
    await page.waitForTimeout(300);
    await page.keyboard.press(" ");
    let beats = 0;
    for (; beats < 400; beats++) {
      await page.keyboard.press("ArrowRight");
      await page.waitForTimeout(200);
      const [cur, total] = (await page.textContent(".stage-counter")).split("/").map((s) => s.trim());
      if (cur === total && (await page.locator(".slide.active .step:not(.shown)").count()) === 0) break;
    }
    await page.waitForTimeout(1200);
    console.log(`${lang} ${slug}: ${beats + 1} beats`, errors.length ? errors.join(" | ") : "clean");
    bad += errors.length;
    await page.close();
  }
} else {
  const slug = arg(1);
  const lang = arg(2, "zh");
  const hash = arg(3, "");
  const times = arg(4, "1,3,6,10,15").split(",").map(Number);
  const out = arg(5, "verify-out");
  mkdirSync(out, { recursive: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const errors = watch(page);
  await page.goto(`${BASE}/${lang}/course/${slug}/${hash}`, { waitUntil: "networkidle" });
  await page.screenshot({ path: `${out}/${slug}-start.png` });
  await page.click("[data-act=start]");
  const t0 = Date.now();
  for (const t of times) {
    const wait = t * 1000 - (Date.now() - t0);
    if (wait > 0) await page.waitForTimeout(wait);
    console.log(`t=${t}s`, await page.textContent(".stage-counter"), "|", (await page.textContent(".subtitle-text")).slice(0, 60));
    await page.screenshot({ path: `${out}/${slug}-t${String(t).replace(".", "_")}.png` });
  }
  console.log("console:", errors.length ? errors.join("\n") : "clean");
  bad += errors.length;
}
await browser.close();
process.exit(bad ? 1 : 0);

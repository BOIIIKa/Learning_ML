// Hydration debugger: open a page, scroll so every island hydrates, and print React's
// hydration-mismatch diffs (the "+ client / - server" lines) together with the island they occur in.
//   node scripts/verify/hydration.mjs /zh/chapters/optimization/ [dark]
// Most mismatches in this project came from unrounded floats in SVG, Math.random() in render,
// multi-value `style` in SSR, or Math.cos / Math.sin differing in the last bit between Node and the
// browser (see `npm run verify:mathdiff`). Exits non-zero when any mismatch is found.
import { BASE, arg, chromium } from "./_browser.mjs";

const path = arg(0, "/zh/");
const theme = arg(1, "dark");
const browser = await chromium.launch();
const page = await browser.newPage({ colorScheme: theme, viewport: { width: 1280, height: 900 } });
let found = 0;
const report = (text) => {
  if (!/hydrat/i.test(text)) return;
  found++;
  const lines = text.split("\n");
  const keep = lines.filter((l) => /^\s*[+-] /.test(l) || /^\s*<[A-Z]\w*/.test(l));
  console.log(`--- mismatch ${found} ---\n${lines[0].slice(0, 200)}\n${keep.slice(0, 40).join("\n")}`);
};
page.on("console", (m) => { if (m.type() === "error") report(m.text()); });
page.on("pageerror", (e) => report(String(e)));
await page.goto(`${BASE}${path}`, { waitUntil: "networkidle" });
for (let i = 0; i < 60; i++) { await page.mouse.wheel(0, 400); await page.waitForTimeout(150); }
await page.waitForTimeout(2000);
await browser.close();
console.log(found ? `${found} hydration mismatch(es) on ${path}` : `no hydration mismatches on ${path}`);
process.exit(found ? 1 : 0);

// Compare Math.* between this Node and the Playwright browser, bit for bit, on 20 000 inputs.
//   node scripts/verify/mathdiff.mjs
// Widgets render on the server (Node) and hydrate in the browser; a function that differs in the
// last bit can change a chaotic computation (an unstable training run) and break hydration.
// In Node 22 / Chromium 141, cos, sin and pow (x ** y with a fractional y) differ, while tanh, exp, log,
// log1p, sqrt and atan2 agree:
// round data with r6() and use detCos (engines/optimization.ts) where it matters.
import { chromium } from "./_browser.mjs";

const fns = ["tanh", "exp", "log", "log1p", "sqrt", "cos", "sin", "atan2", "pow"];
const sample = (names) => {
  const out = {};
  for (const f of names) {
    const a = [];
    for (let i = 0; i < 20000; i++) {
      const x = (i - 10000) / 997.3;
      const pos = Math.abs(x) + 1e-3;
      a.push(f === "atan2" ? Math.atan2(x, 1.3) : f === "pow" ? Math.pow(pos, 1.7) : Math[f](["log", "sqrt"].includes(f) ? pos : f === "log1p" ? Math.abs(x) : x));
    }
    out[f] = a;
  }
  return out;
};
const node = sample(fns);
const browser = await chromium.launch();
const page = await browser.newPage();
const br = await page.evaluate(`(${sample.toString()})(${JSON.stringify(fns)})`);
console.log(`node ${process.version} vs chromium ${browser.version()}`);
for (const f of fns) {
  const d = node[f].filter((v, i) => v !== br[f][i]).length;
  console.log(`${f.padEnd(6)} ${d === 0 ? "identical" : `${d} / 20000 differ`}`);
}
await browser.close();

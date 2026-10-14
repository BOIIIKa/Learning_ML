// Chapter {{ID}} ({{TITLE_EN}}): pure, deterministic maths for the widgets. No DOM, no React.
// Rules: seeded randomness only (rng from ./data), round generated data with r6() so the server
// render and the client hydration see identical numbers.
import { gaussian, r6, rng } from "./data";

/** TODO(scaffold) Replace with the chapter's toy data. */
export function toyData(n = 40, seed = 1) {
  const rand = rng(seed);
  const xs = Array.from({ length: n }, () => r6(10 * rand()));
  return { xs, ys: xs.map((x) => r6(2 * x + 1 + gaussian(rand))) };
}

/** TODO(scaffold) The chapter's algorithm, small enough to run on every slider move. */
export function fit(xs: number[], ys: number[]): { slope: number; intercept: number } {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  const sxy = xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0);
  const sxx = xs.reduce((s, x) => s + (x - mx) ** 2, 0);
  const slope = sxy / sxx;
  return { slope, intercept: my - slope * mx };
}

// 1-D regression toolkit for chapters 01–03: features, losses, closed form, gradient descent, random search.
import { gaussian, r6, rng } from "./data";

export type Loss = "squared" | "absolute" | "huber";
export type Model = "constant" | "linear" | "cubic";

/** Study hours → exam score, with a few absent students scoring 0 (outliers). Scores in [0, 100]. */
export function studyData(n = 40, seed = 4, outliers = 4) {
  const rand = rng(seed);
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = r6(0.3 + 9.4 * rand());
    xs.push(x);
    ys.push(r6(Math.min(100, Math.max(0, 38 + 5.2 * x + 6 * gaussian(rand)))));
  }
  for (let k = 0; k < outliers; k++) ys[k] = r6(3 * rand());
  return { xs, ys };
}

/** Features scaled to roughly [−1, 1] so one learning rate works for every model. */
export const features = (model: Model, x: number): number[] => {
  const u = (x - 5) / 5;
  return model === "constant" ? [1] : model === "linear" ? [1, u] : [1, u, u * u, u * u * u];
};

export const predict = (model: Model, w: number[], x: number) => features(model, x).reduce((s, f, i) => s + f * w[i], 0);

const H = 10; // Huber threshold (score points)
export function lossValue(loss: Loss, r: number): number {
  if (loss === "squared") return r * r;
  if (loss === "absolute") return Math.abs(r);
  return Math.abs(r) <= H ? 0.5 * r * r : H * (Math.abs(r) - 0.5 * H);
}
/** d loss / d prediction, where r = prediction − y. */
export function lossGrad(loss: Loss, r: number): number {
  if (loss === "squared") return 2 * r;
  if (loss === "absolute") return Math.sign(r);
  return Math.abs(r) <= H ? r : H * Math.sign(r);
}

export const risk = (loss: Loss, model: Model, w: number[], xs: number[], ys: number[]) =>
  xs.reduce((s, x, i) => s + lossValue(loss, predict(model, w, x) - ys[i]), 0) / xs.length;

/** Solve A w = b by Gaussian elimination with partial pivoting (tiny systems only). */
export function solve(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const w = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) w[r] = (M[r][n] - M[r].slice(r + 1, n).reduce((s, v, k) => s + v * w[r + 1 + k], 0)) / M[r][r];
  return w;
}

/** Least squares via the normal equations ΦᵀΦ w = Φᵀy (+ ridge λ, intercept unpenalized). */
export function leastSquares(model: Model, xs: number[], ys: number[], lambda = 0): number[] {
  const Phi = xs.map((x) => features(model, x));
  const d = Phi[0].length;
  const A = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => Phi.reduce((s, f) => s + f[i] * f[j], 0) + (i === j && i > 0 ? lambda : 0)));
  const b = Array.from({ length: d }, (_, i) => Phi.reduce((s, f, n) => s + f[i] * ys[n], 0));
  return solve(A, b);
}

export interface Path { ws: number[][]; risks: number[] }

/** Full-batch (sub)gradient descent; returns every iterate. */
export function gradientDescent(loss: Loss, model: Model, xs: number[], ys: number[], steps = 120, lr?: number): Path {
  const d = features(model, 0).length;
  let w = new Array(d).fill(0);
  const eta = lr ?? (loss === "squared" ? 0.12 : loss === "huber" ? 0.25 : 6);
  const ws = [w];
  const risks = [risk(loss, model, w, xs, ys)];
  for (let t = 0; t < steps; t++) {
    const g = new Array(d).fill(0);
    xs.forEach((x, i) => {
      const f = features(model, x);
      const dr = lossGrad(loss, predict(model, w, x) - ys[i]);
      f.forEach((v, k) => (g[k] += (dr * v) / xs.length));
    });
    // Subgradient methods need a decaying step to settle.
    const step = loss === "absolute" ? eta / Math.sqrt(t + 1) : eta;
    w = w.map((v, k) => v - step * g[k]);
    ws.push(w);
    risks.push(risk(loss, model, w, xs, ys));
  }
  return { ws, risks };
}

/** Random search: propose a Gaussian perturbation of the best point, keep it if better. */
export function randomSearch(loss: Loss, model: Model, xs: number[], ys: number[], steps = 120, seed = 1): Path {
  const rand = rng(seed);
  const d = features(model, 0).length;
  let w = new Array(d).fill(0);
  let best = risk(loss, model, w, xs, ys);
  const ws = [w];
  const risks = [best];
  for (let t = 0; t < steps; t++) {
    const scale = 25 * Math.pow(0.97, t);
    const cand = w.map((v) => v + scale * gaussian(rand));
    const r = risk(loss, model, cand, xs, ys);
    if (r < best) { w = cand; best = r; }
    ws.push(w);
    risks.push(best);
  }
  return { ws, risks };
}

// ---------- risk of a constant predictor under absolute loss, Y ~ Exp(1) ----------

/** Expected risk E|Y − c| for Y ~ Exp(1): c − 1 + 2e^{−c} (c ≥ 0); minimized at the median ln 2. */
export const expAbsRisk = (c: number) => (c >= 0 ? c - 1 + 2 * Math.exp(-c) : 1 - c);

export function expSample(n: number, seed: number): number[] {
  const rand = rng(seed);
  return Array.from({ length: n }, () => r6(-Math.log(1 - rand())));
}

export const empiricalAbsRisk = (ys: number[], c: number) => ys.reduce((s, y) => s + Math.abs(y - c), 0) / ys.length;

export function median(v: number[]): number {
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

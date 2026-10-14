// Chapter 03: least squares as projection, ridge and lasso (closed form, coordinate descent), paths.
import { gaussian, r6, rng } from "./data";
import { solve } from "./regression";

export type Vec3 = [number, number, number];
export const dot3 = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** Orthogonal projection of y onto span(a1, a2) in R³: returns the coefficients and ŷ. */
export function project3(a1: Vec3, a2: Vec3, y: Vec3): { coef: [number, number]; yhat: Vec3 } {
  const G = [[dot3(a1, a1), dot3(a1, a2)], [dot3(a2, a1), dot3(a2, a2)]];
  const c = solve(G, [dot3(a1, y), dot3(a2, y)]) as [number, number];
  const yhat = [0, 1, 2].map((i) => c[0] * a1[i] + c[1] * a2[i]) as Vec3;
  return { coef: c, yhat };
}

/** Soft thresholding S(z, g) = sign(z)·max(|z| − g, 0), the proximal map of g|·|. */
export const softThreshold = (z: number, g: number) => Math.sign(z) * Math.max(Math.abs(z) - g, 0);

const gram = (X: number[][]) => {
  const p = X[0].length;
  return Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => X.reduce((s, r) => s + r[i] * r[j], 0)));
};
const xty = (X: number[][], y: number[]) => X[0].map((_, j) => X.reduce((s, r, i) => s + r[j] * y[i], 0));

/** Ridge for (1/2n)‖y − Xw‖² + (λ/2)‖w‖² (no intercept: centre y and standardize X first). */
export function ridge(X: number[][], y: number[], lambda: number): number[] {
  const n = X.length;
  const A = gram(X).map((row, i) => row.map((v, j) => v / n + (i === j ? lambda : 0)));
  return solve(A, xty(X, y).map((v) => v / n));
}

/** Lasso for (1/2n)‖y − Xw‖² + λ‖w‖₁ by cyclic coordinate descent with warm start. */
export function lassoCD(X: number[][], y: number[], lambda: number, w0?: number[], sweeps = 300): number[] {
  const n = X.length;
  const p = X[0].length;
  const w = w0 ? [...w0] : new Array(p).fill(0);
  const r = y.map((yi, i) => yi - X[i].reduce((s, v, j) => s + v * w[j], 0));
  const sq = X[0].map((_, j) => X.reduce((s, row) => s + row[j] ** 2, 0) / n);
  for (let it = 0; it < sweeps; it++) {
    let delta = 0;
    for (let j = 0; j < p; j++) {
      if (sq[j] === 0) continue;
      // Partial residual correlation: (1/n) Σ x_ij (r_i + x_ij w_j)
      let rho = 0;
      for (let i = 0; i < n; i++) rho += X[i][j] * (r[i] + X[i][j] * w[j]);
      rho /= n;
      const next = softThreshold(rho, lambda) / sq[j];
      const d = next - w[j];
      if (d !== 0) {
        for (let i = 0; i < n; i++) r[i] -= X[i][j] * d;
        w[j] = next;
        delta = Math.max(delta, Math.abs(d));
      }
    }
    if (delta < 1e-9) break;
  }
  return w;
}

/** Column-standardize X; returns Z plus the means and standard deviations (population). */
export function standardize(X: number[][]) {
  const n = X.length;
  const p = X[0].length;
  const mu = Array.from({ length: p }, (_, j) => X.reduce((s, r) => s + r[j], 0) / n);
  const sd = Array.from({ length: p }, (_, j) => Math.sqrt(X.reduce((s, r) => s + (r[j] - mu[j]) ** 2, 0) / n) || 1);
  return { Z: X.map((r) => r.map((v, j) => (v - mu[j]) / sd[j])), mu, sd };
}

/** Raw polynomial features u, u², …, u^d of u = (x − 5)/5 (no constant column). */
export const polyRow = (x: number, d: number) => Array.from({ length: d }, (_, k) => ((x - 5) / 5) ** (k + 1));

export const logGrid = (lo: number, hi: number, n: number) =>
  Array.from({ length: n }, (_, i) => 10 ** (Math.log10(hi) + ((Math.log10(lo) - Math.log10(hi)) * i) / (n - 1)));

export type Penalty = "ridge" | "lasso";

/** A fitted polynomial model: standardized features, centred y. */
export function fitPoly(xs: number[], ys: number[], d: number, pen: Penalty, lambda: number, w0?: number[]) {
  const { Z, mu, sd } = standardize(xs.map((x) => polyRow(x, d)));
  const ybar = ys.reduce((a, b) => a + b, 0) / ys.length;
  const yc = ys.map((v) => v - ybar);
  const w = pen === "ridge" ? ridge(Z, yc, lambda) : lassoCD(Z, yc, lambda, w0);
  const f = (x: number) => ybar + polyRow(x, d).reduce((s, v, j) => s + ((v - mu[j]) / sd[j]) * w[j], 0);
  return { w, f };
}

/** Coefficient path over λ (largest first, warm-started) with 5-fold CV error at each λ. */
export function regPath(xs: number[], ys: number[], d: number, pen: Penalty, lambdas: number[], k = 5) {
  const n = xs.length;
  const fold = xs.map((_, i) => i % k);
  const ws: number[][] = [];
  const fs: ((x: number) => number)[] = [];
  const cv: number[] = [];
  let warm: number[] | undefined;
  const warmFold: (number[] | undefined)[] = new Array(k).fill(undefined);
  for (const lam of lambdas) {
    const m = fitPoly(xs, ys, d, pen, lam, warm);
    warm = m.w;
    ws.push(m.w);
    fs.push(m.f);
    let err = 0;
    for (let j = 0; j < k; j++) {
      const tr = xs.flatMap((_, i) => (fold[i] !== j ? [i] : []));
      const te = xs.flatMap((_, i) => (fold[i] === j ? [i] : []));
      const mj = fitPoly(tr.map((i) => xs[i]), tr.map((i) => ys[i]), d, pen, lam, warmFold[j]);
      warmFold[j] = mj.w;
      err += te.reduce((s, i) => s + (mj.f(xs[i]) - ys[i]) ** 2, 0);
    }
    cv.push(err / n);
  }
  return { ws, fs, cv };
}

/** Two correlated features (correlation rho), y = x₁ + x₂ + noise. Returns OLS and ridge estimates over many resamples. */
export function collinearDraws(rho: number, lambda: number, reps = 60, n = 25, seed = 3) {
  const rand = rng(seed);
  const ols: [number, number][] = [];
  const rdg: [number, number][] = [];
  for (let r = 0; r < reps; r++) {
    const X: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < n; i++) {
      const a = gaussian(rand);
      const b = rho * a + Math.sqrt(1 - rho * rho) * gaussian(rand);
      X.push([a, b]);
      y.push(a + b + gaussian(rand));
    }
    const o = ridge(X, y, 0);
    const g = ridge(X, y, lambda);
    ols.push([r6(o[0]), r6(o[1])]);
    rdg.push([r6(g[0]), r6(g[1])]);
  }
  return { ols, ridge: rdg };
}

/** 2-D penalized quadratic: min (w − c)ᵀA(w − c) + pen(w). Ridge pen = λ‖w‖², lasso pen = λ‖w‖₁. */
export function quadPenalized(A: number[][], c: [number, number], pen: Penalty, lambda: number): [number, number] {
  if (pen === "ridge") {
    const Ac = [A[0][0] * c[0] + A[0][1] * c[1], A[1][0] * c[0] + A[1][1] * c[1]];
    return solve([[A[0][0] + lambda, A[0][1]], [A[1][0], A[1][1] + lambda]], Ac) as [number, number];
  }
  const w: [number, number] = [c[0], c[1]];
  for (let it = 0; it < 500; it++) {
    for (let j = 0; j < 2; j++) {
      const o = 1 - j;
      const z = A[j][j] * c[j] - A[j][o] * (w[o] - c[o]);
      w[j] = softThreshold(z, lambda / 2) / A[j][j];
    }
  }
  return w;
}

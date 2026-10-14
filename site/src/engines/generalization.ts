// Chapter 02: polynomial fits (Legendre basis), bias–variance by resampling, k-fold CV, double descent.
import { gaussian, r6, rng } from "./data";
import { solve } from "./regression";

/** The "true" study-hours curve: scores saturate as hours grow. */
export const truth = (x: number) => 30 + 60 / (1 + Math.exp(-(x - 4.5) / 1.3));
export const NOISE = 7;

export function sample(n: number, seed: number) {
  const rand = rng(seed);
  const xs = Array.from({ length: n }, () => r6(0.2 + 9.6 * rand()));
  return { xs, ys: xs.map((x) => r6(truth(x) + NOISE * gaussian(rand))) };
}

/** Legendre polynomials P_0..P_d at u ∈ [−1, 1] (well conditioned, unlike raw powers). */
export function legendre(d: number, u: number): number[] {
  const p = [1, u];
  for (let k = 1; k < d; k++) p.push(((2 * k + 1) * u * p[k] - k * p[k - 1]) / (k + 1));
  return p.slice(0, d + 1);
}
const toU = (x: number) => (x - 5) / 5;

/** Least-squares polynomial of degree d (tiny ridge for numerical safety). Returns the fitted function. */
export function polyFit(xs: number[], ys: number[], d: number, ridge = 1e-8): (x: number) => number {
  const Phi = xs.map((x) => legendre(d, toU(x)));
  const k = d + 1;
  const A = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => Phi.reduce((s, f) => s + f[i] * f[j], 0) + (i === j ? ridge : 0)));
  const b = Array.from({ length: k }, (_, i) => Phi.reduce((s, f, n) => s + f[i] * ys[n], 0));
  const w = solve(A, b);
  return (x: number) => legendre(d, toU(x)).reduce((s, v, i) => s + v * w[i], 0);
}

export const mse = (f: (x: number) => number, xs: number[], ys: number[]) => xs.reduce((s, x, i) => s + (f(x) - ys[i]) ** 2, 0) / xs.length;

/** Monte-Carlo bias², variance and noise of degree-d fits over many training sets, averaged over a test grid. */
export function biasVariance(d: number, n: number, reps = 40, seed = 100) {
  const grid = Array.from({ length: 41 }, (_, i) => 0.2 + (9.6 * i) / 40);
  const fits = Array.from({ length: reps }, (_, r) => { const s = sample(n, seed + r); return polyFit(s.xs, s.ys, d); });
  let bias2 = 0;
  let variance = 0;
  for (const x of grid) {
    const preds = fits.map((f) => f(x));
    const mean = preds.reduce((a, b) => a + b, 0) / reps;
    const v = preds.reduce((a, p) => a + (p - mean) ** 2, 0) / reps;
    // The Monte-Carlo mean is itself noisy: E[(mean − f)²] = bias² + Var/reps. Subtract the
    // unbiased estimate of that term so wild high-degree fits don't masquerade as bias.
    bias2 += (mean - truth(x)) ** 2 - v / (reps - 1);
    variance += v;
  }
  return { fits, bias2: Math.max(bias2 / grid.length, 0.1), variance: variance / grid.length, noise: NOISE ** 2 };
}

/** k-fold assignment: index i goes to fold i mod k after a seeded shuffle. */
export function folds(n: number, k: number, seed = 5): number[] {
  const rand = rng(seed);
  const order = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  const fold = new Array(n);
  order.forEach((idx, pos) => (fold[idx] = pos % k));
  return fold;
}

export function cvError(xs: number[], ys: number[], d: number, fold: number[], k: number): number {
  let total = 0;
  for (let f = 0; f < k; f++) {
    const tr = xs.map((_, i) => i).filter((i) => fold[i] !== f);
    const te = xs.map((_, i) => i).filter((i) => fold[i] === f);
    if (!te.length || tr.length <= d) return NaN;
    const fit = polyFit(tr.map((i) => xs[i]), tr.map((i) => ys[i]), d);
    total += te.reduce((s, i) => s + (fit(xs[i]) - ys[i]) ** 2, 0);
  }
  return total / xs.length;
}

// ---------- double descent: minimum-norm least squares on isotropic Gaussian features ----------

/**
 * y = xᵀβ + ε with x ~ N(0, I_P). The model sees only the first p features and fits them by
 * minimum-norm least squares. For isotropic x the test risk is exact:
 *   R = ‖β_{1:p} − ŵ‖² + ‖β_{p+1:P}‖² + σ².
 * Averaged over training draws, R peaks at the interpolation threshold p = n.
 */
export function doubleDescentCurve(n = 30, P = 100, draws = 12, sigma = 0.5, seed = 2) {
  const rand = rng(seed);
  const beta = Array.from({ length: P }, (_, j) => 1 / Math.sqrt(1 + j)); // signal spread over all features
  const ps = Array.from({ length: P }, (_, i) => i + 1);
  const sums = new Array(P).fill(0);
  for (let d = 0; d < draws; d++) {
    const X = Array.from({ length: n }, () => Array.from({ length: P }, () => gaussian(rand)));
    const y = X.map((row) => row.reduce((s, v, j) => s + v * beta[j], 0) + sigma * gaussian(rand));
    for (const p of ps) {
      const Xp = X.map((row) => row.slice(0, p));
      const w = minNorm(Xp, y);
      let r = sigma * sigma;
      for (let j = 0; j < P; j++) r += (beta[j] - (j < p ? w[j] : 0)) ** 2;
      sums[p - 1] += r;
    }
  }
  return { ps, errs: sums.map((s) => s / draws), n };
}

/** Minimum-norm least-squares solution of X w ≈ y. */
export function minNorm(X: number[][], y: number[]): number[] {
  const n = X.length;
  const p = X[0].length;
  const tiny = 1e-10;
  if (p <= n) {
    const A = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => X.reduce((s, r) => s + r[i] * r[j], 0) + (i === j ? tiny : 0)));
    return solve(A, Array.from({ length: p }, (_, i) => X.reduce((s, r, k) => s + r[i] * y[k], 0)));
  }
  const G = X.map((a, i) => X.map((b, j) => a.reduce((s, v, k) => s + v * b[k], 0) + (i === j ? tiny : 0)));
  const c = solve(G, y);
  return Array.from({ length: p }, (_, k) => X.reduce((s, r, i) => s + c[i] * r[k], 0));
}

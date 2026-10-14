// Chapter 02 toolkit: polynomial fits, bias–variance simulation, k-fold CV and random-features double descent.
import { gaussian, r6, rng } from "./data";
import { solve } from "./regression";

/** The true mean score after h weekly study hours: a slow start, a jump, then saturation (30 → 80). */
export const truth = (h: number) => 30 + 50 / (1 + Math.exp(-(h - 4)));

export const X_LO = 0.5;
export const X_HI = 9.5;
/** Hours are mapped to u ∈ [−1, 1] before building features. */
export const toU = (h: number) => (h - 5) / 5;

/** Legendre polynomials P0(u) … Pd(u) by the three-term recurrence (well conditioned on [−1, 1]). */
export function legendre(u: number, d: number): number[] {
  const p = [1];
  if (d >= 1) p.push(u);
  for (let k = 1; k < d; k++) p.push(((2 * k + 1) * u * p[k] - k * p[k - 1]) / (k + 1));
  return p;
}

/**
 * Least squares (optionally ridge with weight λ, intercept unpenalized) on Legendre features of degree d.
 * With p = d + 1 > n there are many solutions; a tiny ridge picks (nearly) the minimum-norm one.
 */
export function polyFit(xs: number[], ys: number[], d: number, lambda = 0): number[] {
  const Phi = xs.map((x) => legendre(toU(x), d));
  const p = d + 1;
  const jitter = p > xs.length ? 1e-9 : 0;
  const A = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => {
    let s = 0;
    for (const f of Phi) s += f[i] * f[j];
    return s + (i === j ? (i > 0 ? lambda : 0) + jitter : 0);
  }));
  const b = Array.from({ length: p }, (_, i) => Phi.reduce((s, f, n) => s + f[i] * ys[n], 0));
  return solve(A, b);
}

export const polyPredict = (w: number[], x: number) => legendre(toU(x), w.length - 1).reduce((s, v, i) => s + v * w[i], 0);

export const mse = (pred: number[], ys: number[]) => pred.reduce((s, p, i) => s + (p - ys[i]) ** 2, 0) / ys.length;

/** Evenly spread hours: one student in each of n equal bins (jittered), or an exact grid from X_LO to X_HI. */
export function hoursDesign(n: number, rand?: () => number): number[] {
  if (!rand) return linspace(X_LO, X_HI, n).map(r6);
  return Array.from({ length: n }, (_, i) => r6(X_LO + ((X_HI - X_LO) * (i + rand())) / n));
}

/** One cohort of students: jittered hours, scores = truth + N(0, σ²). */
export function cohort(n: number, sigma: number, seed: number) {
  const rand = rng(seed);
  const xs = hoursDesign(n, rand);
  const ys = xs.map((x) => r6(truth(x) + sigma * gaussian(rand)));
  return { xs, ys };
}

/** Next year's students: hours uniform on [X_LO, X_HI]. */
export function testSet(n: number, sigma: number, seed: number) {
  const rand = rng(seed);
  const xs = Array.from({ length: n }, () => r6(X_LO + (X_HI - X_LO) * rand()));
  const ys = xs.map((x) => r6(truth(x) + sigma * gaussian(rand)));
  return { xs, ys };
}

export const linspace = (lo: number, hi: number, n: number) => Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1));

export interface BiasVariance {
  /** Predictions on `grid` for every simulated training set. */
  preds: number[][];
  mean: number[];
  bias2: number;
  variance: number;
  noise: number;
}

/**
 * Fixed design: the same n study-hour values every time, fresh noise for each of R training sets.
 * Bias² and variance are averaged over `grid` (uniform over the hours range).
 */
export function biasVariance(d: number, n: number, sigma: number, R: number, seed: number, grid: number[]): BiasVariance {
  const xs = hoursDesign(n);
  const rand = rng(seed);
  // r6 on every transcendental result keeps server and browser renders bit-identical.
  const fx = xs.map((x) => r6(truth(x)));
  const preds = Array.from({ length: R }, () => {
    const ys = fx.map((f) => r6(f + sigma * gaussian(rand)));
    const w = polyFit(xs, ys, d);
    return grid.map((g) => polyPredict(w, g));
  });
  const mean = grid.map((_, j) => preds.reduce((s, p) => s + p[j], 0) / R);
  const bias2 = grid.reduce((s, g, j) => s + (mean[j] - r6(truth(g))) ** 2, 0) / grid.length;
  const variance = grid.reduce((s, _, j) => s + preds.reduce((t, p) => t + (p[j] - mean[j]) ** 2, 0) / R, 0) / grid.length;
  return { preds, mean, bias2, variance, noise: sigma * sigma };
}

/** Shuffle 0..n−1 (Fisher–Yates) and cut into k nearly equal consecutive folds, like np.array_split. */
export function kfoldSplits(n: number, k: number, seed: number): number[][] {
  const rand = rng(seed);
  const perm = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [perm[i], perm[j]] = [perm[j], perm[i]];
  }
  const folds: number[][] = [];
  let start = 0;
  for (let f = 0; f < k; f++) {
    const size = Math.floor(n / k) + (f < n % k ? 1 : 0);
    folds.push(perm.slice(start, start + size).sort((a, b) => a - b));
    start += size;
  }
  return folds;
}

export interface CvResult { folds: number[][]; weights: number[][]; foldMse: number[]; estimate: number }

/** k-fold cross-validation of a degree-d polynomial; the estimate is the size-weighted mean fold MSE. */
export function crossValidate(xs: number[], ys: number[], d: number, k: number, seed: number): CvResult {
  const folds = kfoldSplits(xs.length, k, seed);
  const weights: number[][] = [];
  const foldMse: number[] = [];
  let total = 0;
  for (const val of folds) {
    const inVal = new Set(val);
    const tr = xs.map((_, i) => i).filter((i) => !inVal.has(i));
    const w = polyFit(tr.map((i) => xs[i]), tr.map((i) => ys[i]), d);
    const err = mse(val.map((i) => polyPredict(w, xs[i])), val.map((i) => ys[i]));
    weights.push(w);
    foldMse.push(err);
    total += err * val.length;
  }
  return { folds, weights, foldMse, estimate: total / xs.length };
}

/** Expected test MSE of a degree-d fit: squared distance to the truth on `grid`, plus the noise σ². */
export const testError = (w: number[], sigma: number, grid: number[]) =>
  grid.reduce((s, g) => s + (polyPredict(w, g) - truth(g)) ** 2, 0) / grid.length + sigma * sigma;

export interface CvSimulation { ks: number[]; estimates: number[][]; trueError: number }

/**
 * Draw R cohorts; for each, compute the k-fold CV estimate for every k in `ks` (same cohort, same shuffle seed),
 * and the true test error of the model trained on the whole cohort. Returns the CV estimates per k and the
 * average true error, so the estimator's bias and spread can be read off.
 */
export function cvSimulation(n: number, sigma: number, d: number, ks: number[], R: number, seed: number, grid = linspace(X_LO, X_HI, 80)): CvSimulation {
  const estimates = ks.map(() => [] as number[]);
  let trueSum = 0;
  for (let r = 0; r < R; r++) {
    const { xs, ys } = cohort(n, sigma, seed + 101 * r);
    ks.forEach((k, i) => estimates[i].push(crossValidate(xs, ys, d, k, seed + r).estimate));
    trueSum += testError(polyFit(xs, ys, d), sigma, grid);
  }
  return { ks, estimates, trueError: trueSum / R };
}

export const meanSd = (v: number[]) => {
  const m = v.reduce((a, b) => a + b, 0) / v.length;
  return { mean: m, sd: Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length) };
};

// ---------- random-features regression and double descent ----------

export interface Features { omega: number[]; phase: number[] }

/** p random Fourier features cos(ω u + b), ω ~ N(0, scale²), b ~ U[0, 2π). */
export function randomFeatures(p: number, seed: number, scale = 15): Features {
  const rand = rng(seed);
  const omega: number[] = [];
  const phase: number[] = [];
  for (let j = 0; j < p; j++) {
    omega.push(scale * gaussian(rand));
    phase.push(2 * Math.PI * rand());
  }
  return { omega, phase };
}

export const featureRow = (F: Features, x: number) => {
  const u = toU(x);
  const c = Math.sqrt(2 / F.omega.length);
  return F.omega.map((w, j) => c * Math.cos(w * u + F.phase[j]));
};

export interface RfModel { F: Features; w: number[]; offset: number }

/**
 * Least squares on random features, centring y first. λ = 0 gives the ordinary solution when p ≤ n and the
 * minimum-norm interpolant when p > n (solved in the n×n dual, wᵀ = Φᵀ(ΦΦᵀ + λI)⁻¹y). λ > 0 is ridge.
 */
export function rfFit(F: Features, xs: number[], ys: number[], lambda = 0): RfModel {
  const n = xs.length;
  const p = F.omega.length;
  const offset = ys.reduce((a, b) => a + b, 0) / n;
  const yc = ys.map((y) => y - offset);
  const Phi = xs.map((x) => featureRow(F, x));
  // A 1e-12 jitter keeps the exactly-square case p = n solvable without visibly regularizing it.
  const tiny = 1e-12;
  if (p <= n) {
    const A = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => Phi.reduce((s, f) => s + f[i] * f[j], 0) + (i === j ? lambda + tiny : 0)));
    const b = Array.from({ length: p }, (_, i) => Phi.reduce((s, f, k) => s + f[i] * yc[k], 0));
    return { F, w: solve(A, b), offset };
  }
  const K = Phi.map((a, i) => Phi.map((b, j) => a.reduce((s, v, k) => s + v * b[k], 0) + (i === j ? lambda + tiny : 0)));
  const alpha = solve(K, yc);
  const w = Array.from({ length: p }, (_, j) => Phi.reduce((s, f, i) => s + f[j] * alpha[i], 0));
  return { F, w, offset };
}

export const rfPredict = (m: RfModel, x: number) => m.offset + featureRow(m.F, x).reduce((s, v, j) => s + v * m.w[j], 0);

/** Feature counts swept by the double-descent widget (log-spaced, with every p near n = 20). */
export const DD_PS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 28, 32, 36, 40, 50, 60, 80, 100, 130, 160, 200, 300, 400, 600, 1000];

export interface DdPoint { p: number; test: number; train: number; norm: number }

/** Median (over `draws` feature draws) test MSE against the noiseless truth + σ², train MSE and ‖w‖ for each p. */
export function doubleDescent(xs: number[], ys: number[], sigma: number, lambda: number, ps = DD_PS, draws = 9, grid = linspace(X_LO, X_HI, 120)): DdPoint[] {
  const med = (v: number[]) => [...v].sort((a, b) => a - b)[v.length >> 1];
  const target = grid.map(truth);
  return ps.map((p) => {
    const test: number[] = [];
    const train: number[] = [];
    const norm: number[] = [];
    for (let s = 0; s < draws; s++) {
      const m = rfFit(randomFeatures(p, 7919 * p + 104729 * s + 17), xs, ys, lambda);
      test.push(mse(grid.map((g) => rfPredict(m, g)), target) + sigma * sigma);
      train.push(mse(xs.map((x) => rfPredict(m, x)), ys));
      norm.push(Math.sqrt(m.w.reduce((a, b) => a + b * b, 0)));
    }
    return { p, test: med(test), train: med(train), norm: med(norm) };
  });
}

// ---------- learning theory ----------

/** Hoeffding + union bound: with probability ≥ 1 − δ every f in a finite class has |R − R̂| ≤ ε. */
export const hoeffdingEps = (H: number, n: number, delta: number) => Math.sqrt(Math.log((2 * H) / delta) / (2 * n));

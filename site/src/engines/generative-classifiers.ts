// Chapter 05 (generative classifiers): Gaussian discriminant analysis (LDA / QDA), Gaussian naive
// Bayes versus logistic regression, and Laplace-smoothed naive Bayes. Pure, deterministic maths.
import { gaussian, r6, rng } from "./data";
import { newton, type V2 } from "./linear-classification";
import { solve } from "./regression";

export type M2 = [[number, number], [number, number]];

export const inv2 = (S: M2): M2 => {
  const d = S[0][0] * S[1][1] - S[0][1] * S[1][0];
  return [[S[1][1] / d, -S[0][1] / d], [-S[1][0] / d, S[0][0] / d]];
};
export const det2 = (S: M2) => S[0][0] * S[1][1] - S[0][1] * S[1][0];
const quad = (S: M2, v: V2) => v[0] * (S[0][0] * v[0] + S[0][1] * v[1]) + v[1] * (S[1][0] * v[0] + S[1][1] * v[1]);

/** Covariance with standard deviations (a, b) along axes rotated by `deg`. */
export function covFrom(a: number, b: number, deg: number): M2 {
  const t = (deg * Math.PI) / 180;
  const c = Math.cos(t), s = Math.sin(t);
  return [
    [a * a * c * c + b * b * s * s, (a * a - b * b) * c * s],
    [(a * a - b * b) * c * s, a * a * s * s + b * b * c * c],
  ];
}

/** Lower Cholesky factor of a 2×2 SPD matrix. */
const chol = (S: M2): M2 => {
  const l00 = Math.sqrt(S[0][0]);
  const l10 = S[1][0] / l00;
  return [[l00, 0], [l10, Math.sqrt(Math.max(1e-12, S[1][1] - l10 * l10))]];
};

/** Fixed standard-normal draws, so the sample moves smoothly when the parameters change. */
export function stdNormals(n: number, seed: number): V2[] {
  const rand = rng(seed);
  return Array.from({ length: n }, () => [gaussian(rand), gaussian(rand)] as V2);
}

export interface Gauss { mean: V2; cov: M2 }

/** Transform fixed normals into a sample: n₊ points from class +1 and n₋ from class −1. */
export function sampleClasses(pos: Gauss, neg: Gauss, zPos: V2[], zNeg: V2[]): { X: V2[]; y: number[] } {
  const tr = (g: Gauss, z: V2): V2 => {
    const L = chol(g.cov);
    return [r6(g.mean[0] + L[0][0] * z[0]), r6(g.mean[1] + L[1][0] * z[0] + L[1][1] * z[1])];
  };
  return { X: [...zPos.map((z) => tr(pos, z)), ...zNeg.map((z) => tr(neg, z))], y: [...zPos.map(() => 1), ...zNeg.map(() => -1)] };
}

/* ------------------------------------------------------------------ Gaussian discriminant analysis */

export interface GdaFit {
  prior: [number, number]; // [P(+), P(−)]
  mean: [V2, V2];
  cov: [M2, M2];
  pooled: M2;
}

/** Maximum-likelihood estimates: class frequencies, class means, class covariances and their pooled average. */
export function fitGda(X: V2[], y: number[]): GdaFit {
  const parts = [1, -1].map((c) => X.filter((_, i) => y[i] === c));
  const mean = parts.map((P) => [P.reduce((s, p) => s + p[0], 0) / P.length, P.reduce((s, p) => s + p[1], 0) / P.length] as V2) as [V2, V2];
  const cov = parts.map((P, k) => {
    const m = mean[k];
    const S: M2 = [[0, 0], [0, 0]];
    P.forEach((p) => {
      const d = [p[0] - m[0], p[1] - m[1]];
      for (let r = 0; r < 2; r++) for (let c = 0; c < 2; c++) S[r][c] += (d[r] * d[c]) / P.length;
    });
    return S;
  }) as [M2, M2];
  const n = X.length;
  const w = parts.map((P) => P.length / n);
  const pooled: M2 = [0, 1].map((r) => [0, 1].map((c) => w[0] * cov[0][r][c] + w[1] * cov[1][r][c])) as M2;
  return { prior: [w[0], w[1]], mean, cov, pooled };
}

/** log N(x; m, S). */
export function logGauss(x: V2, m: V2, S: M2): number {
  const d: V2 = [x[0] - m[0], x[1] - m[1]];
  return -0.5 * quad(inv2(S), d) - 0.5 * Math.log(det2(S)) - Math.log(2 * Math.PI);
}

/** QDA log-odds log P(+|x) − log P(−|x) with each class's own covariance. */
export const qdaLogOdds = (f: GdaFit, x: V2) =>
  Math.log(f.prior[0] / f.prior[1]) + logGauss(x, f.mean[0], f.cov[0]) - logGauss(x, f.mean[1], f.cov[1]);

/** LDA: with the shared (pooled) covariance the log-odds are linear, w·x + b. */
export function ldaLine(f: GdaFit): { w: V2; b: number } {
  const Si = inv2(f.pooled);
  const dm: V2 = [f.mean[0][0] - f.mean[1][0], f.mean[0][1] - f.mean[1][1]];
  const w: V2 = [Si[0][0] * dm[0] + Si[0][1] * dm[1], Si[1][0] * dm[0] + Si[1][1] * dm[1]];
  const b = -0.5 * (quad(Si, f.mean[0]) - quad(Si, f.mean[1])) + Math.log(f.prior[0] / f.prior[1]);
  return { w, b };
}

/** LDA log-odds through the general Gaussian formula (equals w·x + b; used in tests). */
export const ldaLogOdds = (f: GdaFit, x: V2) =>
  Math.log(f.prior[0] / f.prior[1]) + logGauss(x, f.mean[0], f.pooled) - logGauss(x, f.mean[1], f.pooled);

/** Logistic regression on the same data (Newton from Chapter 04). */
export function logisticLine(X: V2[], y: number[]): { w: V2; b: number } {
  const t = newton(X, y, 30).at(-1)!;
  return { w: [t[0], t[1]], b: t[2] };
}

export const errorRate = (score: (x: V2) => number, X: V2[], y: number[]) =>
  X.reduce((s, x, i) => s + (y[i] * score(x) > 0 ? 0 : 1), 0) / X.length;

/* ------------------------------------------------------------------ naive Bayes vs logistic in p dimensions */

/**
 * Two classes with `k` independent informative features (means ±δ/2, unit variance) plus `dup`
 * noisy copies of the first one. The copies break naive Bayes' independence assumption: it
 * counts the same evidence 1 + dup times. Logistic regression simply learns small weights for them.
 */
export function dupClasses(n: number, k: number, dup: number, delta: number, seed: number): { X: number[][]; y: number[] } {
  const rand = rng(seed);
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const c = i % 2 === 0 ? 1 : -1;
    const base = Array.from({ length: k }, () => (c * delta) / 2 + gaussian(rand));
    X.push([...base, ...Array.from({ length: dup }, () => base[0] + 0.3 * gaussian(rand))]);
    y.push(c);
  }
  return { X, y };
}

/**
 * Gaussian naive Bayes: per-class means, one variance per feature pooled over the classes (stable
 * with very few samples); returns the log-odds function.
 */
export function gnb(X: number[][], y: number[]): (x: number[]) => number {
  const p = X[0].length;
  const stats = [1, -1].map((c) => {
    const P = X.filter((_, i) => y[i] === c);
    return { P, m: Array.from({ length: p }, (_, j) => P.reduce((s, x) => s + x[j], 0) / P.length), prior: P.length / X.length };
  });
  const v = Array.from({ length: p }, (_, j) =>
    Math.max(1e-3, stats.reduce((s, st) => s + st.P.reduce((t, x) => t + (x[j] - st.m[j]) ** 2, 0), 0) / X.length));
  return (x) => {
    let s = Math.log(stats[0].prior / stats[1].prior);
    for (let j = 0; j < p; j++) s += ((x[j] - stats[1].m[j]) ** 2 - (x[j] - stats[0].m[j]) ** 2) / (2 * v[j]);
    return s;
  };
}

/** L2-regularized logistic regression in p dimensions (Newton); returns the log-odds function. */
export function logisticP(X: number[][], y: number[], lambda = 1e-2, iters = 12): (x: number[]) => number {
  const p = X[0].length + 1;
  const A = X.map((x) => [...x, 1]);
  let w = new Array(p).fill(0);
  for (let t = 0; t < iters; t++) {
    const g = new Array(p).fill(0);
    const H = Array.from({ length: p }, () => new Array(p).fill(0));
    A.forEach((a, i) => {
      const z = a.reduce((s, v, j) => s + v * w[j], 0);
      const q = 1 / (1 + Math.exp(-z));
      const r = q - (y[i] + 1) / 2;
      const s = q * (1 - q);
      for (let j = 0; j < p; j++) {
        g[j] += r * a[j];
        for (let k = 0; k < p; k++) H[j][k] += s * a[j] * a[k];
      }
    });
    for (let j = 0; j < p - 1; j++) { g[j] += lambda * w[j]; H[j][j] += lambda; }
    H[p - 1][p - 1] += 1e-9;
    const d = solve(H, g);
    w = w.map((v, j) => v - d[j]);
  }
  return (x) => x.reduce((s, v, j) => s + v * w[j], w[p - 1]);
}

export const SIZES = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512];

/** Test error of naive Bayes and logistic regression versus training size, averaged over draws. */
export function learningCurves(dup: number, { k = 10, delta = 0.8, reps = 16, test = 2000 } = {}) {
  const T = dupClasses(test, k, dup, delta, 999);
  const err = (f: (x: number[]) => number) => T.X.reduce((s, x, i) => s + (T.y[i] * f(x) > 0 ? 0 : 1), 0) / test;
  const nb = SIZES.map(() => 0);
  const lr = SIZES.map(() => 0);
  SIZES.forEach((n, j) => {
    for (let r = 0; r < reps; r++) {
      const D = dupClasses(n, k, dup, delta, 1000 * r + n);
      nb[j] += err(gnb(D.X, D.y)) / reps;
      lr[j] += err(logisticP(D.X, D.y)) / reps;
    }
  });
  return { nb, lr };
}

/* ------------------------------------------------------------------ Laplace smoothing */

/** P(feature = 1 | class) for binary features with Laplace (add-α) smoothing: (count + α) / (n + 2α). */
export const bernoulliProb = (count: number, n: number, alpha: number) => (count + alpha) / (n + 2 * alpha);

/**
 * Posterior P(+ | x) of a Bernoulli naive Bayes model. counts[c][j] = number of class-c training
 * points with feature j = 1; ns[c] = class size; x = binary feature vector. Priors are frequencies.
 */
export function bernoulliPosterior(counts: [number[], number[]], ns: [number, number], x: number[], alpha: number): number {
  const score = [0, 1].map((c) => {
    let s = ns[c] / (ns[0] + ns[1]);
    x.forEach((v, j) => {
      const q = bernoulliProb(counts[c][j], ns[c], alpha);
      s *= v ? q : 1 - q;
    });
    return s;
  });
  return score[0] + score[1] === 0 ? 0.5 : score[0] / (score[0] + score[1]);
}

// Generative classifiers (chapter 05): Gaussian discriminant analysis (LDA / QDA), logistic regression
// by Newton's method, multinomial naive Bayes with Laplace smoothing, and the naive-Bayes-vs-logistic
// learning curves of Ng & Jordan (2002).
import { gaussian, rng } from "./data";
import type { Mat, Vec } from "./linalg";
import { solve } from "./regression";

// ---------- small helpers ----------

/** Standard normal CDF via the Abramowitz–Stegun erf approximation (|error| < 1.5e-7). */
export function normCdf(z: number): number {
  const t = 1 / (1 + (0.3275911 * Math.abs(z)) / Math.SQRT2);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-(z * z) / 2);
  return z >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
}

export const sigmoid = (t: number) => (t >= 0 ? 1 / (1 + Math.exp(-t)) : Math.exp(t) / (1 + Math.exp(t)));

const det2 = (A: Mat) => A[0][0] * A[1][1] - A[0][1] * A[1][0];
const inv2 = (A: Mat): Mat => {
  const d = det2(A);
  return [[A[1][1] / d, -A[0][1] / d], [-A[1][0] / d, A[0][0] / d]];
};
/** xᵀ A x for a 2-vector. */
const quad2 = (A: Mat, x: Vec) => A[0][0] * x[0] * x[0] + (A[0][1] + A[1][0]) * x[0] * x[1] + A[1][1] * x[1] * x[1];

/** Covariance with axis standard deviations s1, s2 and the first axis at angle theta: R diag(s1², s2²) Rᵀ. */
export function covFromAxes(s1: number, s2: number, theta: number): Mat {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [
    [s1 * s1 * c * c + s2 * s2 * s * s, (s1 * s1 - s2 * s2) * c * s],
    [(s1 * s1 - s2 * s2) * c * s, s1 * s1 * s * s + s2 * s2 * c * c],
  ];
}

/** n fixed standard-normal 2-vectors; moving a class's parameters then moves its points smoothly. */
export function standardNormals(n: number, seed: number): Vec[] {
  const rand = rng(seed);
  return Array.from({ length: n }, () => [gaussian(rand), gaussian(rand)] as Vec);
}

/** Map standard normals z to N(mean, R diag(s)²Rᵀ): x = mean + R diag(s) z. */
export function placeGaussian(z: Vec[], mean: Vec, s1: number, s2: number, theta: number): Vec[] {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return z.map(([a, b]) => [mean[0] + c * s1 * a - s * s2 * b, mean[1] + s * s1 * a + c * s2 * b] as Vec);
}

/** log N(x; mean, cov) in two dimensions. */
export function logGauss2(x: Vec, mean: Vec, cov: Mat): number {
  const d: Vec = [x[0] - mean[0], x[1] - mean[1]];
  return -Math.log(2 * Math.PI) - 0.5 * Math.log(det2(cov)) - 0.5 * quad2(inv2(cov), d);
}

// ---------- Gaussian discriminant analysis (labels ±1; "positive" is class 1) ----------

export interface ClassFit {
  prior: number;
  mean: Vec;
  /** Maximum-likelihood covariance (divide by n_k). */
  cov: Mat;
  n: number;
}

/** Per-class MLE: prior n_k / n, sample mean, covariance with 1/n_k. Returns [negative class, positive class]. */
export function fitClasses(X: number[][], y: number[]): [ClassFit, ClassFit] {
  const fit = (label: number): ClassFit => {
    const pts = X.filter((_, i) => (y[i] > 0) === (label > 0));
    const n = pts.length;
    const mean: Vec = [pts.reduce((s, p) => s + p[0], 0) / n, pts.reduce((s, p) => s + p[1], 0) / n];
    let a = 0, b = 0, c = 0;
    for (const p of pts) {
      const dx = p[0] - mean[0];
      const dy = p[1] - mean[1];
      a += dx * dx; b += dx * dy; c += dy * dy;
    }
    return { prior: n / X.length, mean, cov: [[a / n, b / n], [b / n, c / n]], n };
  };
  return [fit(-1), fit(1)];
}

/** Pooled (shared) covariance: Σ = Σ_k (n_k / n) Σ_k. */
export function pooledCov([c0, c1]: [ClassFit, ClassFit]): Mat {
  const n = c0.n + c1.n;
  const m = (i: number, j: number) => (c0.n * c0.cov[i][j] + c1.n * c1.cov[i][j]) / n;
  return [[m(0, 0), m(0, 1)], [m(1, 0), m(1, 1)]];
}

/** LDA log-odds log P(+|x)/P(−|x) = w·x + b with w = Σ⁻¹(μ₊ − μ₋), b = −½(μ₊ᵀΣ⁻¹μ₊ − μ₋ᵀΣ⁻¹μ₋) + log(π₊/π₋). */
export function lda2(X: number[][], y: number[]): { w: Vec; b: number } {
  const cls = fitClasses(X, y);
  return ldaFromParams(cls[0].mean, cls[1].mean, pooledCov(cls), cls[1].prior);
}

export function ldaFromParams(m0: Vec, m1: Vec, cov: Mat, prior1: number): { w: Vec; b: number } {
  const P = inv2(cov);
  const w: Vec = [P[0][0] * (m1[0] - m0[0]) + P[0][1] * (m1[1] - m0[1]), P[1][0] * (m1[0] - m0[0]) + P[1][1] * (m1[1] - m0[1])];
  const b = -0.5 * (quad2(P, m1) - quad2(P, m0)) + Math.log(prior1 / (1 - prior1));
  return { w, b };
}

/** QDA log-odds as a function: log π₊N(x; μ₊, Σ₊) − log π₋N(x; μ₋, Σ₋). */
export function qda2(X: number[][], y: number[]): (x: Vec) => number {
  const [c0, c1] = fitClasses(X, y);
  return qdaFromParams(c0.mean, c0.cov, c1.mean, c1.cov, c1.prior);
}

export function qdaFromParams(m0: Vec, S0: Mat, m1: Vec, S1: Mat, prior1: number): (x: Vec) => number {
  const lp = Math.log(prior1 / (1 - prior1));
  return (x) => logGauss2(x, m1, S1) - logGauss2(x, m0, S0) + lp;
}

// ---------- logistic regression (any dimension), Newton's method with an L2 penalty on w ----------

/**
 * Minimize Σᵢ log(1 + e^{−ỹᵢ(w·xᵢ + b)}) + (λ/2)‖w‖² by Newton's method (labels y ∈ {0, 1}).
 * Same objective as scikit-learn's LogisticRegression(C = 1/λ).
 */
export function logisticNewton(X: number[][], y01: number[], lambda = 1e-2, iters = 50): { w: number[]; b: number } {
  const d = X[0].length;
  const th = new Array(d + 1).fill(0);
  for (let it = 0; it < iters; it++) {
    const g = new Array(d + 1).fill(0);
    const H = Array.from({ length: d + 1 }, () => new Array(d + 1).fill(0));
    for (let i = 0; i < X.length; i++) {
      const z = [...X[i], 1];
      let t = 0;
      for (let j = 0; j <= d; j++) t += th[j] * z[j];
      const p = sigmoid(t);
      const r = p - y01[i];
      const s = p * (1 - p);
      for (let j = 0; j <= d; j++) {
        g[j] += r * z[j];
        for (let k = j; k <= d; k++) H[j][k] += s * z[j] * z[k];
      }
    }
    for (let j = 0; j <= d; j++) {
      for (let k = 0; k < j; k++) H[j][k] = H[k][j];
      if (j < d) { g[j] += lambda * th[j]; H[j][j] += lambda; }
      H[j][j] += 1e-9;
    }
    const step = solve(H, g);
    let size = 0;
    for (let j = 0; j <= d; j++) { th[j] -= step[j]; size += Math.abs(step[j]); }
    if (size < 1e-10) break;
  }
  return { w: th.slice(0, d), b: th[d] };
}

// ---------- multinomial naive Bayes ----------

/**
 * Smoothed word probabilities θ_kj = (N_kj + α) / (N_k + αV) for each class row of a count table.
 * α = 0 is the maximum-likelihood estimate; α = 1 is Laplace smoothing.
 */
export function wordProbs(counts: number[][], alpha: number): number[][] {
  return counts.map((row) => {
    const total = row.reduce((s, c) => s + c, 0);
    return row.map((c) => (c + alpha) / (total + alpha * row.length));
  });
}

/** Per-class log scores log π_k + Σ_j x_j log θ_kj for a message given as word counts (−∞ allowed). */
export function nbLogScores(counts: number[][], priors: number[], alpha: number, message: number[]): number[] {
  const th = wordProbs(counts, alpha);
  return th.map((row, k) => message.reduce((s, x, j) => (x === 0 ? s : s + x * Math.log(row[j])), Math.log(priors[k])));
}

/** Normalize log scores to posterior probabilities (log-sum-exp; all −∞ gives NaN). */
export function softmaxLog(scores: number[]): number[] {
  const m = Math.max(...scores);
  if (!Number.isFinite(m)) return scores.map(() => NaN);
  const e = scores.map((s) => Math.exp(s - m));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / z);
}

// ---------- generative vs discriminative learning curves ----------

/**
 * The learning-curve population: d0 independent features (feature 0 shifted by `strong`, the others by
 * `weak`) plus `copies` redundant copies of feature 0, correlated with it and with each other by rho.
 * Classes are N(0, S) and N(mu, S) with equal priors.
 */
export function redundantPopulation(rho: number, d0 = 10, copies = 4, weak = 0.45, strong = 0.8) {
  const d = d0 + copies;
  const mu = Array.from({ length: d }, (_, j) => (j === 0 || j >= d0 ? strong : weak));
  const linked = (j: number) => j === 0 || j >= d0;
  const S = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? 1 : linked(i) && linked(j) ? rho : 0)));
  return { mu, S };
}

function cholesky(S: number[][]): number[][] {
  const n = S.length;
  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = S[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j];
    }
  }
  return L;
}

/** Exact error of the rule "predict 1 iff w·x + b > 0" on the population (equal priors). */
export function linearRuleError(w: number[], b: number, mu: number[], S: number[][]): number {
  let wSw = 0;
  for (let i = 0; i < w.length; i++) for (let j = 0; j < w.length; j++) wSw += w[i] * S[i][j] * w[j];
  const sd = Math.sqrt(Math.max(wSw, 1e-300));
  const wm = w.reduce((s, v, i) => s + v * mu[i], 0);
  return 0.5 * (normCdf((b) / sd) + normCdf(-(wm + b) / sd));
}

/** Gaussian naive Bayes with one diagonal variance shared by both classes: a linear rule. */
export function gaussianNbLinear(X: number[][], y01: number[]): { w: number[]; b: number } {
  const d = X[0].length;
  const mean = [0, 1].map((k) => {
    const pts = X.filter((_, i) => y01[i] === k);
    return Array.from({ length: d }, (_, j) => pts.reduce((s, p) => s + p[j], 0) / pts.length);
  });
  const n1 = y01.filter((v) => v === 1).length;
  const v = Array.from({ length: d }, (_, j) => X.reduce((s, p, i) => s + (p[j] - mean[y01[i]][j]) ** 2, 0) / X.length + 1e-9);
  const w = v.map((vj, j) => (mean[1][j] - mean[0][j]) / vj);
  const b = -w.reduce((s, wj, j) => s + (wj * (mean[1][j] + mean[0][j])) / 2, 0) + Math.log(n1 / (X.length - n1));
  return { w, b };
}

export const CURVE_SIZES = [12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512];

/**
 * Average test error (computed exactly, not by sampling) of naive Bayes and logistic regression
 * trained on m samples (half per class), for each m in `sizes`, over `reps` training sets.
 */
export function learningCurves(rho: number, reps = 24, sizes = CURVE_SIZES, seed = 5) {
  const { mu, S } = redundantPopulation(rho);
  const d = mu.length;
  const L = cholesky(S);
  const rand = rng(seed);
  const draw = (shift: boolean) => {
    const z = Array.from({ length: d }, () => gaussian(rand));
    return Array.from({ length: d }, (_, i) => {
      let s = shift ? mu[i] : 0;
      for (let k = 0; k <= i; k++) s += L[i][k] * z[k];
      return s;
    });
  };
  const nb: number[] = [];
  const lr: number[] = [];
  for (const m of sizes) {
    let eNb = 0;
    let eLr = 0;
    for (let r = 0; r < reps; r++) {
      const X: number[][] = [];
      const y: number[] = [];
      for (let i = 0; i < m; i++) { const pos = i >= m / 2; X.push(draw(pos)); y.push(pos ? 1 : 0); }
      const g = gaussianNbLinear(X, y);
      eNb += linearRuleError(g.w, g.b, mu, S);
      const l = logisticNewton(X, y, 1e-2, 25);
      eLr += linearRuleError(l.w, l.b, mu, S);
    }
    nb.push(eNb / reps);
    lr.push(eLr / reps);
  }
  // Asymptotes: the Bayes rule uses S⁻¹mu; naive Bayes in the limit uses diag(S)⁻¹mu = mu.
  const wOpt = solve(S.map((r) => [...r]), [...mu]);
  const bayes = linearRuleError(wOpt, -wOpt.reduce((s, v, i) => s + v * mu[i], 0) / 2, mu, S);
  const nbInf = linearRuleError(mu, -mu.reduce((s, v) => s + v * v, 0) / 2, mu, S);
  return { sizes, nb, lr, bayes, nbInf };
}

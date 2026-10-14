// Chapter 04 (linear classification): perceptron, logistic regression, maximum entropy and the
// classification losses. Pure, deterministic maths for the widgets. No DOM, no React.
import { gaussian, r6, rng } from "./data";

export type V2 = [number, number];

/* ------------------------------------------------------------------ data */

/**
 * The screening data: two standardized blood markers, label +1 = needs a follow-up. The class
 * centres sit `gap` apart along a diagonal; `spread` is the within-class standard deviation.
 */
export function screening(n = 24, seed = 4, spread = 0.55, gap = 2.4): { X: V2[]; y: number[] } {
  const rand = rng(seed);
  const X: V2[] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const c = i % 2 === 0 ? 1 : -1;
    const cx = 0.3 + (c * gap * 0.8) / 2;
    const cy = 0.2 + (c * gap * 0.6) / 2;
    X.push([r6(cx + spread * gaussian(rand)), r6(cy + spread * gaussian(rand))]);
    y.push(c);
  }
  return { X, y };
}

/** Overlapping screening data for logistic regression (not separable; MLE accuracy 85 %). */
export const overlapping = () => screening(40, 2, 0.85, 1.8);

/* ------------------------------------------------------------------ perceptron */

export interface PerceptronStep {
  w: V2;
  b: number;
  /** Index of the point that triggered this update (−1 for the initial state). */
  i: number;
  /** Dual coefficients: α_i = η · (number of updates made on point i so far). */
  alpha: number[];
}

/**
 * Primal perceptron with w₀ = 0, b₀ = 0. Sweeps the points in `order`, updating on every point with
 * y(w·x + b) ≤ 0, until a full sweep makes no mistake. Returns the initial state plus one entry
 * per update.
 */
export function perceptron(X: V2[], y: number[], eta = 1, order?: number[], maxUpdates = 5000): { steps: PerceptronStep[]; converged: boolean } {
  const idx = order ?? X.map((_, i) => i);
  let w: V2 = [0, 0];
  let b = 0;
  const alpha = X.map(() => 0);
  const steps: PerceptronStep[] = [{ w: [0, 0], b: 0, i: -1, alpha: alpha.slice() }];
  for (let sweep = 0; steps.length <= maxUpdates; sweep++) {
    let clean = true;
    for (const i of idx) {
      if (y[i] * (w[0] * X[i][0] + w[1] * X[i][1] + b) <= 0) {
        w = [w[0] + eta * y[i] * X[i][0], w[1] + eta * y[i] * X[i][1]];
        b += eta * y[i];
        alpha[i] += eta;
        steps.push({ w, b, i, alpha: alpha.slice() });
        clean = false;
        if (steps.length > maxUpdates) break;
      }
    }
    if (clean) return { steps, converged: true };
  }
  return { steps, converged: false };
}

/** Dual form: w = Σ α_i y_i x_i, b = Σ α_i y_i. */
export function dualToPrimal(X: V2[], y: number[], alpha: number[]): { w: V2; b: number } {
  let w0 = 0, w1 = 0, b = 0;
  alpha.forEach((a, i) => { w0 += a * y[i] * X[i][0]; w1 += a * y[i] * X[i][1]; b += a * y[i]; });
  return { w: [w0, w1], b };
}

/**
 * Quantities of Novikoff's theorem in the augmented space x̂ = (x, 1): R = max‖x̂_i‖ and the best
 * margin γ = max_{‖v‖=1} min_i y_i v·x̂_i, from the hard-margin problem min ‖v‖² s.t. y_i v·x̂_i ≥ 1
 * (no free bias, so its dual has only α ≥ 0 and coordinate ascent solves it). γ = 0 if not separable.
 */
export function novikoff(X: V2[], y: number[], maxSweeps = 200000): { R: number; gamma: number; bound: number } {
  const A = X.map((x) => [x[0], x[1], 1]);
  const R = Math.sqrt(Math.max(...A.map((a) => a[0] ** 2 + a[1] ** 2 + a[2] ** 2)));
  const alpha = A.map(() => 0);
  const v = [0, 0, 0];
  for (let t = 0; t < maxSweeps; t++) {
    let moved = 0;
    for (let i = 0; i < A.length; i++) {
      const a = A[i];
      const q = a[0] ** 2 + a[1] ** 2 + a[2] ** 2;
      const m = y[i] * (v[0] * a[0] + v[1] * a[1] + v[2] * a[2]);
      const next = Math.max(0, alpha[i] + (1 - m) / q);
      const d = (next - alpha[i]) * y[i];
      alpha[i] = next;
      moved = Math.max(moved, Math.abs(d));
      v[0] += d * a[0]; v[1] += d * a[1]; v[2] += d * a[2];
    }
    if (moved < 1e-13) break;
  }
  const nv = Math.hypot(v[0], v[1], v[2]);
  const ok = A.every((a, i) => y[i] * (v[0] * a[0] + v[1] * a[1] + v[2] * a[2]) >= 1 - 1e-6);
  const gamma = ok && nv > 0 ? 1 / nv : 0;
  return { R, gamma, bound: gamma > 0 ? (R / gamma) ** 2 : Infinity };
}

/** Number of training points with y(w·x + b) ≤ 0. */
export const mistakes = (X: V2[], y: number[], w: V2, b: number) =>
  X.reduce((s, x, i) => s + (y[i] * (w[0] * x[0] + w[1] * x[1] + b) <= 0 ? 1 : 0), 0);

/** A deterministic permutation of 0..n−1. */
export function shuffled(n: number, seed: number): number[] {
  const rand = rng(seed);
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/* ------------------------------------------------------------------ logistic regression */

export const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
/** log(1 + e^{−m}) without overflow. */
export const softplusNeg = (m: number) => (m > 0 ? Math.log1p(Math.exp(-m)) : -m + Math.log1p(Math.exp(m)));

/** Parameters θ = (w₁, w₂, b); labels y ∈ {−1, +1}. */
export type Theta = [number, number, number];

/** Mean cross-entropy (negative log-likelihood per point): mean log(1 + e^{−y f(x)}). */
export function crossEntropy(X: V2[], y: number[], t: Theta): number {
  let s = 0;
  X.forEach((x, i) => { s += softplusNeg(y[i] * (t[0] * x[0] + t[1] * x[1] + t[2])); });
  return s / X.length;
}

/** Gradient of the mean cross-entropy: mean (σ(f) − y01) x̂, with y01 = (y + 1)/2. */
export function ceGrad(X: V2[], y: number[], t: Theta): Theta {
  const g: Theta = [0, 0, 0];
  X.forEach((x, i) => {
    const r = sigmoid(t[0] * x[0] + t[1] * x[1] + t[2]) - (y[i] + 1) / 2;
    g[0] += r * x[0]; g[1] += r * x[1]; g[2] += r;
  });
  return g.map((v) => v / X.length) as Theta;
}

/** Hessian of the mean cross-entropy: mean σ(1−σ) x̂ x̂ᵀ (positive semidefinite). */
export function ceHessian(X: V2[], t: Theta): number[][] {
  const H = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  X.forEach((x) => {
    const p = sigmoid(t[0] * x[0] + t[1] * x[1] + t[2]);
    const s = p * (1 - p);
    const a = [x[0], x[1], 1];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) H[r][c] += (s * a[r] * a[c]) / X.length;
  });
  return H;
}

/** Solve a 3×3 system by Gaussian elimination with partial pivoting. */
export function solve3(A: number[][], b: number[]): number[] {
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < 3; c++) {
    let p = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < 3; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k < 4; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = [0, 0, 0];
  for (let r = 2; r >= 0; r--) {
    let s = M[r][3];
    for (let k = r + 1; k < 3; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

/** Newton's method (= IRLS) from θ = 0. Returns every iterate; stops when ‖∇‖ < tol. */
export function newton(X: V2[], y: number[], maxIter = 25, tol = 1e-8): Theta[] {
  let t: Theta = [0, 0, 0];
  const path: Theta[] = [t];
  for (let k = 0; k < maxIter; k++) {
    const g = ceGrad(X, y, t);
    if (Math.hypot(...g) < tol) break;
    const H = ceHessian(X, t);
    // A tiny ridge keeps the step finite when the Hessian degenerates (separable data).
    const d = solve3(H.map((row, i) => row.map((v, j) => v + (i === j ? 1e-10 : 0))), g);
    t = [t[0] - d[0], t[1] - d[1], t[2] - d[2]];
    path.push(t);
  }
  return path;
}

/** Plain gradient descent with a fixed step: number of iterations until ‖∇‖ < tol (capped). */
export function gdIterations(X: V2[], y: number[], step: number, tol = 1e-4, cap = 100000): number {
  let t: Theta = [0, 0, 0];
  for (let k = 0; k < cap; k++) {
    const g = ceGrad(X, y, t);
    if (Math.hypot(...g) < tol) return k;
    t = [t[0] - step * g[0], t[1] - step * g[1], t[2] - step * g[2]];
  }
  return cap;
}

/** Iterations Newton needs until ‖∇‖ < tol. */
export function newtonIterations(X: V2[], y: number[], tol = 1e-4): number {
  const path = newton(X, y, 50, tol);
  return path.length - 1;
}

/**
 * The widget's parametrisation: direction angle θ (deg), steepness s = ‖w‖ and boundary offset c
 * along the direction, i.e. w = s(cos θ, sin θ), b = −s c. Returns θ = (w₁, w₂, b).
 */
export const fromPolar = (deg: number, s: number, c: number): Theta => {
  const a = (deg * Math.PI) / 180;
  return [s * Math.cos(a), s * Math.sin(a), -s * c];
};
export const toPolar = (t: Theta): { deg: number; s: number; c: number } => {
  const s = Math.hypot(t[0], t[1]);
  return { deg: (Math.atan2(t[1], t[0]) * 180) / Math.PI, s, c: s > 0 ? -t[2] / s : 0 };
};

/* ------------------------------------------------------------------ losses of the margin m = y f(x) */

export type LossId = "zeroOne" | "perceptron" | "logistic" | "hinge" | "exp";
export const LOSS_IDS: LossId[] = ["zeroOne", "perceptron", "logistic", "hinge", "exp"];

/** The loss as a function of the margin. The logistic loss is in base 2 so that it passes through (0, 1). */
export const loss: Record<LossId, (m: number) => number> = {
  zeroOne: (m) => (m <= 0 ? 1 : 0),
  perceptron: (m) => Math.max(0, -m),
  logistic: (m) => softplusNeg(m) / Math.LN2,
  hinge: (m) => Math.max(0, 1 - m),
  exp: (m) => Math.exp(-m),
};

/** −ℓ′(m): how hard a point with margin m pulls on the parameters (its weight in the gradient). */
export const pull: Record<LossId, (m: number) => number> = {
  zeroOne: () => 0,
  perceptron: (m) => (m <= 0 ? 1 : 0),
  logistic: (m) => sigmoid(-m) / Math.LN2,
  hinge: (m) => (m < 1 ? 1 : 0),
  exp: (m) => Math.exp(-m),
};

/* ------------------------------------------------------------------ maximum entropy on a die */

export const entropy = (p: number[]) => -p.reduce((s, q) => s + (q > 0 ? q * Math.log(q) : 0), 0);

/** Distribution p_k ∝ e^{λk} on faces 1..6. */
export const gibbsDie = (lambda: number) => {
  const e = [1, 2, 3, 4, 5, 6].map((k) => Math.exp(lambda * (k - 3.5)));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / z);
};
const meanOf = (p: number[]) => p.reduce((s, q, k) => s + q * (k + 1), 0);

/** The maximum-entropy distribution on {1..6} with mean μ: p ∝ e^{λk}, λ found by bisection. */
export function maxentDie(mu: number): { p: number[]; lambda: number; H: number } {
  let lo = -30, hi = 30;
  for (let k = 0; k < 200; k++) {
    const mid = (lo + hi) / 2;
    if (meanOf(gibbsDie(mid)) < mu) lo = mid; else hi = mid;
  }
  const lambda = (lo + hi) / 2;
  const p = gibbsDie(lambda);
  return { p, lambda, H: entropy(p) };
}

/** Another distribution with the same mean μ ∈ [1, 6]: all mass on the two faces around μ. */
export function twoFaceDie(mu: number): number[] {
  const lo = Math.min(5, Math.floor(mu));
  const t = mu - lo;
  return [1, 2, 3, 4, 5, 6].map((k) => (k === lo ? 1 - t : k === lo + 1 ? t : 0));
}

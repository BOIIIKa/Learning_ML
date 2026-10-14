// Linear classifiers for chapter 04: the perceptron (primal and dual), logistic regression by
// Newton's method (IRLS), the maximum-entropy die and the zoo of margin losses.
import { gaussian, r6, rng, type Dataset } from "./data";

export const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

/** log(1 + e^{−m}) computed without overflow. */
export const softplusNeg = (m: number) => (m > 0 ? Math.log1p(Math.exp(-m)) : -m + Math.log1p(Math.exp(m)));

// ---------------------------------------------------------------------------------------------
// Data: the triage scenario. Feature 1 = standardized fasting glucose, feature 2 = standardized BMI;
// label +1 = later diagnosed with diabetes, −1 = not.

/** Two overlapping groups of patients (not linearly separable). */
export function patients(n = 60, seed = 11, spread = 0.78): Dataset {
  const rand = rng(seed);
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const label = i % 2 === 0 ? -1 : 1;
    const [cx, cy] = label < 0 ? [-0.75, -0.55] : [0.8, 0.6];
    X.push([r6(cx + spread * gaussian(rand)), r6(cy + spread * gaussian(rand))]);
    y.push(label);
  }
  return { X, y };
}

/** The separator used to label `separablePatients`. */
export const TRUE_SEPARATOR = { w: [0.8, 0.6] as [number, number], b: -0.2 };

/** A small linearly separable sample: points closer than `gap` to the true separator are rejected. */
export function separablePatients(n = 14, seed = 5, gap = 0.3): Dataset {
  const rand = rng(seed);
  const { w, b } = TRUE_SEPARATOR;
  const X: number[][] = [];
  const y: number[] = [];
  while (X.length < n) {
    const p = [r6(-2.5 + 5 * rand()), r6(-2.5 + 5 * rand())];
    const d = w[0] * p[0] + w[1] * p[1] + b; // ‖w‖ = 1, so d is the signed distance
    if (Math.abs(d) < gap) continue;
    X.push(p);
    y.push(d > 0 ? 1 : -1);
  }
  return { X, y };
}

// ---------------------------------------------------------------------------------------------
// The perceptron.

export interface PerceptronStep {
  w: [number, number];
  b: number;
  /** Index of the misclassified point that triggered this update (−1 for the initial state). */
  i: number;
  /** α_i = number of updates on point i so far (times η in the dual form). */
  alpha: number[];
  /** Training points misclassified (y·f ≤ 0) by this (w, b). */
  mistakes: number;
}

/**
 * Primal perceptron: scan the points in `order` cyclically and update on every mistake
 * (y_i(w·x_i + b) ≤ 0): w ← w + η y_i x_i, b ← b + η y_i. Stops after a full clean pass.
 */
export function perceptron(X: number[][], y: number[], eta = 1, order?: number[], maxUpdates = 500): PerceptronStep[] {
  const n = y.length;
  const ord = order ?? X.map((_, i) => i);
  let w: [number, number] = [0, 0];
  let b = 0;
  const alpha = new Array(n).fill(0);
  const count = () => X.filter((x, k) => y[k] * (w[0] * x[0] + w[1] * x[1] + b) <= 0).length;
  const steps: PerceptronStep[] = [{ w: [0, 0], b: 0, i: -1, alpha: [...alpha], mistakes: count() }];
  let clean = 0;
  let pos = 0;
  while (clean < n && steps.length <= maxUpdates) {
    const i = ord[pos];
    if (y[i] * (w[0] * X[i][0] + w[1] * X[i][1] + b) <= 0) {
      w = [r6(w[0] + eta * y[i] * X[i][0]), r6(w[1] + eta * y[i] * X[i][1])];
      b = r6(b + eta * y[i]);
      alpha[i] += 1;
      steps.push({ w, b, i, alpha: [...alpha], mistakes: count() });
      clean = 0;
    } else clean += 1;
    pos = (pos + 1) % n;
  }
  return steps;
}

/**
 * Dual perceptron: keep α (update counts × η) and use only the Gram matrix G_ij = x_i·x_j.
 * f(x_i) = Σ_j α_j y_j G_ji + b with b = Σ_j α_j y_j. Same visiting rule as the primal form.
 */
export function perceptronDual(X: number[][], y: number[], eta = 1, order?: number[], maxUpdates = 500) {
  const n = y.length;
  const ord = order ?? X.map((_, i) => i);
  const G = X.map((a) => X.map((c) => a[0] * c[0] + a[1] * c[1]));
  const alpha = new Array(n).fill(0);
  let b = 0;
  let updates = 0;
  let clean = 0;
  let pos = 0;
  while (clean < n && updates < maxUpdates) {
    const i = ord[pos];
    let f = b;
    for (let j = 0; j < n; j++) f += alpha[j] * y[j] * G[j][i];
    if (y[i] * f <= 0) {
      alpha[i] += eta;
      b += eta * y[i];
      updates++;
      clean = 0;
    } else clean += 1;
    pos = (pos + 1) % n;
  }
  const w: [number, number] = [0, 0];
  for (let j = 0; j < n; j++) {
    w[0] += alpha[j] * y[j] * X[j][0];
    w[1] += alpha[j] * y[j] * X[j][1];
  }
  return { alpha, b, w, updates, gram: G };
}

/**
 * Novikoff's bound (R/γ)² for a given unit-norm separator (w, b), using augmented vectors x̂ = (x, 1):
 * R = max‖x̂_i‖ and γ = min_i y_i(ŵ·x̂_i) with ŵ = (w, b)/‖(w, b)‖.
 */
export function novikoffBound(X: number[][], y: number[], w: [number, number], b: number) {
  const norm = Math.hypot(w[0], w[1], b);
  const R = Math.max(...X.map((x) => Math.hypot(x[0], x[1], 1)));
  const gamma = Math.min(...X.map((x, i) => (y[i] * (w[0] * x[0] + w[1] * x[1] + b)) / norm));
  return { R, gamma, bound: (R / gamma) ** 2 };
}

// ---------------------------------------------------------------------------------------------
// Logistic regression with labels y ∈ {−1, +1}: P(y | x) = σ(y f(x)), f(x) = w·x + b.

/** Mean cross-entropy (1/n) Σ log(1 + e^{−y_i f(x_i)}) in nats. */
export function crossEntropy(w: [number, number], b: number, X: number[][], y: number[]) {
  let s = 0;
  for (let i = 0; i < y.length; i++) s += softplusNeg(y[i] * (w[0] * X[i][0] + w[1] * X[i][1] + b));
  return s / y.length;
}

export function accuracy(w: [number, number], b: number, X: number[][], y: number[]) {
  return X.filter((x, i) => y[i] * (w[0] * x[0] + w[1] * x[1] + b) > 0).length / y.length;
}

function solve3(A: number[][], g: number[]): number[] {
  // Gaussian elimination with partial pivoting for a 3×3 system.
  const M = A.map((row, i) => [...row, g[i]]);
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

/**
 * One Newton step on the summed negative log-likelihood (with an optional ridge λ‖w‖²/2):
 * θ ← θ + H⁻¹ ∇ℓ, ∇ℓ = Σ (t_i − p_i) x̂_i, H = Σ p_i(1 − p_i) x̂_i x̂_iᵀ, t = (y + 1)/2, x̂ = (x, 1).
 */
export function newtonStep(w: [number, number], b: number, X: number[][], y: number[], lambda = 0) {
  const g = [0, 0, 0];
  const H = [[lambda, 0, 0], [0, lambda, 0], [0, 0, 1e-9]];
  for (let i = 0; i < y.length; i++) {
    const xh = [X[i][0], X[i][1], 1];
    const p = sigmoid(w[0] * xh[0] + w[1] * xh[1] + b);
    const r = (y[i] + 1) / 2 - p;
    const s = p * (1 - p);
    for (let a = 0; a < 3; a++) {
      g[a] += r * xh[a];
      for (let c = 0; c < 3; c++) H[a][c] += s * xh[a] * xh[c];
    }
  }
  g[0] -= lambda * w[0];
  g[1] -= lambda * w[1];
  const d = solve3(H, g);
  return { w: [w[0] + d[0], w[1] + d[1]] as [number, number], b: b + d[2], gradNorm: Math.hypot(...g) };
}

/** Newton's method (IRLS) from (w, b) = 0; returns every iterate. */
export function fitLogistic(X: number[][], y: number[], iters = 10, lambda = 0) {
  const path = [{ w: [0, 0] as [number, number], b: 0, loss: crossEntropy([0, 0], 0, X, y), gradNorm: NaN }];
  for (let k = 0; k < iters; k++) {
    const cur = path[path.length - 1];
    const nx = newtonStep(cur.w, cur.b, X, y, lambda);
    path[path.length - 1].gradNorm = nx.gradNorm;
    path.push({ w: nx.w, b: nx.b, loss: crossEntropy(nx.w, nx.b, X, y), gradNorm: NaN });
  }
  return path;
}

// ---------------------------------------------------------------------------------------------
// Maximum entropy on a die: maximize H(p) subject to Σ p_k = 1 and Σ k p_k = μ ⇒ p_k ∝ e^{λk}.

export function maxEntDie(mu: number, faces = 6) {
  const ks = Array.from({ length: faces }, (_, k) => k + 1);
  const dist = (l: number) => {
    const m = Math.max(...ks.map((k) => l * k));
    const q = ks.map((k) => Math.exp(l * k - m));
    const z = q.reduce((a, c) => a + c, 0);
    return q.map((v) => v / z);
  };
  const mean = (l: number) => dist(l).reduce((a, p, k) => a + p * (k + 1), 0);
  // mean(λ) increases monotonically in λ (its derivative is a variance), so bisect.
  let lo = -30;
  let hi = 30;
  for (let it = 0; it < 200; it++) {
    const mid = (lo + hi) / 2;
    if (mean(mid) < mu) lo = mid;
    else hi = mid;
  }
  const lambda = (lo + hi) / 2;
  const p = dist(lambda);
  return { lambda, p, entropy: entropy(p) };
}

export const entropy = (p: number[]) => -p.reduce((a, v) => a + (v > 0 ? v * Math.log(v) : 0), 0);

/**
 * Another distribution on 1..faces with the same mean: move from p along a random direction that keeps
 * Σ p_k and Σ k p_k fixed, as far as `frac` of the way to the simplex boundary.
 */
export function sameMeanAlternative(p: number[], seed: number, frac = 0.85) {
  const rand = rng(seed);
  const K = p.length;
  const ones = new Array(K).fill(1 / Math.sqrt(K));
  const ks = Array.from({ length: K }, (_, k) => k + 1);
  const kc = ks.map((k) => k - (K + 1) / 2);
  const kn = Math.hypot(...kc);
  const e2 = kc.map((v) => v / kn); // orthonormal to `ones`
  let v = ks.map(() => gaussian(rand));
  for (const e of [ones, e2]) {
    const d = v.reduce((a, c, i) => a + c * e[i], 0);
    v = v.map((c, i) => c - d * e[i]);
  }
  let tMax = Infinity;
  for (let i = 0; i < K; i++) if (v[i] < 0) tMax = Math.min(tMax, -p[i] / v[i]);
  const q = p.map((pi, i) => Math.max(0, pi + frac * tMax * v[i]));
  const z = q.reduce((a, c) => a + c, 0);
  return q.map((c) => c / z);
}

// ---------------------------------------------------------------------------------------------
// Margin losses L(m) with m = y f(x).

export type LossKey = "zeroOne" | "perceptron" | "logistic" | "hinge" | "exponential";

export const LOSSES: Record<LossKey, (m: number) => number> = {
  zeroOne: (m) => (m <= 0 ? 1 : 0),
  perceptron: (m) => Math.max(0, -m),
  /** In bits, log₂(1 + e^{−m}), so it passes through (0, 1) and bounds the 0-1 loss from above. */
  logistic: (m) => softplusNeg(m) / Math.LN2,
  hinge: (m) => Math.max(0, 1 - m),
  exponential: (m) => Math.exp(-m),
};

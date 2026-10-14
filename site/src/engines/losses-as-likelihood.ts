// Chapter 15 (Loss Functions as Maximum Likelihood): the recipe "choose an output distribution,
// minimise its negative log-likelihood", heteroscedastic regression, and cross-entropy = H + KL.
import { gaussian, r6, rng } from "./data";
import { solve } from "./regression";

export const softplus = (z: number) => (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)));
export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
const lgamma = (n: number) => { let s = 0; for (let k = 2; k <= n; k++) s += Math.log(k); return s; }; // integers only

export type DistId = "gaussian" | "bernoulli" | "categorical" | "laplace" | "poisson";

/** Negative log-likelihood of one observation y given the network's raw output f, and its derivative.
 * Gaussian (σ = 1) and Laplace (b = 1): f is the location. Bernoulli: f is the logit. Categorical:
 * three classes with logits (f, 0, 0), y is the true class. Poisson: f is the log-rate. */
export const DISTS: Record<DistId, { nll: (y: number, f: number) => number; grad: (y: number, f: number) => number; mean: (f: number) => number }> = {
  gaussian: { nll: (y, f) => 0.5 * (y - f) ** 2 + 0.5 * Math.log(2 * Math.PI), grad: (y, f) => f - y, mean: (f) => f },
  bernoulli: { nll: (y, f) => softplus(f) - y * f, grad: (y, f) => sigmoid(f) - y, mean: sigmoid },
  categorical: {
    nll: (y, f) => Math.log(Math.exp(f) + 2) - (y === 0 ? f : 0),
    grad: (y, f) => Math.exp(f) / (Math.exp(f) + 2) - (y === 0 ? 1 : 0),
    mean: (f) => Math.exp(f) / (Math.exp(f) + 2),
  },
  laplace: { nll: (y, f) => Math.abs(y - f) + Math.log(2), grad: (y, f) => Math.sign(f - y), mean: (f) => f },
  poisson: { nll: (y, f) => Math.exp(f) - y * f + lgamma(y), grad: (y, f) => Math.exp(f) - y, mean: Math.exp },
};

/* ------------------------------------------------------------------ heteroscedastic regression */

/** Study hours vs score where the spread grows with the hours: σ(x) = 1.5 + 0.9x. */
export const heteroTruth = { mean: (x: number) => 40 + 4 * x, sd: (x: number) => 1.5 + 0.9 * x };
export function heteroData(n = 90, seed = 3) {
  const rand = rng(seed);
  const xs = Array.from({ length: n }, () => r6(10 * rand()));
  return { xs, ys: xs.map((x) => r6(heteroTruth.mean(x) + heteroTruth.sd(x) * gaussian(rand))) };
}

export interface GaussFit { w: [number, number]; v: [number, number]; mu: (x: number) => number; sd: (x: number) => number }

const fitOf = (w: [number, number], v: [number, number]): GaussFit => ({ w, v, mu: (x) => w[0] + w[1] * x, sd: (x) => Math.exp(v[0] + v[1] * x) });

/** Homoscedastic maximum likelihood: least-squares line, σ = RMS residual. */
export function fitHomo(xs: number[], ys: number[]): GaussFit {
  const n = xs.length, mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n;
  const b = xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0) / xs.reduce((s, x) => s + (x - mx) ** 2, 0);
  const w: [number, number] = [my - b * mx, b];
  const s2 = xs.reduce((s, x, i) => s + (ys[i] - w[0] - w[1] * x) ** 2, 0) / n;
  return fitOf(w, [0.5 * Math.log(s2), 0]);
}

/** Heteroscedastic maximum likelihood, log σ(x) = v₀ + v₁x: alternate weighted least squares for the
 * mean and Newton steps for v (for a fixed mean the NLL is convex in v). */
export function fitHetero(xs: number[], ys: number[], rounds = 60): GaussFit {
  let { w, v } = fitHomo(xs, ys);
  for (let r = 0; r < rounds; r++) {
    // Mean: minimise Σ (y − w₀ − w₁x)² / σ_i².
    const A = [[0, 0], [0, 0]], b = [0, 0];
    xs.forEach((x, i) => {
      const q = Math.exp(-2 * (v[0] + v[1] * x));
      A[0][0] += q; A[0][1] += q * x; A[1][1] += q * x * x;
      b[0] += q * ys[i]; b[1] += q * x * ys[i];
    });
    A[1][0] = A[0][1];
    w = solve(A, b) as [number, number];
    // Spread: Newton on Σ [ r² e^{−2s} / 2 + s ],  s = v₀ + v₁x.
    for (let k = 0; k < 3; k++) {
      const g = [0, 0], H = [[0, 0], [0, 0]];
      xs.forEach((x, i) => {
        const r2 = (ys[i] - w[0] - w[1] * x) ** 2, e = Math.exp(-2 * (v[0] + v[1] * x));
        const d1 = 1 - r2 * e, d2 = 2 * r2 * e;
        g[0] += d1; g[1] += d1 * x;
        H[0][0] += d2; H[0][1] += d2 * x; H[1][1] += d2 * x * x;
      });
      H[1][0] = H[0][1];
      const step = solve(H, g);
      v = [v[0] - step[0], v[1] - step[1]];
    }
  }
  return fitOf(w, v);
}

/** Average Gaussian negative log-likelihood of the data under a fit. */
export const avgNll = (m: GaussFit, xs: number[], ys: number[]) =>
  xs.reduce((s, x, i) => s + (ys[i] - m.mu(x)) ** 2 / (2 * m.sd(x) ** 2) + Math.log(m.sd(x)) + 0.5 * Math.log(2 * Math.PI), 0) / xs.length;

/** Share of points inside the ±2σ band, separately for x below / above a split. */
export function coverage(m: GaussFit, xs: number[], ys: number[], split = 5) {
  const inside = (i: number) => Math.abs(ys[i] - m.mu(xs[i])) <= 2 * m.sd(xs[i]);
  const lo = xs.map((x, i) => i).filter((i) => xs[i] < split), hi = xs.map((x, i) => i).filter((i) => xs[i] >= split);
  return { low: lo.filter(inside).length / lo.length, high: hi.filter(inside).length / hi.length };
}

/* ------------------------------------------------------------------ cross-entropy = entropy + KL */

const xlogy = (x: number, y: number) => (x === 0 ? 0 : x * Math.log(y));
/** Bernoulli entropy H(p), cross-entropy H(p, q) and KL(p ‖ q), in nats. */
export const entropy = (p: number) => -xlogy(p, p) - xlogy(1 - p, 1 - p);
export const crossEntropy = (p: number, q: number) => -xlogy(p, q) - xlogy(1 - p, 1 - q);
export const kl = (p: number, q: number) => xlogy(p, p / q) + xlogy(1 - p, (1 - p) / (1 - q));

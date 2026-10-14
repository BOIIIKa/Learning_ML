// Chapter 11 (EM and mixture models): EM for 2-D Gaussian mixtures, a 1-D model where the ELBO can
// be drawn, and the hard-assignment limit. Pure, deterministic maths. No DOM, no React.
import { gaussian, r6, rng } from "./data";
import { det2, inv2, type M2 } from "./generative-classifiers";
import type { V2 } from "./linear-classification";

/** Three patient subtypes with different shapes: one elongated, one round, one tilted. */
export function shapedGroups(n = 150, seed = 2): V2[] {
  const rand = rng(seed);
  const C: V2[] = [[-1.4, -0.4], [1.4, -0.8], [0.3, 1.5]];
  const L: M2[] = [[[0.9, 0], [0.55, 0.2]], [[0.35, 0], [0, 0.35]], [[0.25, 0], [-0.3, 0.55]]];
  return Array.from({ length: n }, (_, i) => {
    const k = i % 3;
    const z = [gaussian(rand), gaussian(rand)];
    return [r6(C[k][0] + L[k][0][0] * z[0]), r6(C[k][1] + L[k][1][0] * z[0] + L[k][1][1] * z[1])] as V2;
  });
}

export interface Gmm { w: number[]; mu: V2[]; S: M2[] }

const logN = (x: V2, m: V2, S: M2) => {
  const Si = inv2(S);
  const d = [x[0] - m[0], x[1] - m[1]];
  return -0.5 * (d[0] * (Si[0][0] * d[0] + Si[0][1] * d[1]) + d[1] * (Si[1][0] * d[0] + Si[1][1] * d[1])) - 0.5 * Math.log(det2(S)) - Math.log(2 * Math.PI);
};
const lse = (v: number[]) => { const m = Math.max(...v); return m + Math.log(v.reduce((s, x) => s + Math.exp(x - m), 0)); };

/** E-step: responsibilities r[i][k] and the log-likelihood. */
export function eStep(X: V2[], g: Gmm): { r: number[][]; ll: number } {
  let ll = 0;
  const r = X.map((x) => {
    const a = g.w.map((w, k) => Math.log(w) + logN(x, g.mu[k], g.S[k]));
    const z = lse(a);
    ll += z;
    return a.map((v) => Math.exp(v - z));
  });
  return { r, ll };
}

/** M-step: weighted counts, means and covariances (with a small ridge to stay invertible). */
export function mStep(X: V2[], r: number[][]): Gmm {
  const K = r[0].length;
  const N = Array.from({ length: K }, (_, k) => r.reduce((s, ri) => s + ri[k], 0));
  const mu = N.map((nk, k) => [0, 1].map((j) => r.reduce((s, ri, i) => s + ri[k] * X[i][j], 0) / nk) as V2);
  const S = N.map((nk, k) => {
    const M: M2 = [[1e-4, 0], [0, 1e-4]];
    X.forEach((x, i) => {
      const d = [x[0] - mu[k][0], x[1] - mu[k][1]];
      for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) M[a][b] += (r[i][k] * d[a] * d[b]) / nk;
    });
    return M;
  });
  return { w: N.map((nk) => nk / X.length), mu, S };
}

export function emRun(X: V2[], g0: Gmm, iters = 40): { models: Gmm[]; ll: number[]; r: number[][][] } {
  const models: Gmm[] = [];
  const ll: number[] = [];
  const rs: number[][][] = [];
  let g = g0;
  for (let t = 0; t <= iters; t++) {
    const e = eStep(X, g);
    models.push(g);
    ll.push(e.ll);
    rs.push(e.r);
    g = mStep(X, e.r);
  }
  return { models, ll, r: rs };
}

/* ------------------------------------------------------------------ a 1-D model with a drawable ELBO */

/** Data from 0.5·N(0, 1) + 0.5·N(θ*, 1); the model knows everything except θ. */
export function oneD(n = 40, theta = 2.5, seed = 4): number[] {
  const rand = rng(seed);
  return Array.from({ length: n }, (_, i) => r6((i % 2 ? theta : 0) + gaussian(rand)));
}
const phi = (x: number, m: number) => Math.exp(-0.5 * (x - m) ** 2) / Math.sqrt(2 * Math.PI);

/** log p(x | θ) = Σ log(½φ(x − 0) + ½φ(x − θ)). */
export const loglik1 = (xs: number[], th: number) => xs.reduce((s, x) => s + Math.log(0.5 * phi(x, 0) + 0.5 * phi(x, th)), 0);

/** q from the E-step at θ_t: q_i = P(z_i = θ-component | x_i, θ_t). */
export const resp1 = (xs: number[], th: number) => xs.map((x) => (0.5 * phi(x, th)) / (0.5 * phi(x, 0) + 0.5 * phi(x, th)));

/** ELBO(q, θ) = Σ_i Σ_k q_ik [log p(x_i, z_i = k | θ) − log q_ik]. */
export function elbo1(xs: number[], q: number[], th: number): number {
  const h = (p: number) => (p > 0 ? -p * Math.log(p) : 0);
  return xs.reduce((s, x, i) => s + (1 - q[i]) * Math.log(0.5 * phi(x, 0)) + q[i] * Math.log(0.5 * phi(x, th)) + h(q[i]) + h(1 - q[i]), 0);
}

/** M-step for θ: the q-weighted mean. */
export const mStep1 = (xs: number[], q: number[]) => xs.reduce((s, x, i) => s + q[i] * x, 0) / q.reduce((s, v) => s + v, 0);

/* ------------------------------------------------------------------ the hard-assignment limit */

/** Responsibility of the component at m2 (equal weights, shared variance s = σ²) for a point x. */
export const softAssign = (x: number, m1: number, m2: number, s: number) => 1 / (1 + Math.exp(-((m2 - m1) * (x - (m1 + m2) / 2)) / s));

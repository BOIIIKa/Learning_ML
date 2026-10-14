// Chapter 09 (structured prediction): hidden Markov models (forward, backward, Viterbi,
// Baum–Welch) and the partition function of a linear-chain CRF. Pure, deterministic maths.
import { rng } from "./data";

export interface Hmm { pi: number[]; A: number[][]; B: number[][] }

/** The chapter's model: states 0 = stable, 1 = deteriorating; observations 0 = normal, 1 = abnormal test. */
export const HMM: Hmm = { pi: [0.6, 0.4], A: [[0.7, 0.3], [0.4, 0.6]], B: [[0.9, 0.1], [0.2, 0.8]] };
export const EXAMPLE_OBS = [0, 1, 1];

/** Forward variables α_t(i) = P(o_1..o_t, s_t = i) (unscaled; fine for short sequences). */
export function forward(m: Hmm, O: number[]): number[][] {
  const N = m.pi.length;
  const al = [m.pi.map((p, i) => p * m.B[i][O[0]])];
  for (let t = 1; t < O.length; t++) {
    al.push(Array.from({ length: N }, (_, j) => al[t - 1].reduce((s, a, i) => s + a * m.A[i][j], 0) * m.B[j][O[t]]));
  }
  return al;
}

/** Backward variables β_t(i) = P(o_{t+1}..o_T | s_t = i). */
export function backward(m: Hmm, O: number[]): number[][] {
  const N = m.pi.length;
  const be: number[][] = [new Array(N).fill(1)];
  for (let t = O.length - 2; t >= 0; t--) {
    const next = be[0];
    be.unshift(Array.from({ length: N }, (_, i) => next.reduce((s, b, j) => s + m.A[i][j] * m.B[j][O[t + 1]] * b, 0)));
  }
  return be;
}

/** Viterbi: δ_t(j) = max_i δ_{t−1}(i) A_ij · B_j(o_t), with back-pointers and the best path. */
export function viterbi(m: Hmm, O: number[]): { delta: number[][]; back: number[][]; path: number[]; prob: number } {
  const N = m.pi.length;
  const delta = [m.pi.map((p, i) => p * m.B[i][O[0]])];
  const back: number[][] = [];
  for (let t = 1; t < O.length; t++) {
    const row: number[] = [];
    const bp: number[] = [];
    for (let j = 0; j < N; j++) {
      let best = 0;
      for (let i = 1; i < N; i++) if (delta[t - 1][i] * m.A[i][j] > delta[t - 1][best] * m.A[best][j]) best = i;
      row.push(delta[t - 1][best] * m.A[best][j] * m.B[j][O[t]]);
      bp.push(best);
    }
    delta.push(row);
    back.push(bp);
  }
  const last = delta.at(-1)!;
  let s = last.indexOf(Math.max(...last));
  const path = [s];
  for (let t = back.length - 1; t >= 0; t--) { s = back[t][s]; path.unshift(s); }
  return { delta, back, path, prob: Math.max(...last) };
}

/** Brute-force P(O) and best path by enumerating all N^T state sequences (for tests). */
export function bruteForce(m: Hmm, O: number[]): { p: number; best: number[]; bestP: number } {
  const N = m.pi.length;
  let p = 0, bestP = -1, best: number[] = [];
  const rec = (seq: number[]) => {
    if (seq.length === O.length) {
      let q = m.pi[seq[0]] * m.B[seq[0]][O[0]];
      for (let t = 1; t < O.length; t++) q *= m.A[seq[t - 1]][seq[t]] * m.B[seq[t]][O[t]];
      p += q;
      if (q > bestP) { bestP = q; best = seq.slice(); }
      return;
    }
    for (let i = 0; i < N; i++) rec([...seq, i]);
  };
  rec([]);
  return { p, best, bestP };
}

/** Sample a state and observation sequence of length T. */
export function sampleHmm(m: Hmm, T: number, seed: number): { S: number[]; O: number[] } {
  const rand = rng(seed);
  const draw = (p: number[]) => { let u = rand(), k = 0; while (k < p.length - 1 && u >= p[k]) { u -= p[k]; k++; } return k; };
  const S = [draw(m.pi)];
  const O = [draw(m.B[S[0]])];
  for (let t = 1; t < T; t++) { S.push(draw(m.A[S[t - 1]])); O.push(draw(m.B[S[t]])); }
  return { S, O };
}

/** Scaled forward–backward: posteriors γ, pairwise ξ sums and log-likelihood (long sequences). */
function scaledFB(m: Hmm, O: number[]) {
  const N = m.pi.length, T = O.length;
  const al: number[][] = [];
  const c: number[] = [];
  for (let t = 0; t < T; t++) {
    const row = t === 0 ? m.pi.map((p, i) => p * m.B[i][O[0]])
      : Array.from({ length: N }, (_, j) => al[t - 1].reduce((s, a, i) => s + a * m.A[i][j], 0) * m.B[j][O[t]]);
    const z = row.reduce((a, b) => a + b, 0);
    c.push(z);
    al.push(row.map((v) => v / z));
  }
  const be: number[][] = new Array(T);
  be[T - 1] = new Array(N).fill(1);
  for (let t = T - 2; t >= 0; t--) be[t] = Array.from({ length: N }, (_, i) => be[t + 1].reduce((s, b, j) => s + m.A[i][j] * m.B[j][O[t + 1]] * b, 0) / c[t + 1]);
  const gamma = al.map((a, t) => a.map((v, i) => v * be[t][i]));
  const xi = Array.from({ length: N }, () => new Array(N).fill(0));
  for (let t = 0; t < T - 1; t++) for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) xi[i][j] += (al[t][i] * m.A[i][j] * m.B[j][O[t + 1]] * be[t + 1][j]) / c[t + 1];
  return { gamma, xi, loglik: c.reduce((s, z) => s + Math.log(z), 0) };
}

/** Baum–Welch (EM) from `init`; returns every iterate's parameters and log-likelihood. */
export function baumWelch(O: number[], init: Hmm, iters = 30, M = 2): { models: Hmm[]; loglik: number[] } {
  let m = init;
  const models: Hmm[] = [];
  const loglik: number[] = [];
  for (let k = 0; k <= iters; k++) {
    const { gamma, xi, loglik: ll } = scaledFB(m, O);
    models.push(m);
    loglik.push(ll);
    const N = m.pi.length;
    const occ = Array.from({ length: N }, (_, i) => gamma.slice(0, -1).reduce((s, g) => s + g[i], 0));
    const occAll = Array.from({ length: N }, (_, i) => gamma.reduce((s, g) => s + g[i], 0));
    m = {
      pi: gamma[0].slice(),
      A: xi.map((row, i) => row.map((v) => v / occ[i])),
      B: Array.from({ length: N }, (_, i) => Array.from({ length: M }, (_, o) => gamma.reduce((s, g, t) => s + (O[t] === o ? g[i] : 0), 0) / occAll[i])),
    };
  }
  return { models, loglik };
}

/* ------------------------------------------------------------------ linear-chain CRF */

/**
 * Log partition function of a linear-chain CRF with unary scores U[t][y] and transition scores
 * Tr[y][y']: log Σ_y exp(Σ_t U[t][y_t] + Σ_t Tr[y_{t−1}][y_t]), by the forward recursion in log space.
 */
export function crfLogZ(U: number[][], Tr: number[][]): number {
  const lse = (v: number[]) => { const m = Math.max(...v); return m + Math.log(v.reduce((s, x) => s + Math.exp(x - m), 0)); };
  let a = U[0].slice();
  for (let t = 1; t < U.length; t++) a = U[t].map((u, j) => u + lse(a.map((v, i) => v + Tr[i][j])));
  return lse(a);
}

export function crfBrute(U: number[][], Tr: number[][]): number {
  const L = U[0].length;
  let z = 0;
  const rec = (y: number[]) => {
    if (y.length === U.length) {
      let s = U[0][y[0]];
      for (let t = 1; t < y.length; t++) s += U[t][y[t]] + Tr[y[t - 1]][y[t]];
      z += Math.exp(s);
      return;
    }
    for (let k = 0; k < L; k++) rec([...y, k]);
  };
  rec([]);
  return Math.log(z);
}

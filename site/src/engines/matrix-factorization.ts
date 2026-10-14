// Chapter 12 (matrix factorization): PCA in 2-D, a symmetric Jacobi eigen-solver, the SVD of a
// small procedural image and a synthetic multi-marker panel for scree plots. Pure maths.
import { gaussian, r6, rng } from "./data";

/** Two strongly correlated markers (centred), for the rotating-axis widget. */
export function correlated(n = 60, seed = 3): [number, number][] {
  const rand = rng(seed);
  const pts = Array.from({ length: n }, () => {
    const a = 1.4 * gaussian(rand), b = 0.45 * gaussian(rand);
    const t = 0.55; // principal direction ≈ 31.5°
    return [a * Math.cos(t) - b * Math.sin(t), a * Math.sin(t) + b * Math.cos(t)] as [number, number];
  });
  const m = [0, 1].map((j) => pts.reduce((s, p) => s + p[j], 0) / n);
  return pts.map((p) => [r6(p[0] - m[0]), r6(p[1] - m[1])]);
}

/** Projected variance and residual variance (per point) of centred data on the unit direction at angle t. */
export function projStats(P: [number, number][], t: number): { proj: number; resid: number; total: number } {
  const u = [Math.cos(t), Math.sin(t)];
  let proj = 0, total = 0;
  for (const p of P) {
    const z = p[0] * u[0] + p[1] * u[1];
    proj += z * z;
    total += p[0] * p[0] + p[1] * p[1];
  }
  const n = P.length;
  return { proj: proj / n, resid: (total - proj) / n, total: total / n };
}

/** Angle of the first principal direction of centred 2-D data. */
export function principalAngle(P: [number, number][]): number {
  const n = P.length;
  const a = P.reduce((s, p) => s + p[0] * p[0], 0) / n, b = P.reduce((s, p) => s + p[0] * p[1], 0) / n, c = P.reduce((s, p) => s + p[1] * p[1], 0) / n;
  return 0.5 * Math.atan2(2 * b, a - c);
}

/** Cyclic Jacobi eigen-decomposition of a symmetric matrix: eigenvalues descending, eigenvectors as columns. */
export function jacobiEig(S0: number[][], sweeps = 30): { values: number[]; vectors: number[][] } {
  const n = S0.length;
  const S = S0.map((r) => r.slice());
  const V: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sw = 0; sw < sweeps; sw++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += S[p][q] ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(S[p][q]) < 1e-300) continue;
      const th = 0.5 * Math.atan2(2 * S[p][q], S[q][q] - S[p][p]);
      const c = Math.cos(th), s = Math.sin(th);
      for (let k = 0; k < n; k++) {
        const skp = S[k][p], skq = S[k][q];
        S[k][p] = c * skp - s * skq;
        S[k][q] = s * skp + c * skq;
      }
      for (let k = 0; k < n; k++) {
        const spk = S[p][k], sqk = S[q][k];
        S[p][k] = c * spk - s * sqk;
        S[q][k] = s * spk + c * sqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = V[k][p], vkq = V[k][q];
        V[k][p] = c * vkp - s * vkq;
        V[k][q] = s * vkp + c * vkq;
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => S[j][j] - S[i][i]);
  return { values: order.map((i) => S[i][i]), vectors: V.map((row) => order.map((i) => row[i])) };
}

/** SVD of an m×n matrix via the eigen-decomposition of AᵀA. */
export function svd(A: number[][]): { U: number[][]; s: number[]; V: number[][] } {
  const m = A.length, n = A[0].length;
  const AtA = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => A.reduce((t, row) => t + row[i] * row[j], 0)));
  const { values, vectors: V } = jacobiEig(AtA);
  const s = values.map((v) => Math.sqrt(Math.max(0, v)));
  const U = Array.from({ length: m }, (_, i) => s.map((sk, k) => (sk > 1e-12 ? A[i].reduce((t, a, j) => t + a * V[j][k], 0) / sk : 0)));
  return { U, s, V };
}

/** Rank-k reconstruction Σ_{j<k} σ_j u_j v_jᵀ. */
export function lowRank(d: { U: number[][]; s: number[]; V: number[][] }, k: number): number[][] {
  return d.U.map((urow) => d.V.map((vrow) => { let t = 0; for (let j = 0; j < k; j++) t += d.s[j] * urow[j] * vrow[j]; return t; }));
}

/** A procedural 40×56 "scan": smooth shading, a window grid, a disc and a diagonal band (no external image). */
export function scanImage(h = 40, w = 56): number[][] {
  return Array.from({ length: h }, (_, i) => Array.from({ length: w }, (_, j) => {
    const x = j / (w - 1), y = i / (h - 1);
    let v = 0.35 + 0.25 * Math.sin(3 * x + 1) * Math.cos(2 * y);
    if (x > 0.08 && x < 0.42 && y > 0.12 && y < 0.62 && ((Math.floor(x * 20) + Math.floor(y * 14)) % 2 === 0)) v += 0.35;
    if ((x - 0.72) ** 2 + ((y - 0.38) * 1.4) ** 2 < 0.035) v = 0.95;
    if (Math.abs(y - (1.1 - x)) < 0.05 && y > 0.6) v = 0.1;
    return Math.max(0, Math.min(1, v));
  }));
}

/** A synthetic 8-marker panel driven by 2 latent factors plus noise, with 3 subtypes. */
export function panel(n = 120, seed = 6): { X: number[][]; g: number[] } {
  const rand = rng(seed);
  const L = [[0.9, 0.1], [0.8, 0.3], [0.7, -0.2], [0.1, 0.9], [-0.2, 0.8], [0.3, 0.7], [0.5, 0.5], [0.6, -0.4]];
  const C = [[-1.5, -0.5], [1.4, -0.9], [0.2, 1.6]];
  const X: number[][] = [];
  const g: number[] = [];
  for (let i = 0; i < n; i++) {
    const k = i % 3;
    const z = [C[k][0] + 0.6 * gaussian(rand), C[k][1] + 0.6 * gaussian(rand)];
    X.push(L.map((l) => r6(l[0] * z[0] + l[1] * z[1] + 0.35 * gaussian(rand))));
    g.push(k);
  }
  return { X, g };
}

/** PCA by eigen-decomposition of the covariance: eigenvalues and 2-D scores. */
export function pca(X: number[][]): { values: number[]; scores: [number, number][] } {
  const n = X.length, d = X[0].length;
  const m = Array.from({ length: d }, (_, j) => X.reduce((s, x) => s + x[j], 0) / n);
  const C = X.map((x) => x.map((v, j) => v - m[j]));
  const S = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => C.reduce((s, x) => s + x[i] * x[j], 0) / n));
  const { values, vectors } = jacobiEig(S);
  return { values, scores: C.map((x) => [0, 1].map((k) => x.reduce((s, v, j) => s + v * vectors[j][k], 0)) as [number, number]) };
}

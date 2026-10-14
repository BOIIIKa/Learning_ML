// Chapter 14 (From Shallow to Deep Networks): ReLU networks as piecewise-linear functions.
// Shallow fits with hinge units, linear regions of a 2-D layer, and depth as folding (tent maps).
import { gaussian, r6, rng } from "./data";
import { solve } from "./regression";
import type { Net } from "./mlp";

export const relu = (z: number) => Math.max(0, z);

/* ------------------------------------------------------------------ shallow fits in 1-D */

/** Hinge positions for D hidden units relu(x − k_d), evenly spread over [lo, hi). */
export const knotsFor = (D: number, lo = 0, hi = 10) => Array.from({ length: D }, (_, d) => r6(lo + ((hi - lo) * d) / D));

/** Least-squares output layer of a shallow net with fixed hinges: f(x) = φ₀ + Σ φ_d relu(x − k_d). */
export function shallowFit(xs: number[], ys: number[], knots: number[], ridge = 1e-6) {
  const feats = (x: number) => [1, ...knots.map((k) => relu(x - k))];
  const p = knots.length + 1;
  const A = Array.from({ length: p }, () => new Array(p).fill(0));
  const rhs = new Array(p).fill(0);
  xs.forEach((x, n) => {
    const f = feats(x);
    for (let i = 0; i < p; i++) {
      rhs[i] += f[i] * ys[n];
      for (let j = 0; j < p; j++) A[i][j] += f[i] * f[j];
    }
  });
  for (let i = 1; i < p; i++) A[i][i] += ridge;
  const phi = solve(A, rhs);
  const f = (x: number) => feats(x).reduce((s, v, i) => s + v * phi[i], 0);
  return { phi, knots, f, unit: (d: number, x: number) => phi[d + 1] * relu(x - knots[d]) };
}

/** Number of linear pieces of f on [lo, hi]: runs of equal slope on a fine grid. A single grid step
 * that straddles a kink has an in-between slope; such one-step runs are not pieces. */
export function countPieces(f: (x: number) => number, lo: number, hi: number, n = 4000): number {
  const h = (hi - lo) / n;
  const slopes = Array.from({ length: n }, (_, i) => (f(lo + (i + 1) * h) - f(lo + i * h)) / h);
  let pieces = 0;
  let run = 1;
  for (let i = 1; i <= n; i++) {
    if (i < n && Math.abs(slopes[i] - slopes[i - 1]) <= 1e-7 * (1 + Math.abs(slopes[i - 1]))) { run++; continue; }
    if (run >= 2) pieces++;
    run = 1;
  }
  return pieces;
}

/* ------------------------------------------------------------------ regions of a 2-D ReLU layer */

export interface Unit { w: [number, number]; b: number }

/** D random hidden units on a 2-D input: each unit's boundary w·x + b = 0 is a line near the origin. */
export function randomUnits(D: number, seed: number): Unit[] {
  const rand = rng(seed);
  return Array.from({ length: D }, () => {
    const a = Math.PI * 2 * rand();
    return { w: [r6(Math.cos(a)), r6(Math.sin(a))], b: r6(0.9 * (2 * rand() - 1)) };
  });
}

/** Most regions D lines in general position can make in the plane: 1 + D + C(D, 2). */
export const maxRegions2d = (D: number) => 1 + D + (D * (D - 1)) / 2;

/** Exact region count inside the square [−R, R]²: 1 + (lines crossing it) + (crossings inside it). */
export function regionsInWindow(units: Unit[], R: number): number {
  const inside = (x: number, y: number) => Math.abs(x) <= R && Math.abs(y) <= R;
  let lines = 0;
  let points = 0;
  units.forEach((u, i) => {
    // The line meets the square iff the corners are not all on one side.
    const s = [[-R, -R], [R, -R], [R, R], [-R, R]].map(([x, y]) => Math.sign(u.w[0] * x + u.w[1] * y + u.b));
    if (s.some((v) => v > 0) && s.some((v) => v < 0)) lines++;
    for (let j = 0; j < i; j++) {
      const v = units[j];
      const det = u.w[0] * v.w[1] - u.w[1] * v.w[0];
      if (Math.abs(det) < 1e-12) continue;
      const x = (-u.b * v.w[1] + v.b * u.w[1]) / det;
      const y = (-u.w[0] * v.b + v.w[0] * u.b) / det;
      if (inside(x, y)) points++;
    }
  });
  return 1 + lines + points;
}

/** Activation pattern of each grid cell (for colouring) and the number of distinct patterns seen. */
export function patternGrid(units: Unit[], R: number, n: number) {
  const ids = new Map<string, number>();
  const cells: number[] = [];
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const x = -R + (2 * R * (c + 0.5)) / n, y = R - (2 * R * (r + 0.5)) / n;
    const key = units.map((u) => (u.w[0] * x + u.w[1] * y + u.b > 0 ? 1 : 0)).join("");
    if (!ids.has(key)) ids.set(key, ids.size);
    cells.push(ids.get(key)!);
  }
  return { cells, distinct: ids.size };
}

/* ------------------------------------------------------------------ depth as folding */

/** The tent map on [0, 1], built from two ReLU units: t(x) = 2 relu(x) − 4 relu(x − ½). */
export const tent = (x: number) => 2 * relu(x) - 4 * relu(x - 0.5);
export const tentPow = (L: number, x: number) => { let z = x; for (let l = 0; l < L; l++) z = tent(z); return z; };

/** t∘…∘t (L times) as an L-hidden-layer ReLU net, 2 units per layer (for mlp.forward). */
export function deepTentNet(L: number): Net {
  const net: Net = [{ W: [[1], [1]], b: [0, -0.5] }];
  for (let l = 1; l < L; l++) net.push({ W: [[2, -4], [2, -4]], b: [0, -0.5] });
  net.push({ W: [[2, -4]], b: [0] });
  return net;
}

/** Parameters of the deep tent net (6L + 1) and of a shallow net with D units (3D + 1). */
export const deepTentParams = (L: number) => 6 * L + 1;
export const shallowParams = (D: number) => 3 * D + 1;

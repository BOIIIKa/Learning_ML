// Chapter 06: k-nearest neighbours, kd-trees, the curse of dimensionality and spline basis expansions.
import { gaussian, r6, rng } from "./data";

export type Metric = "l1" | "l2" | "linf";

/** Minkowski distance between two points for p = 1, 2 or ∞. */
export function dist(a: number[], b: number[], metric: Metric): number {
  let s = 0;
  for (let j = 0; j < a.length; j++) {
    const d = Math.abs(a[j] - b[j]);
    if (metric === "l1") s += d;
    else if (metric === "l2") s += d * d;
    else s = Math.max(s, d);
  }
  return metric === "l2" ? Math.sqrt(s) : s;
}

/** Indices of the k nearest training points to q (ties broken by index), and their distances. */
export function kNearest(X: number[][], q: number[], k: number, metric: Metric): { idx: number[]; d: number[] } {
  const all = X.map((p, i) => [dist(p, q, metric), i] as [number, number]);
  all.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const top = all.slice(0, k);
  return { idx: top.map((t) => t[1]), d: top.map((t) => t[0]) };
}

/** Mean of the ±1 labels of the k nearest neighbours: > 0 votes for +1. */
export function knnScore(X: number[][], y: number[], q: number[], k: number, metric: Metric): number {
  const { idx } = kNearest(X, q, k, metric);
  return idx.reduce((s, i) => s + y[i], 0) / idx.length;
}

/** Majority vote (k odd ⇒ no ties for ±1 labels). */
export const knnPredict = (X: number[][], y: number[], q: number[], k: number, metric: Metric) =>
  knnScore(X, y, q, k, metric) > 0 ? 1 : -1;

export function knnError(Xtr: number[][], ytr: number[], X: number[][], y: number[], k: number, metric: Metric): number {
  return X.filter((q, i) => knnPredict(Xtr, ytr, q, k, metric) !== y[i]).length / X.length;
}

// ---------------------------------------------------------------- kd-tree

export interface KdNode {
  id: number;
  /** Index of the point stored at this node. */
  point: number;
  axis: 0 | 1;
  depth: number;
  left: number | null;
  right: number | null;
  parent: number | null;
  /** The cell this node splits: [x0, x1, y0, y1]. */
  rect: [number, number, number, number];
}

/**
 * Build a 2-D kd-tree: split on axis depth mod 2 at the median (the element at index ⌊m/2⌋ after
 * sorting), points smaller than the median go left. Node ids are assigned in creation (pre-)order.
 */
export function buildKd(P: number[][], rect: [number, number, number, number]): KdNode[] {
  const nodes: KdNode[] = [];
  const rec = (idx: number[], depth: number, r: [number, number, number, number], parent: number | null): number | null => {
    if (idx.length === 0) return null;
    const axis = (depth % 2) as 0 | 1;
    const sorted = [...idx].sort((a, b) => P[a][axis] - P[b][axis] || a - b);
    const m = Math.floor(sorted.length / 2);
    const id = nodes.length;
    const node: KdNode = { id, point: sorted[m], axis, depth, left: null, right: null, parent, rect: r };
    nodes.push(node);
    const s = P[sorted[m]][axis];
    const rl: [number, number, number, number] = axis === 0 ? [r[0], s, r[2], r[3]] : [r[0], r[1], r[2], s];
    const rr: [number, number, number, number] = axis === 0 ? [s, r[1], r[2], r[3]] : [r[0], r[1], s, r[3]];
    node.left = rec(sorted.slice(0, m), depth + 1, rl, id);
    node.right = rec(sorted.slice(m + 1), depth + 1, rr, id);
    return id;
  };
  rec(P.map((_, i) => i), 0, rect, null);
  return nodes;
}

/** Node ids in breadth-first order: the order in which the plane is cut "layer by layer". */
export function bfsOrder(nodes: KdNode[]): number[] {
  return [...nodes].sort((a, b) => a.depth - b.depth || a.id - b.id).map((n) => n.id);
}

export type KdEventKind = "visit" | "update" | "cross" | "prune" | "done";

export interface KdEvent {
  kind: KdEventKind;
  node: number;
  /** Index (into P) of the best point so far, and its distance. */
  best: number;
  r: number;
  visited: number[];
  pruned: number[];
  /** |q_axis − split| for cross / prune events. */
  gap?: number;
}

/**
 * Exact nearest-neighbour search (Euclidean) with branch-and-bound. Go first into the child on the
 * query's side; on the way back, enter the other child only if the splitting line is closer than the
 * current best distance, otherwise prune that whole subtree. Returns every step for replay.
 */
export function kdSearch(nodes: KdNode[], P: number[][], q: number[]): { best: number; r: number; events: KdEvent[] } {
  let best = -1;
  let r = Infinity;
  const visited: number[] = [];
  const pruned: number[] = [];
  const events: KdEvent[] = [];
  const emit = (kind: KdEventKind, node: number, gap?: number) =>
    events.push({ kind, node, best, r, visited: [...visited], pruned: [...pruned], gap });
  const rec = (id: number | null) => {
    if (id === null) return;
    const n = nodes[id];
    visited.push(id);
    emit("visit", id);
    const d = dist(P[n.point], q, "l2");
    if (d < r) {
      r = d;
      best = n.point;
      emit("update", id);
    }
    const diff = q[n.axis] - P[n.point][n.axis];
    const [near, far] = diff < 0 ? [n.left, n.right] : [n.right, n.left];
    rec(near);
    if (far === null) return;
    if (Math.abs(diff) < r) {
      emit("cross", id, Math.abs(diff));
      rec(far);
    } else {
      pruned.push(far);
      emit("prune", far, Math.abs(diff));
    }
  };
  rec(0);
  emit("done", 0);
  return { best, r, events };
}

/** All node ids in the subtree rooted at `id`. */
export function subtree(nodes: KdNode[], id: number): number[] {
  const out: number[] = [];
  const stack = [id];
  while (stack.length) {
    const n = nodes[stack.pop()!];
    out.push(n.id);
    if (n.left !== null) stack.push(n.left);
    if (n.right !== null) stack.push(n.right);
  }
  return out;
}

/** Deterministic points in [lo, hi]², rounded to 0.1 so they read well. */
export function kdPoints(n = 15, seed = 4, lo = 0.5, hi = 9.5): number[][] {
  const rand = rng(seed);
  return Array.from({ length: n }, () => [Math.round((lo + (hi - lo) * rand()) * 10) / 10, Math.round((lo + (hi - lo) * rand()) * 10) / 10]);
}

// ---------------------------------------------------------------- curse of dimensionality

/** Distances from a random query to n uniform points in [0,1]^d. */
export function distancesInCube(d: number, n = 200, seed = 1): number[] {
  const rand = rng(seed * 1000 + d);
  const q = Array.from({ length: d }, () => rand());
  return Array.from({ length: n }, () => {
    let s = 0;
    for (let j = 0; j < d; j++) {
      const v = rand() - q[j];
      s += v * v;
    }
    return Math.sqrt(s);
  });
}

/** Nearest / farthest distance ratio, averaged over a few queries for a steadier curve. */
export function minMaxRatio(d: number, n = 200, reps = 4): number {
  let s = 0;
  for (let t = 1; t <= reps; t++) {
    const ds = distancesInCube(d, n, t);
    s += Math.min(...ds) / Math.max(...ds);
  }
  return r6(s / reps);
}

/** Fraction of a d-ball's volume within eps of its surface: 1 − (1 − ε)^d. */
export const shellFraction = (d: number, eps: number) => 1 - (1 - eps) ** d;

/** Edge of the sub-cube that holds a fraction p of uniform data in [0,1]^d. */
export const edgeFor = (p: number, d: number) => p ** (1 / d);

// ---------------------------------------------------------------- basis expansions

/** Delivery time (minutes) against distance (km): a smooth non-linear trend plus noise. */
export const deliveryTrend = (x: number) => 14 + 3.2 * x + 6 * Math.sin(1.1 * x);

export function deliveryData(n = 60, seed = 3, noise = 3): { xs: number[]; ys: number[] } {
  const rand = rng(seed);
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = 0.3 + 9.4 * rand();
    xs.push(r6(x));
    ys.push(r6(deliveryTrend(x) + noise * gaussian(rand)));
  }
  return { xs, ys };
}

export type BasisKind = "poly" | "step" | "linear" | "cubic";

/** Equally spaced interior knots on (lo, hi). */
export const evenKnots = (K: number, lo = 0, hi = 10) => Array.from({ length: K }, (_, j) => lo + ((hi - lo) * (j + 1)) / (K + 1));

/**
 * Basis row for x (x is first scaled to [0,1] by `scale`). "poly": 1, x, …, x^degree.
 * Splines of degree M ∈ {0, 1, 3}: the truncated power basis 1, x, …, x^M, (x − ξ_k)_+^M.
 * Returns the row and a mask marking the knot terms (the ones the roughness penalty shrinks).
 */
export function basisRow(x: number, kind: BasisKind, degree: number, knots: number[], scale = 10): { row: number[]; penal: boolean[] } {
  const u = x / scale;
  const M = kind === "poly" ? degree : kind === "step" ? 0 : kind === "linear" ? 1 : 3;
  const row: number[] = [];
  const penal: boolean[] = [];
  for (let m = 0; m <= M; m++) {
    row.push(u ** m);
    penal.push(false);
  }
  if (kind !== "poly") {
    for (const k of knots) {
      const t = u - k / scale;
      row.push(t > 0 ? (M === 0 ? 1 : t ** M) : 0);
      penal.push(true);
    }
  }
  return { row, penal };
}

/** Solve A x = b (A square) by Gaussian elimination with partial pivoting. */
export function solve(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const piv = M[c][c] || 1e-300;
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / piv;
      if (f !== 0) for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / (M[r][r] || 1e-300);
  }
  return x;
}

export interface BasisFit {
  coef: number[];
  predict: (x: number) => number;
  /** Effective degrees of freedom tr(S_λ) = tr[(BᵀB + λD)⁻¹ BᵀB]. */
  df: number;
  p: number;
}

/**
 * Penalized least squares on a basis expansion: minimise ‖y − Bβ‖² + λ Σ_{knot terms} β_k².
 * A tiny relative ridge (1e-12 · max diag) on every term keeps the normal equations well posed.
 */
export function fitBasis(xs: number[], ys: number[], kind: BasisKind, degree: number, knots: number[], lambda = 0): BasisFit {
  const rows = xs.map((x) => basisRow(x, kind, degree, knots));
  const p = rows[0].row.length;
  const penal = rows[0].penal;
  const G = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => rows.reduce((s, r) => s + r.row[i] * r.row[j], 0)));
  const eps = 1e-12 * Math.max(...G.map((r, i) => r[i]));
  const A = G.map((r, i) => r.map((v, j) => v + (i === j ? eps + (penal[i] ? lambda : 0) : 0)));
  const c = Array.from({ length: p }, (_, i) => rows.reduce((s, r, k) => s + r.row[i] * ys[k], 0));
  const coef = solve(A, c);
  // tr(A⁻¹ G): solve A z = G e_j for each column j.
  let df = 0;
  for (let j = 0; j < p; j++) df += solve(A, G.map((r) => r[j]))[j];
  const predict = (x: number) => basisRow(x, kind, degree, knots).row.reduce((s, v, i) => s + v * coef[i], 0);
  return { coef, predict, df, p };
}

/** kNN regression in 1-D: the average y of the k nearest x's. */
export function knnRegress(xs: number[], ys: number[], x: number, k: number): number {
  const { idx } = kNearest(xs.map((v) => [v]), [x], k, "l2");
  return idx.reduce((s, i) => s + ys[i], 0) / idx.length;
}

export const rmse = (f: (x: number) => number, xs: number[], ys: number[]) =>
  Math.sqrt(xs.reduce((s, x, i) => s + (f(x) - ys[i]) ** 2, 0) / xs.length);

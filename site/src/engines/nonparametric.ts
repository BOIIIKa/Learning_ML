// Chapter 06 (non-parametric methods): kNN, kd-trees, regression splines and the curse of
// dimensionality. Pure, deterministic maths for the widgets. No DOM, no React.
import { rng } from "./data";
import { solve } from "./regression";

export type Metric = "l1" | "l2" | "linf";
export const dist = (a: number[], b: number[], m: Metric) => {
  if (m === "l1") return a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0);
  if (m === "linf") return a.reduce((s, v, i) => Math.max(s, Math.abs(v - b[i])), 0);
  return Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0));
};

/** Indices of the k nearest training points (ties broken by index). */
export function nearest(X: number[][], q: number[], k: number, m: Metric): number[] {
  return X.map((x, i) => [dist(x, q, m), i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).slice(0, k).map((p) => p[1]);
}

/** kNN vote share in [−1, 1]: mean label of the k nearest neighbours (labels ±1). Positive ⇒ class +1. */
export const knnScore = (X: number[][], y: number[], q: number[], k: number, m: Metric) =>
  nearest(X, q, k, m).reduce((s, i) => s + y[i], 0) / k;

export const knnError = (X: number[][], y: number[], T: number[][], Ty: number[], k: number, m: Metric) =>
  T.reduce((s, q, i) => s + (knnScore(X, y, q, k, m) * Ty[i] > 0 ? 0 : 1), 0) / T.length;

/* ------------------------------------------------------------------ kd-tree */

export interface KdNode {
  i: number; // index of the point stored at this node
  axis: 0 | 1;
  left: KdNode | null;
  right: KdNode | null;
  depth: number;
  /** The cell of the plane this node splits: [x0, x1, y0, y1]. */
  box: [number, number, number, number];
}

/** Build a 2-D kd-tree: split on x, y, x, … at the median (the point at index ⌊n/2⌋ after sorting). */
export function buildKd(P: number[][], box: [number, number, number, number], idx = P.map((_, i) => i), depth = 0): KdNode | null {
  if (!idx.length) return null;
  const axis = (depth % 2) as 0 | 1;
  const sorted = [...idx].sort((a, b) => P[a][axis] - P[b][axis] || a - b);
  const mid = Math.floor(sorted.length / 2);
  const i = sorted[mid];
  const v = P[i][axis];
  const lbox: typeof box = axis === 0 ? [box[0], v, box[2], box[3]] : [box[0], box[1], box[2], v];
  const rbox: typeof box = axis === 0 ? [v, box[1], box[2], box[3]] : [box[0], box[1], v, box[3]];
  return { i, axis, depth, box, left: buildKd(P, lbox, sorted.slice(0, mid), depth + 1), right: buildKd(P, rbox, sorted.slice(mid + 1), depth + 1) };
}

export interface KdEvent {
  kind: "visit" | "prune";
  i: number; // node point
  best: number; // index of the best point after this event
  bestDist: number;
}

/** Nearest-neighbour search with backtracking; records every visited and pruned node. */
export function searchKd(root: KdNode | null, P: number[][], q: number[]): { best: number; events: KdEvent[] } {
  let best = -1;
  let bestDist = Infinity;
  const events: KdEvent[] = [];
  const go = (node: KdNode | null) => {
    if (!node) return;
    const d = dist(P[node.i], q, "l2");
    if (d < bestDist) { bestDist = d; best = node.i; }
    events.push({ kind: "visit", i: node.i, best, bestDist });
    const diff = q[node.axis] - P[node.i][node.axis];
    const [near, far] = diff < 0 ? [node.left, node.right] : [node.right, node.left];
    go(near);
    // The far side can only help if the ball around q crosses the splitting line.
    if (Math.abs(diff) < bestDist) go(far);
    else if (far) events.push({ kind: "prune", i: far.i, best, bestDist });
  };
  go(root);
  return { best, events };
}

export const kdNodes = (root: KdNode | null): KdNode[] => (root ? [root, ...kdNodes(root.left), ...kdNodes(root.right)] : []);

/* ------------------------------------------------------------------ regression splines */

/** Cubic-spline basis with interior knots ξ: 1, x, x², x³, (x − ξ_k)³₊ (x already rescaled to [0, 1]). */
export const splineRow = (u: number, knots: number[]) => [1, u, u * u, u ** 3, ...knots.map((k) => Math.max(0, u - k) ** 3)];

/** Least-squares cubic regression spline with K equally spaced interior knots on [lo, hi]. */
export function fitSpline(xs: number[], ys: number[], K: number, lo = 0, hi = 10): { f: (x: number) => number; knots: number[]; df: number } {
  const knotsU = Array.from({ length: K }, (_, k) => (k + 1) / (K + 1));
  const toU = (x: number) => (x - lo) / (hi - lo);
  const Phi = xs.map((x) => splineRow(toU(x), knotsU));
  const p = K + 4;
  const A = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => Phi.reduce((s, r) => s + r[i] * r[j], 0) + (i === j ? 1e-9 : 0)));
  const b = Array.from({ length: p }, (_, i) => Phi.reduce((s, r, n) => s + r[i] * ys[n], 0));
  const w = solve(A, b);
  return { f: (x) => splineRow(toU(x), knotsU).reduce((s, v, i) => s + v * w[i], 0), knots: knotsU.map((u) => lo + u * (hi - lo)), df: p };
}

/* ------------------------------------------------------------------ the curse of dimensionality */

/** Fraction of a d-ball's volume within the outer shell of relative thickness ε: 1 − (1 − ε)^d. */
export const shellFraction = (d: number, eps: number) => 1 - (1 - eps) ** d;
/** Edge of a sub-cube that holds a fraction r of uniform data in [0, 1]^d. */
export const edgeFor = (r: number, d: number) => r ** (1 / d);

/** Mean ratio (nearest distance) / (farthest distance) from random queries to n uniform points in [0, 1]^d. */
export function contrast(d: number, n = 400, queries = 8, seed = 3): number {
  const rand = rng(seed + d);
  const pts = Array.from({ length: n }, () => Array.from({ length: d }, () => rand()));
  let s = 0;
  for (let q = 0; q < queries; q++) {
    const x = Array.from({ length: d }, () => rand());
    let lo = Infinity, hi = 0;
    for (const p of pts) {
      const v = dist(p, x, "l2");
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    s += lo / hi;
  }
  return s / queries;
}

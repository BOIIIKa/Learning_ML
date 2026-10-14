// CART classification trees (Gini), bagging / random forests and 1-D gradient boosting
// for the chapter 08 visuals. Labels are ±1.
import { rng } from "./data";

export type Node =
  | { leaf: true; p: number; n: number }
  | { leaf: false; feature: number; threshold: number; gain: number; n: number; p: number; left: Node; right: Node };

export const gini = (p: number) => 2 * p * (1 - p);
export const entropy = (p: number) => (p <= 0 || p >= 1 ? 0 : -p * Math.log2(p) - (1 - p) * Math.log2(1 - p));

const posRate = (y: number[], idx: number[]) => (idx.length ? idx.filter((i) => y[i] > 0).length / idx.length : 0.5);

export interface Split {
  feature: number;
  threshold: number;
  gain: number;
}

/** Every candidate split (midpoints between distinct values) with its Gini decrease. */
export function candidateSplits(X: number[][], y: number[], idx: number[], features = [0, 1]): Split[] {
  const parent = gini(posRate(y, idx));
  const out: Split[] = [];
  for (const f of features) {
    const sorted = [...idx].sort((a, b) => X[a][f] - X[b][f]);
    let leftPos = 0;
    const totalPos = sorted.filter((i) => y[i] > 0).length;
    for (let k = 0; k < sorted.length - 1; k++) {
      if (y[sorted[k]] > 0) leftPos++;
      const v = X[sorted[k]][f];
      const next = X[sorted[k + 1]][f];
      if (v === next) continue;
      const nl = k + 1;
      const nr = sorted.length - nl;
      const child = (nl / sorted.length) * gini(leftPos / nl) + (nr / sorted.length) * gini((totalPos - leftPos) / nr);
      out.push({ feature: f, threshold: (v + next) / 2, gain: parent - child });
    }
  }
  return out;
}

export function bestSplit(X: number[][], y: number[], idx: number[], features?: number[]): Split | null {
  let best: Split | null = null;
  for (const s of candidateSplits(X, y, idx, features)) if (!best || s.gain > best.gain + 1e-12) best = s;
  return best;
}

export interface TreeOptions {
  maxDepth?: number;
  minLeaf?: number;
  /** Features considered at each split (random forests pass a random subset). */
  pickFeatures?: () => number[];
}

export function growTree(X: number[][], y: number[], idx: number[] = X.map((_, i) => i), opts: TreeOptions = {}, depth = 0): Node {
  const { maxDepth = 3, minLeaf = 1, pickFeatures } = opts;
  const p = posRate(y, idx);
  if (depth >= maxDepth || p === 0 || p === 1 || idx.length < 2 * minLeaf) return { leaf: true, p, n: idx.length };
  const s = bestSplit(X, y, idx, pickFeatures?.());
  if (!s || s.gain <= 1e-12) return { leaf: true, p, n: idx.length };
  const L = idx.filter((i) => X[i][s.feature] <= s.threshold);
  const R = idx.filter((i) => X[i][s.feature] > s.threshold);
  if (L.length < minLeaf || R.length < minLeaf) return { leaf: true, p, n: idx.length };
  return {
    leaf: false, ...s, n: idx.length, p,
    left: growTree(X, y, L, opts, depth + 1),
    right: growTree(X, y, R, opts, depth + 1),
  };
}

/** Probability of the positive class. */
export function predictProba(node: Node, x: number[]): number {
  while (!node.leaf) node = x[node.feature] <= node.threshold ? node.left : node.right;
  return node.p;
}

export const accuracy = (node: Node, X: number[][], y: number[]) =>
  X.filter((x, i) => (predictProba(node, x) >= 0.5 ? 1 : -1) === y[i]).length / X.length;

export function depthOf(node: Node): number {
  return node.leaf ? 0 : 1 + Math.max(depthOf(node.left), depthOf(node.right));
}

export function leavesOf(node: Node): number {
  return node.leaf ? 1 : leavesOf(node.left) + leavesOf(node.right);
}

/** Bagged trees; `randomFeatures` turns bagging into a (2-D) random forest. */
export function forest(X: number[][], y: number[], nTrees: number, seed = 1, maxDepth = 8, randomFeatures = true): Node[] {
  const rand = rng(seed);
  const trees: Node[] = [];
  for (let t = 0; t < nTrees; t++) {
    const idx = X.map(() => Math.floor(rand() * X.length));
    const pickFeatures = randomFeatures ? () => [rand() < 0.5 ? 0 : 1] : undefined;
    trees.push(growTree(X, y, idx, { maxDepth, pickFeatures }));
  }
  return trees;
}

export const forestProba = (trees: Node[], x: number[]) => trees.reduce((s, t) => s + predictProba(t, x), 0) / trees.length;

// ---------- 1-D gradient boosting with regression stumps (squared loss) ----------

export interface Stump1D { threshold: number; left: number; right: number }

/** The least-squares stump for targets r over sorted xs. */
export function fitStump1D(xs: number[], r: number[]): Stump1D {
  const order = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b]);
  const total = r.reduce((s, v) => s + v, 0);
  let best: Stump1D & { sse: number } = { threshold: -Infinity, left: 0, right: total / r.length, sse: Infinity };
  let leftSum = 0;
  for (let k = 0; k < order.length - 1; k++) {
    leftSum += r[order[k]];
    const nl = k + 1;
    const nr = order.length - nl;
    if (xs[order[k]] === xs[order[k + 1]]) continue;
    // SSE = Σr² − (S_L²/n_L + S_R²/n_R); the Σr² term is shared, so maximize the bracket.
    const score = -(leftSum ** 2 / nl + (total - leftSum) ** 2 / nr);
    if (score < best.sse) best = { threshold: (xs[order[k]] + xs[order[k + 1]]) / 2, left: leftSum / nl, right: (total - leftSum) / nr, sse: score };
  }
  return { threshold: best.threshold, left: best.left, right: best.right };
}

export const stumpValue = (s: Stump1D, x: number) => (x <= s.threshold ? s.left : s.right);

/** F_m = F_{m-1} + ν · stump fitted to the residuals y − F_{m-1}. Returns every stage. */
export function gradientBoost1D(xs: number[], ys: number[], rounds: number, nu = 0.3) {
  const mean = ys.reduce((s, v) => s + v, 0) / ys.length;
  const stumps: Stump1D[] = [];
  const stages: { F: number[]; residual: number[]; stump: Stump1D | null; mse: number }[] = [];
  let F = ys.map(() => mean);
  const mse = (F: number[]) => F.reduce((s, f, i) => s + (ys[i] - f) ** 2, 0) / ys.length;
  stages.push({ F, residual: ys.map((v, i) => v - F[i]), stump: null, mse: mse(F) });
  for (let m = 0; m < rounds; m++) {
    const residual = ys.map((v, i) => v - F[i]);
    const s = fitStump1D(xs, residual);
    stumps.push(s);
    F = F.map((f, i) => f + nu * stumpValue(s, xs[i]));
    stages.push({ F, residual, stump: s, mse: mse(F) });
  }
  const predict = (x: number, upTo = stumps.length) => stumps.slice(0, upTo).reduce((f, s) => f + nu * stumpValue(s, x), mean);
  return { stages, predict, mean, nu };
}

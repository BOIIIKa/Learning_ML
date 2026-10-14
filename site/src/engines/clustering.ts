// Chapter 10 (clustering): Lloyd's k-means, k-means++ seeding, agglomerative clustering and
// silhouettes. Pure, deterministic maths for the widgets. No DOM, no React.
import { gaussian, r6, rng } from "./data";

export type P2 = [number, number];
const d2 = (a: number[], b: number[]) => a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0);

/** Three patient subgroups in two standardized markers (unlabelled for the algorithms). */
export function subgroups(n = 90, seed = 5): { X: P2[]; g: number[] } {
  const rand = rng(seed);
  const C: P2[] = [[-1.4, -0.6], [1.3, -0.9], [0.1, 1.4]];
  const S = [0.55, 0.5, 0.6];
  const X: P2[] = [];
  const g: number[] = [];
  for (let i = 0; i < n; i++) {
    const k = i % 3;
    X.push([r6(C[k][0] + S[k] * gaussian(rand)), r6(C[k][1] + S[k] * gaussian(rand))]);
    g.push(k);
  }
  return { X, g };
}

export const assign = (X: number[][], C: number[][]) => X.map((x) => C.reduce((b, c, k) => (d2(x, c) < d2(x, C[b]) ? k : b), 0));

export function update(X: number[][], lab: number[], C: number[][]): number[][] {
  return C.map((c, k) => {
    const members = X.filter((_, i) => lab[i] === k);
    if (!members.length) return c.slice(); // empty cluster: keep the old centre
    return c.map((_, j) => members.reduce((s, x) => s + x[j], 0) / members.length);
  });
}

export const wcss = (X: number[][], lab: number[], C: number[][]) => X.reduce((s, x, i) => s + d2(x, C[lab[i]]), 0);

/** Random seeding: k distinct data points chosen uniformly. */
export function seedRandom(X: number[][], k: number, seed: number): number[][] {
  const rand = rng(seed);
  const idx: number[] = [];
  while (idx.length < k) { const i = Math.floor(rand() * X.length); if (!idx.includes(i)) idx.push(i); }
  return idx.map((i) => X[i].slice());
}

/** k-means++: first centre uniform, each next one with probability ∝ D(x)², the squared distance to the nearest chosen centre. */
export function seedPlusPlus(X: number[][], k: number, seed: number): number[][] {
  const rand = rng(seed);
  const C = [X[Math.floor(rand() * X.length)].slice()];
  while (C.length < k) {
    const D = X.map((x) => Math.min(...C.map((c) => d2(x, c))));
    let u = rand() * D.reduce((a, b) => a + b, 0);
    let i = 0;
    while (i < X.length - 1 && u >= D[i]) { u -= D[i]; i++; }
    C.push(X[i].slice());
  }
  return C;
}

export interface LloydState { C: number[][]; lab: number[]; obj: number; phase: "init" | "assign" | "update" }

/** Lloyd's algorithm as half-steps: init, assign, update, assign, update, … until nothing changes. */
export function lloyd(X: number[][], C0: number[][], maxIter = 30): LloydState[] {
  let C = C0.map((c) => c.slice());
  let lab = assign(X, C);
  const out: LloydState[] = [{ C, lab, obj: wcss(X, lab, C), phase: "init" }];
  for (let t = 0; t < maxIter; t++) {
    const nl = assign(X, C);
    out.push({ C, lab: nl, obj: wcss(X, nl, C), phase: "assign" });
    const nc = update(X, nl, C);
    out.push({ C: nc, lab: nl, obj: wcss(X, nl, nc), phase: "update" });
    const same = nl.every((v, i) => v === lab[i]) && nc.every((c, k) => d2(c, C[k]) < 1e-18);
    lab = nl;
    C = nc;
    if (same) break;
  }
  return out;
}

/* ------------------------------------------------------------------ hierarchical clustering */

export type Linkage = "single" | "complete" | "average";
export interface Merge { a: number; b: number; h: number; members: number[] } // a, b: cluster ids (< n: points, ≥ n: merges)

/** Agglomerative clustering (naive O(n³), fine for small n). Cluster n + m is created by merge m. */
export function agglomerate(X: number[][], link: Linkage): Merge[] {
  const n = X.length;
  const D = X.map((a) => X.map((b) => Math.sqrt(d2(a, b))));
  let active = X.map((_, i) => ({ id: i, members: [i] }));
  const merges: Merge[] = [];
  while (active.length > 1) {
    let best = { i: 0, j: 1, h: Infinity };
    for (let i = 0; i < active.length; i++) for (let j = i + 1; j < active.length; j++) {
      const ds = active[i].members.flatMap((p) => active[j].members.map((q) => D[p][q]));
      const h = link === "single" ? Math.min(...ds) : link === "complete" ? Math.max(...ds) : ds.reduce((s, v) => s + v, 0) / ds.length;
      if (h < best.h - 1e-12) best = { i, j, h };
    }
    const A = active[best.i], B = active[best.j];
    const members = [...A.members, ...B.members];
    merges.push({ a: A.id, b: B.id, h: best.h, members });
    active = [...active.filter((_, k) => k !== best.i && k !== best.j), { id: n + merges.length - 1, members }];
  }
  return merges;
}

/** Cluster labels after cutting the tree at height h (merges with height ≤ h are applied). */
export function cut(n: number, merges: Merge[], h: number): number[] {
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  for (const m of merges) if (m.h <= h) { const [r, ...rest] = m.members.map(find); rest.forEach((x) => (parent[x] = r)); }
  const roots = [...new Set(Array.from({ length: n }, (_, i) => find(i)))].sort((a, b) => a - b);
  return Array.from({ length: n }, (_, i) => roots.indexOf(find(i)));
}

/* ------------------------------------------------------------------ choosing k */

/** Mean silhouette: s(i) = (b − a)/max(a, b) with a = mean intra-cluster distance, b = nearest other cluster's mean distance. */
export function silhouette(X: number[][], lab: number[]): number {
  const K = Math.max(...lab) + 1;
  let s = 0;
  X.forEach((x, i) => {
    const mean = (k: number) => {
      const ds = X.flatMap((y, j) => (lab[j] === k && j !== i ? [Math.sqrt(d2(x, y))] : []));
      return ds.length ? ds.reduce((p, q) => p + q, 0) / ds.length : 0;
    };
    const a = mean(lab[i]);
    const b = Math.min(...Array.from({ length: K }, (_, k) => k).filter((k) => k !== lab[i]).map(mean));
    s += Math.max(a, b) > 0 ? (b - a) / Math.max(a, b) : 0;
  });
  return s / X.length;
}

/** Best-of-5 k-means++ runs for each k: WCSS and mean silhouette. */
export function chooseK(X: number[][], ks: number[]): { wcss: number[]; sil: number[] } {
  const res = ks.map((k) => {
    let best: LloydState | null = null;
    for (let r = 0; r < 5; r++) {
      const last = lloyd(X, seedPlusPlus(X, k, 100 * k + r)).at(-1)!;
      if (!best || last.obj < best.obj) best = last;
    }
    return { w: best!.obj, s: k > 1 ? silhouette(X, best!.lab) : 0 };
  });
  return { wcss: res.map((r) => r.w), sil: res.map((r) => r.s) };
}

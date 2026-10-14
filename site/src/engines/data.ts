// Deterministic 2-D toy datasets for the in-browser visuals.

/** mulberry32: a tiny seeded PRNG returning floats in [0, 1). */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rand: () => number): number {
  // Box–Muller; 1 - u keeps the log argument in (0, 1].
  const u = 1 - rand();
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Round to 1e-6 so server-side (Node) and browser renders agree exactly: Math.cos / Math.log may
 * differ in the last ulp between JS engines, which would otherwise break React hydration.
 */
export const r6 = (v: number) => Math.round(v * 1e6) / 1e6;

export interface Dataset {
  X: number[][];
  y: number[];
}

/** Two Gaussian blobs with labels −1 / +1. */
export function blobs(n = 40, seed = 7, spread = 0.9): Dataset {
  const rand = rng(seed);
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const label = i < n / 2 ? -1 : 1;
    const [cx, cy] = label < 0 ? [-1.2, -0.8] : [1.2, 0.9];
    X.push([r6(cx + spread * gaussian(rand)), r6(cy + spread * gaussian(rand))]);
    y.push(label);
  }
  return { X, y };
}

/** Two interleaving half-moons with labels −1 / +1. */
export function moons(n = 120, seed = 0, noise = 0.25): Dataset {
  const rand = rng(seed);
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const outer = i < n / 2;
    const t = Math.PI * rand();
    const [px, py] = outer ? [Math.cos(t), Math.sin(t)] : [1 - Math.cos(t), 0.5 - Math.sin(t)];
    X.push([r6(px + noise * gaussian(rand)), r6(py + noise * gaussian(rand))]);
    y.push(outer ? -1 : 1);
  }
  return { X, y };
}

/** A regular grid padded around the data, row-major like learnml.Grid. */
export function gridFor(X: number[][], n = 40, pad = 0.6) {
  const xs0 = X.map((p) => p[0]);
  const ys0 = X.map((p) => p[1]);
  const lin = (lo: number, hi: number) => Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1));
  const x = lin(Math.min(...xs0) - pad, Math.max(...xs0) + pad);
  const y = lin(Math.min(...ys0) - pad, Math.max(...ys0) + pad);
  const points: number[][] = [];
  for (const yy of y) for (const xx of x) points.push([xx, yy]);
  return { x, y, points };
}

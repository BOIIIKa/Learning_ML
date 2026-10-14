// Chapter 13 (Markov chain methods): Metropolis–Hastings on a 2-D target, PageRank and the random
// surfer, and a collapsed Gibbs sampler for a tiny LDA. Pure, deterministic maths.
import { gaussian, rng } from "./data";

/* ------------------------------------------------------------------ Metropolis–Hastings */

/** A two-bump target (unnormalized log density): a mixture of two correlated Gaussians. */
export function logTarget(x: number, y: number): number {
  const a = -0.5 * (((x + 1.2) ** 2 - 1.4 * (x + 1.2) * (y + 0.6) + (y + 0.6) ** 2) / 0.51) / 0.6;
  const b = -0.5 * ((x - 1.3) ** 2 + (y - 0.9) ** 2) / 0.35;
  const m = Math.max(a, b);
  return m + Math.log(Math.exp(a - m) + 0.8 * Math.exp(b - m));
}

/** Random-walk Metropolis with Gaussian proposals of scale `step`. */
export function metropolis(n: number, step: number, seed: number, start: [number, number] = [-2.5, 2]): { path: [number, number][]; acc: number } {
  const rand = rng(seed);
  let [x, y] = start;
  let lp = logTarget(x, y);
  const path: [number, number][] = [[x, y]];
  let acc = 0;
  for (let t = 0; t < n; t++) {
    const px = x + step * gaussian(rand), py = y + step * gaussian(rand);
    const lq = logTarget(px, py);
    if (Math.log(rand()) < lq - lp) { x = px; y = py; lp = lq; acc++; }
    path.push([x, y]);
  }
  return { path, acc: acc / n };
}

/** Lag-1 autocorrelation of the x-coordinate: high = slow mixing. */
export function lag1(path: [number, number][]): number {
  const xs = path.map((p) => p[0]);
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  let num = 0, den = 0;
  for (let i = 0; i < xs.length; i++) { den += (xs[i] - m) ** 2; if (i) num += (xs[i] - m) * (xs[i - 1] - m); }
  return num / den;
}

/* ------------------------------------------------------------------ PageRank */

/** PageRank by power iteration on a list of out-links: r ← (1 − d)/n + d Σ_{j→i} r_j / out(j); dangling pages link everywhere. */
export function pagerank(links: number[][], d = 0.85, iters = 100): number[] {
  const n = links.length;
  let r = new Array(n).fill(1 / n);
  for (let t = 0; t < iters; t++) {
    const next = new Array(n).fill((1 - d) / n);
    links.forEach((out, j) => {
      if (!out.length) for (let i = 0; i < n; i++) next[i] += (d * r[j]) / n;
      else for (const i of out) next[i] += (d * r[j]) / out.length;
    });
    r = next;
  }
  return r;
}

/** Visit frequencies of a random surfer: follow a random out-link with probability d, otherwise jump anywhere. */
export function surfer(links: number[][], d: number, steps: number, seed: number): number[] {
  const rand = rng(seed);
  const n = links.length;
  const visits = new Array(n).fill(0);
  let p = 0;
  for (let t = 0; t < steps; t++) {
    visits[p]++;
    const out = links[p];
    p = rand() < d && out.length ? out[Math.floor(rand() * out.length)] : Math.floor(rand() * n);
  }
  return visits.map((v) => v / steps);
}

/* ------------------------------------------------------------------ LDA by collapsed Gibbs sampling */

export interface LdaState { z: number[][]; nkw: number[][]; ndk: number[][]; nk: number[] }

/** A tiny corpus over 8 words: 0–3 medical, 4–7 financial; documents mix the two themes. */
export const VOCAB = { zh: ["病人", "医生", "医院", "治疗", "股票", "市场", "投资", "价格"], en: ["patient", "doctor", "hospital", "treatment", "stock", "market", "invest", "price"] };
export function corpus(seed = 1, D = 12, L = 14): number[][] {
  const rand = rng(seed);
  return Array.from({ length: D }, (_, d) => {
    const mix = d < 4 ? 0.9 : d < 8 ? 0.1 : 0.5; // share of medical words
    return Array.from({ length: L }, () => (rand() < mix ? Math.floor(rand() * 4) : 4 + Math.floor(rand() * 4)));
  });
}

function counts(docs: number[][], z: number[][], K: number, V: number): LdaState {
  const nkw = Array.from({ length: K }, () => new Array(V).fill(0));
  const ndk = docs.map(() => new Array(K).fill(0));
  const nk = new Array(K).fill(0);
  docs.forEach((doc, d) => doc.forEach((w, i) => { const k = z[d][i]; nkw[k][w]++; ndk[d][k]++; nk[k]++; }));
  return { z, nkw, ndk, nk };
}

/** Collapsed Gibbs: P(z = k | rest) ∝ (n_dk + α)(n_kw + β)/(n_k + Vβ), counts excluding the current word. Returns the state after every sweep. */
export function ldaGibbs(docs: number[][], K: number, V: number, sweeps: number, seed: number, alpha = 0.5, beta = 0.1): LdaState[] {
  const rand = rng(seed);
  const z = docs.map((doc) => doc.map(() => Math.floor(rand() * K)));
  let st = counts(docs, z, K, V);
  const out = [JSON.parse(JSON.stringify(st)) as LdaState];
  for (let s = 0; s < sweeps; s++) {
    docs.forEach((doc, d) => doc.forEach((w, i) => {
      const k0 = st.z[d][i];
      st.nkw[k0][w]--; st.ndk[d][k0]--; st.nk[k0]--;
      const p = Array.from({ length: K }, (_, k) => ((st.ndk[d][k] + alpha) * (st.nkw[k][w] + beta)) / (st.nk[k] + V * beta));
      let u = rand() * p.reduce((a, b) => a + b, 0), k = 0;
      while (k < K - 1 && u >= p[k]) { u -= p[k]; k++; }
      st.z[d][i] = k;
      st.nkw[k][w]++; st.ndk[d][k]++; st.nk[k]++;
    }));
    out.push(JSON.parse(JSON.stringify(st)) as LdaState);
  }
  return out;
}

/** How cleanly topics separate the two themes: mean over topics of max(share of medical, share of financial) in n_kw. */
export function purity(st: LdaState): number {
  return st.nkw.reduce((s, row) => {
    const tot = row.reduce((a, b) => a + b, 0) || 1;
    const med = row.slice(0, 4).reduce((a, b) => a + b, 0);
    return s + Math.max(med, tot - med) / tot;
  }, 0) / st.nkw.length;
}

// Chapter 18: regularisation and generalisation in deep networks.
// - The scenario: a wide one-hidden-layer ReLU network fitted to 12 students, trained with Adam, with
//   decoupled weight decay, dropout or input-jitter augmentation; early stopping read off the test curve.
// - Early stopping ≈ L2: gradient descent's spectral filter against ridge regression's.
// - Gradient descent on an underdetermined least-squares problem converges to the solution nearest its start.
// - Replays of two precomputed experiments (scripts/precompute_ch18.py): loss-landscape slices and grokking.
// Everything here uses only + − × ÷ and √ on r6-rounded data, so Node and browsers agree bit for bit.
import { gaussian, r6, rng } from "./data";
import { sample, truth } from "./generalization";
import grokData from "./precomputed/grokking.json";
import landData from "./precomputed/landscape.json";

// ---------- The scenario: twelve students, a network with 193 parameters ----------

export const students = sample(12, 17);
export const newStudents = sample(200, 77);
export const HIDDEN = 64;
export const LAB_STEPS = 3000;
export const SNAP_EVERY = 50;
export const LAB_LR = 0.004;
export type Reg = "wd" | "dropout" | "augment";
/** Strength levels per method; index 0 is "no regulariser" for every method. */
export const LEVELS: Record<Reg, number[]> = {
  wd: [0, 0.3, 1, 3, 10],
  dropout: [0, 0.2, 0.4, 0.6, 0.8],
  augment: [0, 0.5, 1, 1.5, 2.5],
};

// Hours and scores are rescaled to u = (x − 5)/5 and t = (y − 60)/20 for training.
const toU = (x: number) => (x - 5) / 5;
const toT = (y: number) => (y - 60) / 20;

export interface LabNet { W1: Float64Array; b1: Float64Array; W2: Float64Array; b2: number }

export function predict(net: LabNet, x: number) {
  const u = toU(x);
  let f = net.b2;
  for (let k = 0; k < HIDDEN; k++) { const z = net.W1[k] * u + net.b1[k]; if (z > 0) f += net.W2[k] * z; }
  return 60 + 20 * f;
}

const mseOf = (net: LabNet, xs: number[], ys: number[]) => xs.reduce((s, x, i) => s + (predict(net, x) - ys[i]) ** 2, 0) / xs.length;

/**
 * Train the 1–64–1 ReLU network on the 12 students with full-batch Adam (lr 0.01) for 3000 steps.
 * `wd`: decoupled weight decay (AdamW, each step w ← w − lr·wd·w); `dropout`: probability of
 * dropping each hidden unit (inverted dropout, off at evaluation); `augment`: std of the Gaussian
 * jitter added to every student's hours at every step. Snapshots every 50 steps.
 */
export function trainLab(reg: Reg, level: number, seed = 3, data = students) {
  const s = LEVELS[reg][level];
  const rand = rng(seed);
  const W1 = Float64Array.from({ length: HIDDEN }, () => r6(Math.SQRT2 * gaussian(rand)));
  // Kinks −b/w spread over the data range [−1, 1].
  const b1 = W1.map((w) => r6(-w * (2 * rand() - 1)));
  const W2 = Float64Array.from({ length: HIDDEN }, () => r6(gaussian(rand) / 8));
  const net: LabNet = { W1, b1, W2, b2: 0 };
  const P = [W1, b1, W2];
  const m = P.map((p) => new Float64Array(p.length)), v = P.map((p) => new Float64Array(p.length));
  let mb = 0, vb = 0;
  const g = P.map((p) => new Float64Array(p.length));
  const us = data.xs.map(toU), ts = data.ys.map(toT), n = us.length;
  const z = new Float64Array(HIDDEN), mask = new Float64Array(HIDDEN);
  const noise = rng(seed + 100);
  const snaps: LabNet[] = [];
  const train: number[] = [], test: number[] = [];
  const snap = () => {
    const c = { W1: W1.slice(), b1: b1.slice(), W2: W2.slice(), b2: net.b2 };
    snaps.push(c);
    train.push(r6(mseOf(c, data.xs, data.ys)));
    test.push(r6(mseOf(c, newStudents.xs, newStudents.ys)));
  };
  snap();
  let b1m = 1, b2m = 1;
  for (let step = 1; step <= LAB_STEPS; step++) {
    g.forEach((a) => a.fill(0));
    let gb = 0;
    for (let i = 0; i < n; i++) {
      const u = reg === "augment" && s > 0 ? us[i] + r6((s / 5) * gaussian(noise)) : us[i];
      let f = net.b2;
      for (let k = 0; k < HIDDEN; k++) {
        mask[k] = reg === "dropout" && s > 0 ? (rand() < s ? 0 : 1 / (1 - s)) : 1;
        z[k] = W1[k] * u + b1[k];
        if (z[k] > 0) f += W2[k] * z[k] * mask[k];
      }
      const d = (f - ts[i]) / n; // ∂/∂f of the mean of ½(f − t)²
      gb += d;
      for (let k = 0; k < HIDDEN; k++) {
        if (z[k] <= 0 || mask[k] === 0) continue;
        const dz = d * W2[k] * mask[k];
        g[2][k] += d * z[k] * mask[k];
        g[0][k] += dz * u;
        g[1][k] += dz;
      }
    }
    b1m *= 0.9; b2m *= 0.999;
    for (let q = 0; q < 3; q++) {
      const p = P[q], mq = m[q], vq = v[q], gq = g[q];
      for (let k = 0; k < p.length; k++) {
        mq[k] = 0.9 * mq[k] + 0.1 * gq[k];
        vq[k] = 0.999 * vq[k] + 0.001 * gq[k] * gq[k];
        p[k] -= LAB_LR * (mq[k] / (1 - b1m) / (Math.sqrt(vq[k] / (1 - b2m)) + 1e-8) + (reg === "wd" ? s * p[k] : 0));
      }
    }
    mb = 0.9 * mb + 0.1 * gb;
    vb = 0.999 * vb + 0.001 * gb * gb;
    net.b2 -= LAB_LR * (mb / (1 - b1m) / (Math.sqrt(vb / (1 - b2m)) + 1e-8));
    if (step % SNAP_EVERY === 0) snap();
  }
  let best = 0;
  test.forEach((v, i) => { if (v < test[best]) best = i; });
  return { snaps, train, test, best };
}

/** The noise floor: the error of the true curve itself on the new students. */
export const truthMse = r6(newStudents.xs.reduce((s, x, i) => s + (truth(x) - newStudents.ys[i]) ** 2, 0) / newStudents.xs.length);

// ---------- Early stopping ≈ ridge on a 2-D least-squares problem ----------

/** L(w) = ½ (w − w*)ᵀ A (w − w*), A = Q diag(λ₁, λ₂) Qᵀ with an exact rotation (no cos / sin). */
export const ES = { lam: [1, 0.04] as const, Q: [[0.8, -0.6], [0.6, 0.8]] as const, wStar: [1, 2] as const, eta: 0.5 };

const powi = (b: number, t: number) => { let r = 1; for (let i = 0; i < t; i++) r *= b; return r; };
/** Fraction of the least-squares solution that gradient descent from 0 has recovered along an eigenvalue λ after t steps. */
export const gdFilter = (lam: number, eta: number, t: number) => 1 - powi(1 - eta * lam, t);
/** The same fraction for ridge regression with penalty α. */
export const ridgeFilter = (lam: number, alpha: number) => lam / (lam + alpha);

/** Coordinates of w* in the eigenbasis, then back: w = Q diag(φᵢ) Qᵀ w*. */
function filtered(phi: [number, number]) {
  const { Q, wStar } = ES;
  const c = [Q[0][0] * wStar[0] + Q[1][0] * wStar[1], Q[0][1] * wStar[0] + Q[1][1] * wStar[1]];
  const e = [phi[0] * c[0], phi[1] * c[1]];
  return [Q[0][0] * e[0] + Q[0][1] * e[1], Q[1][0] * e[0] + Q[1][1] * e[1]] as [number, number];
}
export const gdAt = (t: number) => filtered([gdFilter(ES.lam[0], ES.eta, t), gdFilter(ES.lam[1], ES.eta, t)]);
export const ridgeAt = (alpha: number) => filtered([ridgeFilter(ES.lam[0], alpha), ridgeFilter(ES.lam[1], alpha)]);
/** The ridge penalty that early stopping after t steps imitates. */
export const alphaFor = (t: number) => 1 / (ES.eta * t);
export const esLoss = ([x, y]: number[]) => {
  const { Q, wStar, lam } = ES;
  const d = [x - wStar[0], y - wStar[1]];
  const c = [Q[0][0] * d[0] + Q[1][0] * d[1], Q[0][1] * d[0] + Q[1][1] * d[1]];
  return 0.5 * (lam[0] * c[0] * c[0] + lam[1] * c[1] * c[1]);
};

// ---------- Gradient descent on an underdetermined problem ----------

/** One equation a·w = y in two unknowns: a line of exact solutions. */
export const MN = { a: [1, 2] as const, y: 3, eta: 0.1 };
export function gdUnder(w0: [number, number], steps = 40) {
  const { a, y, eta } = MN;
  const path: [number, number][] = [w0];
  let w = w0;
  for (let t = 0; t < steps; t++) {
    const r = a[0] * w[0] + a[1] * w[1] - y;
    w = [w[0] - eta * a[0] * r, w[1] - eta * a[1] * r];
    path.push(w);
  }
  return path;
}
/** The limit: the solution closest to the start, w₀ + a (y − aᵀw₀)/‖a‖². From 0 it is the minimum-norm solution. */
export function underLimit(w0: [number, number]): [number, number] {
  const { a, y } = MN;
  const k = (y - a[0] * w0[0] - a[1] * w0[1]) / (a[0] * a[0] + a[1] * a[1]);
  return [w0[0] + k * a[0], w0[1] + k * a[1]];
}

// ---------- Precomputed: loss-landscape slices ----------

export type LandName = "plain" | "residual" | "small" | "large";
export interface LandRun { loss: number[][]; trainAcc: number; testAcc: number }
export const LAND = landData as unknown as { K: number; R: number; runs: Record<LandName, LandRun[]> };
/** Mean loss over the four points at distance r (in grid steps) from the centre along the axes. */
export function ringLoss(S: number[][], r: number) {
  const c = (S.length - 1) / 2;
  return (S[c][c + r] + S[c][c - r] + S[c + r][c] + S[c - r][c]) / 4;
}

// ---------- Precomputed: grokking ----------

export interface GrokRun { step: number[]; train: number[]; test: number[]; norm: number[] }
export const GROK = grokData as unknown as {
  p: number; hidden: number; frac: number; lr: number; steps: number; nTrain: number; wds: number[];
  runs: GrokRun[]; seeds: { wd1: GrokRun[]; wd0: GrokRun[] };
};
/** First recorded step at which the accuracy reaches `level` (Infinity if never). */
export function firstStep(run: GrokRun, key: "train" | "test", level: number) {
  const i = run[key].findIndex((v) => v >= level);
  return i < 0 ? Infinity : run.step[i];
}

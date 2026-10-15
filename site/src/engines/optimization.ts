// Chapter 16: first-order optimizers (GD, heavy-ball momentum, Nesterov, RMSProp, Adam, AdamW),
// the test surfaces of the optimizer race, GD on a stretched quadratic, the student regression
// with raw vs standardised hours, and learning-rate schedules for a small MLP.
import { moons, r6, rng } from "./data";
import { sample } from "./generalization";
import { initNet, type Net } from "./mlp";

export type Vec = number[];
export type OptName = "gd" | "momentum" | "nesterov" | "rmsprop" | "adam" | "adamw";
export interface OptCfg { lr: number; beta?: number; beta2?: number; eps?: number; wd?: number; bias?: boolean }

/** A stateful optimizer: `step(x, grad)` returns the next iterate. */
export function optimizer(name: OptName, cfg: OptCfg) {
  const { lr, beta = 0.9, beta2 = 0.999, eps = 1e-8, wd = 0, bias = true } = cfg;
  let v: Vec | null = null; // velocity (momentum) or first moment (Adam)
  let s: Vec | null = null; // second moment
  let t = 0;
  return (x: Vec, grad: (x: Vec) => Vec): Vec => {
    t++;
    v ??= x.map(() => 0);
    s ??= x.map(() => 0);
    if (name === "gd") { const g = grad(x); return x.map((xi, i) => xi - lr * g[i]); }
    if (name === "momentum") {
      const g = grad(x);
      v = v.map((vi, i) => beta * vi - lr * g[i]);
      return x.map((xi, i) => xi + v![i]);
    }
    if (name === "nesterov") {
      // Look ahead along the velocity, take the gradient there.
      const g = grad(x.map((xi, i) => xi + beta * v![i]));
      v = v.map((vi, i) => beta * vi - lr * g[i]);
      return x.map((xi, i) => xi + v![i]);
    }
    const g = grad(x);
    if (name === "rmsprop") {
      s = s.map((si, i) => beta2 * si + (1 - beta2) * g[i] ** 2);
      return x.map((xi, i) => xi - (lr * g[i]) / (Math.sqrt(s![i]) + eps));
    }
    // Adam / AdamW. AdamW's weight decay is decoupled: it shrinks x directly, outside the m/√v ratio.
    v = v.map((mi, i) => beta * mi + (1 - beta) * g[i]);
    s = s.map((si, i) => beta2 * si + (1 - beta2) * g[i] ** 2);
    const c1 = bias ? 1 - beta ** t : 1;
    const c2 = bias ? 1 - beta2 ** t : 1;
    return x.map((xi, i) => xi - lr * (v![i] / c1 / (Math.sqrt(s![i] / c2) + eps) + (name === "adamw" ? wd * xi : 0)));
  };
}

/** Iterates x₀, x₁, …, x_steps (stops early, keeping the last finite point, if it blows up). */
export function run(name: OptName, cfg: OptCfg, grad: (x: Vec) => Vec, x0: Vec, steps: number): Vec[] {
  const step = optimizer(name, cfg);
  const path = [x0];
  for (let t = 0; t < steps; t++) {
    const x = step(path[t], grad);
    if (!x.every((v) => Number.isFinite(v) && Math.abs(v) < 1e6)) break;
    path.push(x);
  }
  return path;
}

// ---------- Surfaces of the optimizer race ----------

export interface Surface {
  f: (x: Vec) => number;
  grad: (x: Vec) => Vec;
  fmin: number;
  start: Vec;
  domain: { x: [number, number]; y: [number, number] };
  /** Learning rates tuned per surface: one for the GD family, one for Adam (it works in parameter units). */
  lr: { gd: number; adam: number };
}

const C30 = Math.sqrt(3) / 2, S30 = 0.5;
const rot = ([x, y]: Vec): Vec => [C30 * x + S30 * y, -S30 * x + C30 * y];
const unrot = ([u, v]: Vec): Vec => [C30 * u - S30 * v, S30 * u + C30 * v];

export const SURFACES: Record<"valley" | "rosenbrock" | "saddle", Surface> = {
  // A stretched bowl (κ = 25) rotated by 30°, so no coordinate axis lines up with it.
  valley: {
    f: (p) => { const [u, v] = rot(p); return 0.5 * (u * u / 25 + v * v); },
    grad: (p) => { const [u, v] = rot(p); return unrot([u / 25, v]); },
    fmin: 0, start: [-3.6, -0.6], domain: { x: [-4, 4], y: [-2.4, 2.4] }, lr: { gd: 0.5, adam: 0.12 },
  },
  // Rosenbrock's banana valley with b = 10; the minimum is at (1, 1).
  rosenbrock: {
    f: ([x, y]) => (1 - x) ** 2 + 10 * (y - x * x) ** 2,
    grad: ([x, y]) => [-2 * (1 - x) - 40 * x * (y - x * x), 20 * (y - x * x)],
    fmin: 0, start: [-1.2, 1], domain: { x: [-2, 2], y: [-0.8, 3] }, lr: { gd: 0.005, adam: 0.1 },
  },
  // f = ½x² + ¼y⁴ − ½y²: a saddle at the origin, minima at (0, ±1) with f = −¼.
  saddle: {
    f: ([x, y]) => 0.5 * x * x + 0.25 * y ** 4 - 0.5 * y * y,
    grad: ([x, y]) => [x, y ** 3 - y],
    fmin: -0.25, start: [-2.4, 0.001], domain: { x: [-2.6, 2.6], y: [-1.6, 1.6] }, lr: { gd: 0.1, adam: 0.06 },
  },
};

export const RACERS: OptName[] = ["gd", "momentum", "nesterov", "adam"];

/** All racers on one surface, with their learning rates scaled by `mult`. */
export function race(surface: Surface, mult: number, steps = 150) {
  return RACERS.map((name) => {
    const lr = (name === "adam" ? surface.lr.adam : surface.lr.gd) * mult;
    const path = run(name, { lr }, surface.grad, surface.start, steps);
    return { name, path, fs: path.map(surface.f) };
  });
}

// ---------- GD on a stretched quadratic f = ½(μx² + Ly²), L = 1, μ = 1/κ ----------

export const quadGrad = (kappa: number) => ([x, y]: Vec): Vec => [x / kappa, y];
export const quadF = (kappa: number) => ([x, y]: Vec) => 0.5 * (x * x / kappa + y * y);

/** GD's worst per-step contraction max |1 − ηλ| over λ ∈ {μ, L}. */
export const gdRate = (kappa: number, eta: number) => Math.max(Math.abs(1 - eta / kappa), Math.abs(1 - eta));
/** The best step 2/(L + μ) and its rate (κ − 1)/(κ + 1). */
export const optimalEta = (kappa: number) => 2 / (1 + 1 / kappa);
/** Heavy ball with the optimal (η, β) for a quadratic: rate (√κ − 1)/(√κ + 1). */
export function heavyBall(kappa: number) {
  const r = Math.sqrt(kappa);
  return { eta: 4 / (1 + 1 / r) ** 2, beta: ((r - 1) / (r + 1)) ** 2, rate: (r - 1) / (r + 1) };
}

/** Steps until ‖x‖ ≤ tol·‖x₀‖ (Infinity if not reached within `max`). */
export function stepsTo(path: Vec[], tol = 1e-3) {
  const n0 = Math.hypot(...path[0]);
  const i = path.findIndex((p) => Math.hypot(...p) <= tol * n0);
  return i < 0 ? Infinity : i;
}

export const QUAD_START: Vec = [-3, 1];

export function quadRun(kappa: number, eta: number, method: "gd" | "momentum", steps = 400) {
  const cfg = method === "gd" ? { lr: eta } : { lr: heavyBall(kappa).eta, beta: heavyBall(kappa).beta };
  return run(method, cfg, quadGrad(kappa), QUAD_START, steps);
}

// ---------- The scenario: GD on the teacher's regression, raw vs standardised hours ----------

export const students = sample(40, 16);

/** Features (1, u) with u = hours, or standardised hours. */
export function design(standardise: boolean) {
  const xs = students.xs;
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  const sd = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length);
  const us = standardise ? xs.map((x) => (x - m) / sd) : xs;
  // Hessian of ½·mean (w₀ + w₁u − y)² is [[1, ū], [ū, mean u²]].
  const ub = us.reduce((a, b) => a + b, 0) / us.length;
  const u2 = us.reduce((a, b) => a + b * b, 0) / us.length;
  const tr = 1 + u2, det = u2 - ub * ub;
  const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const L = tr / 2 + disc, mu = tr / 2 - disc;
  const loss = (w: Vec) => 0.5 * us.reduce((s, u, i) => s + (w[0] + w[1] * u - students.ys[i]) ** 2, 0) / us.length;
  const grad = (w: Vec): Vec => {
    let g0 = 0, g1 = 0;
    us.forEach((u, i) => { const r = w[0] + w[1] * u - students.ys[i]; g0 += r; g1 += r * u; });
    return [g0 / us.length, g1 / us.length];
  };
  // Least squares, for the best loss and to map weights back to raw hours.
  const yb = students.ys.reduce((a, b) => a + b, 0) / us.length;
  const uy = us.reduce((s, u, i) => s + u * students.ys[i], 0) / us.length;
  const w1 = (uy - ub * yb) / det, w0 = yb - w1 * ub;
  /** Score predicted at raw hours x by weights w. */
  const line = (w: Vec) => (x: number) => w[0] + w[1] * (standardise ? (x - m) / sd : x);
  return { L, mu, kappa: L / mu, loss, grad, best: loss([w0, w1]), wStar: [w0, w1], line };
}

export const DIAL_STEPS = 100;

/** GD from w = 0 for DIAL_STEPS steps; the loss history (Infinity once it blows up). */
export function dial(standardise: boolean, lr: number) {
  const d = design(standardise);
  const path = run("gd", { lr }, d.grad, [0, 0], DIAL_STEPS);
  const losses = Array.from({ length: DIAL_STEPS + 1 }, (_, t) => (t < path.length ? d.loss(path[t]) : Infinity));
  // First step within 1% of the best possible loss (−1 if never within DIAL_STEPS).
  const hit = losses.findIndex((l) => l <= 1.01 * d.best);
  return { ...d, path, losses, diverged: path.length <= DIAL_STEPS, hit };
}

// ---------- Learning-rate schedules on a small MLP ----------

export type Schedule = "constant" | "step" | "cosine" | "warmup";
export const SCHEDULES: Schedule[] = ["constant", "step", "cosine", "warmup"];

/**
 * cos(x) from + − × ÷ only (Taylor series at x/4, then two double-angle steps). Browsers' Math.cos
 * can differ from Node's in the last bit, and an unstable training run amplifies that into a
 * different curve: the server-rendered widget would then not match the client (hydration error).
 */
export function detCos(x: number) {
  const q = x / 4, q2 = q * q;
  let term = 1, c = 1;
  for (let k = 1; k <= 14; k++) { term *= -q2 / ((2 * k - 1) * (2 * k)); c += term; }
  const c2 = 2 * c * c - 1;
  return 2 * c2 * c2 - 1;
}

/** Learning rate at step t of T for a peak rate `peak`. */
export function lrAt(kind: Schedule, peak: number, t: number, T: number) {
  if (kind === "constant") return peak;
  if (kind === "step") return peak * 0.1 ** Math.floor((3 * t) / T); // ÷10 at T/3 and 2T/3
  const warm = kind === "warmup" ? Math.round(0.1 * T) : 0;
  if (t < warm) return (peak * (t + 1)) / warm;
  return 0.5 * peak * (1 + detCos((Math.PI * (t - warm)) / (T - warm)));
}

export const SCHED_EPOCHS = 60;
export const SCHED_PEAKS = [0.03, 0.1, 0.3];
export const moonData = moons(160, 3, 0.2);
export const SCHED_SIZES = [2, 16, 16, 1];

/**
 * A 2-16-16-1 tanh network with flat Float64Array parameters, for speed (the widget trains it
 * 4 × 9600 times). Same maths as mlp.ts; the tests check its gradient against mlp.ts's backward.
 */
export function flatNet(net: Net) {
  const [L1, L2, L3] = net;
  return Float64Array.from([...L1.W.flat(), ...L1.b, ...L2.W.flat(), ...L2.b, ...L3.W.flat(), ...L3.b]);
}
const H = 16, O1 = 0, B1 = 32, O2 = 48, B2 = 304, O3 = 320, B3 = 336, NP = 337;
const h1 = new Float64Array(H), h2 = new Float64Array(H), d2 = new Float64Array(H), d1 = new Float64Array(H);

/** Logit of the flat net at x; fills the hidden activations h1, h2. */
function flatForward(p: Float64Array, x0: number, x1: number) {
  for (let i = 0; i < H; i++) h1[i] = Math.tanh(p[O1 + 2 * i] * x0 + p[O1 + 2 * i + 1] * x1 + p[B1 + i]);
  let f = p[B3];
  for (let i = 0; i < H; i++) {
    let z = p[B2 + i];
    for (let j = 0; j < H; j++) z += p[O2 + H * i + j] * h1[j];
    h2[i] = Math.tanh(z);
    f += p[O3 + i] * h2[i];
  }
  return f;
}

/** Adds scale · ∂(BCE)/∂p at (x, y) to g. */
export function flatGrad(p: Float64Array, x: number[], y: number, g: Float64Array, scale = 1) {
  const f = flatForward(p, x[0], x[1]);
  const df = (1 / (1 + Math.exp(-f)) - y) * scale;
  g[B3] += df;
  for (let i = 0; i < H; i++) { g[O3 + i] += df * h2[i]; d2[i] = df * p[O3 + i] * (1 - h2[i] * h2[i]); }
  d1.fill(0);
  for (let i = 0; i < H; i++) {
    g[B2 + i] += d2[i];
    for (let j = 0; j < H; j++) { g[O2 + H * i + j] += d2[i] * h1[j]; d1[j] += d2[i] * p[O2 + H * i + j]; }
  }
  for (let j = 0; j < H; j++) {
    const d = d1[j] * (1 - h1[j] * h1[j]);
    g[B1 + j] += d; g[O1 + 2 * j] += d * x[0]; g[O1 + 2 * j + 1] += d * x[1];
  }
}

/** Mean binary cross-entropy of the flat net on a dataset. */
export function flatBce(p: Float64Array, data = moonData) {
  let s = 0;
  data.X.forEach((x, i) => {
    const f = flatForward(p, x[0], x[1]);
    s += Math.max(f, 0) - f * (data.y[i] > 0 ? 1 : 0) + Math.log1p(Math.exp(-Math.abs(f)));
  });
  return s / data.X.length;
}

/** Minibatch SGD with momentum 0.9 (batch 8) under a schedule; training loss after every epoch. */
export function trainSchedule(kind: Schedule, peak: number, epochs = SCHED_EPOCHS, seed = 4, data = moonData) {
  // r6: initNet's Box–Muller uses Math.cos, which can differ in the last bit between Node and browsers.
  const p = flatNet(initNet(SCHED_SIZES, seed, 1)).map(r6);
  const vel = new Float64Array(NP), g = new Float64Array(NP);
  const n = data.X.length, B = 8, perEpoch = Math.floor(n / B), T = epochs * perEpoch;
  const rand = rng(seed + 100);
  const losses = [r6(flatBce(p, data))];
  const order = Array.from({ length: n }, (_, i) => i);
  let t = 0;
  for (let e = 0; e < epochs; e++) {
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    for (let b = 0; b < perEpoch; b++, t++) {
      const lr = lrAt(kind, peak, t, T);
      g.fill(0);
      for (let k = b * B; k < b * B + B; k++) flatGrad(p, data.X[order[k]], data.y[order[k]] > 0 ? 1 : 0, g, 1 / B);
      for (let k = 0; k < NP; k++) { vel[k] = 0.9 * vel[k] - lr * g[k]; p[k] += vel[k]; }
    }
    const l = flatBce(p, data);
    losses.push(Number.isFinite(l) ? r6(l) : Infinity);
  }
  return losses;
}

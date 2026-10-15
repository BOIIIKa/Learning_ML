// Chapter 17: backpropagation on a computational graph, activation / gradient variance across
// depth under different initialisations, deep-net training with small vs He initialisation, and
// forward- vs reverse-mode automatic differentiation.
import { gaussian, moons, r6, rng } from "./data";

// ---------- One neuron as a computational graph ----------

export interface GraphInput { x1: number; x2: number; w1: number; w2: number; b: number; y: number }
export const NEURON: GraphInput = { x1: 1.5, x2: -0.5, w1: 0.8, w2: -1.2, b: 0.1, y: 0 };

export type NodeId = "w1" | "x1" | "w2" | "x2" | "b" | "y" | "u1" | "u2" | "s" | "p" | "r" | "L";
/** `local` spells out how the node's gradient is formed: downstream gradient × local derivative. */
export interface GraphNode { id: NodeId; label: string; op: string; inputs: NodeId[]; value: number; grad: number; local: string }

/**
 * Forward values and backward gradients of L = ½(σ(w₁x₁ + w₂x₂ + b) − y)², node by node.
 * `forwardOrder` / `backwardOrder` give the order in which the widget reveals them.
 */
export function neuronGraph(g: GraphInput) {
  const u1 = g.w1 * g.x1, u2 = g.w2 * g.x2, s = u1 + u2 + g.b, p = 1 / (1 + Math.exp(-s)), r = p - g.y, L = 0.5 * r * r;
  // Reverse sweep: each node's gradient = Σ over its consumers of (consumer's gradient × local derivative).
  const dL = 1, dr = dL * r, dp = dr, ds = dp * p * (1 - p), du1 = ds, du2 = ds, db = ds;
  const dw1 = du1 * g.x1, dx1 = du1 * g.w1, dw2 = du2 * g.x2, dx2 = du2 * g.w2, dy = -dr;
  const nodes: GraphNode[] = [
    { id: "w1", label: "w₁", op: "", inputs: [], value: g.w1, grad: dw1, local: "= ∂L/∂u₁ × x₁" },
    { id: "x1", label: "x₁", op: "", inputs: [], value: g.x1, grad: dx1, local: "= ∂L/∂u₁ × w₁" },
    { id: "w2", label: "w₂", op: "", inputs: [], value: g.w2, grad: dw2, local: "= ∂L/∂u₂ × x₂" },
    { id: "x2", label: "x₂", op: "", inputs: [], value: g.x2, grad: dx2, local: "= ∂L/∂u₂ × w₂" },
    { id: "b", label: "b", op: "", inputs: [], value: g.b, grad: db, local: "= ∂L/∂s × 1" },
    { id: "y", label: "y", op: "", inputs: [], value: g.y, grad: dy, local: "= ∂L/∂r × (−1)" },
    { id: "u1", label: "u₁", op: "×", inputs: ["w1", "x1"], value: u1, grad: du1, local: "= ∂L/∂s × 1" },
    { id: "u2", label: "u₂", op: "×", inputs: ["w2", "x2"], value: u2, grad: du2, local: "= ∂L/∂s × 1" },
    { id: "s", label: "s", op: "+", inputs: ["u1", "u2", "b"], value: s, grad: ds, local: "= ∂L/∂p × p(1 − p)" },
    { id: "p", label: "p", op: "σ", inputs: ["s"], value: p, grad: dp, local: "= ∂L/∂r × 1" },
    { id: "r", label: "r", op: "−", inputs: ["p", "y"], value: r, grad: dr, local: "= ∂L/∂L × r" },
    { id: "L", label: "L", op: "½·²", inputs: ["r"], value: L, grad: dL, local: "= 1" },
  ];
  const forwardOrder: NodeId[] = ["u1", "u2", "s", "p", "r", "L"];
  const backwardOrder: NodeId[] = ["L", "r", "p", "s", "u1", "u2", "w1", "w2", "b"];
  return { nodes, forwardOrder, backwardOrder, loss: L, grads: { w1: dw1, w2: dw2, b: db } };
}

/** Loss of the neuron, for finite-difference checks. */
export const neuronLoss = (g: GraphInput) => 0.5 * (1 / (1 + Math.exp(-(g.w1 * g.x1 + g.w2 * g.x2 + g.b))) - g.y) ** 2;

// ---------- A flat-array MLP of any depth (deterministic across Node and browsers) ----------

export type Act = "relu" | "tanh";
export interface FlatNet { sizes: number[]; W: Float64Array[]; b: Float64Array[] }

/** W ~ N(0, gain / fan-in), b = 0, rounded with r6 (Box–Muller uses Math.cos, which differs by a bit across engines). */
export function initFlat(sizes: number[], gain: number, seed = 1, std?: number): FlatNet {
  const rand = rng(seed);
  const W: Float64Array[] = [], b: Float64Array[] = [];
  for (let l = 1; l < sizes.length; l++) {
    const s = std ?? Math.sqrt(gain / sizes[l - 1]);
    W.push(Float64Array.from({ length: sizes[l] * sizes[l - 1] }, () => r6(s * gaussian(rand))));
    b.push(new Float64Array(sizes[l]));
  }
  return { sizes, W, b };
}

const act = (a: Act, z: number) => (a === "relu" ? (z > 0 ? z : 0) : Math.tanh(z));
const dact = (a: Act, z: number, h: number) => (a === "relu" ? (z > 0 ? 1 : 0) : 1 - h * h);

/** Forward pass for one input; returns pre-activations zs and activations hs (hs[0] = x). Last layer linear. */
export function forwardFlat(net: FlatNet, x: ArrayLike<number>, a: Act) {
  const L = net.W.length;
  const hs: Float64Array[] = [Float64Array.from(x)];
  const zs: Float64Array[] = [];
  for (let l = 0; l < L; l++) {
    const nin = net.sizes[l], nout = net.sizes[l + 1], W = net.W[l], h = hs[l];
    const z = new Float64Array(nout);
    for (let i = 0; i < nout; i++) {
      let v = net.b[l][i];
      for (let j = 0; j < nin; j++) v += W[i * nin + j] * h[j];
      z[i] = v;
    }
    zs.push(z);
    hs.push(l === L - 1 ? z : z.map((v) => act(a, v)));
  }
  return { zs, hs };
}

/**
 * Backpropagation for one input given dLoss/dOutput. Accumulates scale·gradients into gW, gb and
 * returns dLoss/dh for every layer (index 0 = input), the quantity whose size the widgets plot.
 */
export function backwardFlat(net: FlatNet, cache: { zs: Float64Array[]; hs: Float64Array[] }, dOut: ArrayLike<number>, a: Act, gW?: Float64Array[], gb?: Float64Array[], scale = 1) {
  const L = net.W.length;
  const dh: Float64Array[] = new Array(L + 1);
  let delta = Float64Array.from(dOut); // dL/dz of layer l
  for (let l = L - 1; l >= 0; l--) {
    const nin = net.sizes[l], nout = net.sizes[l + 1], W = net.W[l], h = cache.hs[l];
    if (gW && gb) for (let i = 0; i < nout; i++) {
      gb[l][i] += scale * delta[i];
      for (let j = 0; j < nin; j++) gW[l][i * nin + j] += scale * delta[i] * h[j];
    }
    const prev = new Float64Array(nin);
    for (let j = 0; j < nin; j++) { let v = 0; for (let i = 0; i < nout; i++) v += W[i * nin + j] * delta[i]; prev[j] = v; }
    dh[l] = prev;
    if (l > 0) { const z = cache.zs[l - 1], hh = cache.hs[l]; delta = prev.map((v, j) => v * dact(a, z[j], hh[j])); }
  }
  dh[L] = Float64Array.from(dOut);
  return dh;
}

/** 6 significant digits: keeps tiny values (r6 would round 1e-7 to 0). */
const sig6 = (v: number) => Number(v.toPrecision(6));
const rms = (v: ArrayLike<number>) => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; return Math.sqrt(s / v.length); };

// ---------- Variance across depth ----------

export const VAR_WIDTH = 96;

/**
 * Push a batch of N(0, 1) inputs through `depth` square layers (width 64) with W ~ N(0, gain/64) and
 * send random N(0, 1) gradients back from the top. Returns, per layer, the RMS of the
 * pre-activations (forward) and of dLoss/dh (backward), averaged over the batch.
 */
export function varianceProfile(a: Act, gain: number, depth: number, batch = 6, nets = 3, seed = 3) {
  const n = VAR_WIDTH;
  const rand = rng(seed + 50);
  const fwd = new Array(depth).fill(0), bwd = new Array(depth + 1).fill(0);
  const total = batch * nets;
  const nets1: FlatNet[] = [];
  for (let k = 0; k < total; k++) {
    const net = (nets1[k % nets] ??= initFlat(Array(depth + 1).fill(n), gain, seed + (k % nets)));
    const x = Array.from({ length: n }, () => gaussian(rand));
    const c = forwardFlat(net, x, a);
    c.zs.forEach((z, l) => (fwd[l] += rms(z) ** 2 / total));
    const g = Array.from({ length: n }, () => gaussian(rand));
    // Treat the top as a hidden layer too, so every layer has the same activation in front of it.
    const top = c.zs[depth - 1];
    const dTop = g.map((v, i) => v * dact(a, top[i], act(a, top[i])));
    const dh = backwardFlat(net, c, dTop, a);
    dh.forEach((d, l) => (bwd[l] += rms(d) ** 2 / total));
  }
  return { forward: fwd.map((v) => sig6(Math.sqrt(v))), backward: bwd.slice(0, depth).map((v) => sig6(Math.sqrt(v))) };
}

/** ReLU theory for the RMS of layer l's pre-activations (l = 0 is the first layer, fed N(0, 1) inputs):
 * √gain at l = 0, then each ReLU layer multiplies the second moment by gain/2. */
export const reluTheory = (gain: number, l: number) => Math.sqrt(gain) * Math.sqrt(gain / 2) ** l;

// ---------- The scenario: a deeper network stops learning ----------

export const STALL_EPOCHS = 24;
export const stallData = moons(120, 7, 0.15);
export type Init = "small" | "he";

/** In-place forward / backward with preallocated buffers (ReLU hidden layers, one linear output): the training loop's fast path. */
function workspace(net: FlatNet) {
  const L = net.W.length, S = net.sizes;
  const z = S.slice(1).map((k) => new Float64Array(k));
  const h = S.map((k) => new Float64Array(k));
  const d = S.slice(1).map((k) => new Float64Array(k));
  const fwd = (x: ArrayLike<number>) => {
    h[0].set(x);
    for (let l = 0; l < L; l++) {
      const nin = S[l], nout = S[l + 1], W = net.W[l], hi = h[l], zl = z[l], ho = h[l + 1];
      for (let i = 0; i < nout; i++) {
        let v = net.b[l][i];
        for (let j = 0; j < nin; j++) v += W[i * nin + j] * hi[j];
        zl[i] = v;
        ho[i] = l === L - 1 ? v : v > 0 ? v : 0;
      }
    }
    return h[L][0];
  };
  const bwd = (dOut: number, gW: Float64Array[], gb: Float64Array[], scale: number) => {
    d[L - 1][0] = dOut;
    for (let l = L - 1; l >= 0; l--) {
      const nin = S[l], nout = S[l + 1], W = net.W[l], hi = h[l], dl = d[l];
      for (let i = 0; i < nout; i++) {
        const di = scale * dl[i];
        gb[l][i] += di;
        for (let j = 0; j < nin; j++) gW[l][i * nin + j] += di * hi[j];
      }
      if (l > 0) {
        const dp = d[l - 1], zp = z[l - 1];
        for (let j = 0; j < nin; j++) {
          if (zp[j] <= 0) { dp[j] = 0; continue; }
          let v = 0;
          for (let i = 0; i < nout; i++) v += W[i * nin + j] * dl[i];
          dp[j] = v;
        }
      }
    }
  };
  return { fwd, bwd };
}

/** Hidden layers of width 16 (ReLU), trained with minibatch SGD + momentum on BCE; loss per epoch and per-layer gradient RMS at init. */
export function trainDeep(depth: number, init: Init, epochs = STALL_EPOCHS, lr = 0.05, seed = 2) {
  const sizes = [2, ...Array(depth).fill(16), 1];
  const net = init === "he" ? initFlat(sizes, 2, seed) : initFlat(sizes, 0, seed, 0.1);
  const ws = workspace(net);
  const gW = net.W.map((w) => new Float64Array(w.length)), gb = net.b.map((v) => new Float64Array(v.length));
  const vW = net.W.map((w) => new Float64Array(w.length)), vb = net.b.map((v) => new Float64Array(v.length));
  const { X, y } = stallData;
  const n = X.length, B = 16;
  const t = y.map((v) => (v > 0 ? 1 : 0));
  const lossOf = () => {
    let s = 0;
    X.forEach((x, i) => { const f = ws.fwd(x); s += Math.max(f, 0) - f * t[i] + Math.log1p(Math.exp(-Math.abs(f))); });
    return r6(s / n);
  };
  // Gradient size per layer (RMS of dLoss/dW over the whole data) at initialisation.
  for (let i = 0; i < n; i++) { const f = ws.fwd(X[i]); ws.bwd(1 / (1 + Math.exp(-f)) - t[i], gW, gb, 1 / n); }
  const gradRms = gW.map((g) => sig6(rms(g)));
  const losses = [lossOf()];
  const rand = rng(seed + 9);
  const order = Array.from({ length: n }, (_, i) => i);
  for (let e = 0; e < epochs; e++) {
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    for (let s0 = 0; s0 + B <= n; s0 += B) {
      gW.forEach((g) => g.fill(0)); gb.forEach((g) => g.fill(0));
      for (let k = s0; k < s0 + B; k++) { const i = order[k]; const f = ws.fwd(X[i]); ws.bwd(1 / (1 + Math.exp(-f)) - t[i], gW, gb, 1 / B); }
      net.W.forEach((W, l) => { for (let k = 0; k < W.length; k++) { vW[l][k] = 0.9 * vW[l][k] - lr * gW[l][k]; W[k] += vW[l][k]; } });
      net.b.forEach((bb, l) => { for (let k = 0; k < bb.length; k++) { vb[l][k] = 0.9 * vb[l][k] - lr * gb[l][k]; bb[k] += vb[l][k]; } });
    }
    losses.push(lossOf());
  }
  return { losses, gradRms };
}

// ---------- Forward- vs reverse-mode autodiff ----------

/** A test function f: Rⁿ → Rᵐ, f_i(x) = tanh(Σ_j a_ij x_j) · (1 + x_{i mod n}²). */
export function testFn(n: number, m: number, seed = 5) {
  const rand = rng(seed);
  const A = Array.from({ length: m }, () => Array.from({ length: n }, () => r6(2 * rand() - 1)));
  return A;
}

/** Dual numbers: value and tangent. */
interface Dual { v: number; d: number }

/** Forward mode: one pass gives J·v (a column of J when v = e_j). */
export function jvp(A: number[][], x: number[], v: number[]): number[] {
  const n = x.length;
  const xs: Dual[] = x.map((xi, j) => ({ v: xi, d: v[j] }));
  return A.map((row, i) => {
    const s = row.reduce<Dual>((acc, a, j) => ({ v: acc.v + a * xs[j].v, d: acc.d + a * xs[j].d }), { v: 0, d: 0 });
    const t = { v: Math.tanh(s.v), d: (1 - Math.tanh(s.v) ** 2) * s.d };
    const q = xs[i % n];
    const g = { v: 1 + q.v * q.v, d: 2 * q.v * q.d };
    return t.v * g.d + t.d * g.v;
  });
}

/** Reverse mode: one pass gives uᵀJ (a row of J when u = e_i): forward sweep, then adjoints backwards. */
export function vjp(A: number[][], x: number[], u: number[]): number[] {
  const n = x.length;
  const xbar = new Array(n).fill(0);
  A.forEach((row, i) => {
    const s = row.reduce((acc, a, j) => acc + a * x[j], 0);
    const t = Math.tanh(s), q = x[i % n], g = 1 + q * q;
    // y_i = t·g;  ȳ_i = u_i
    const tbar = u[i] * g, gbar = u[i] * t;
    const sbar = tbar * (1 - t * t);
    row.forEach((a, j) => (xbar[j] += sbar * a));
    xbar[i % n] += gbar * 2 * q;
  });
  return xbar;
}

export function fnValue(A: number[][], x: number[]) {
  const n = x.length;
  return A.map((row, i) => Math.tanh(row.reduce((acc, a, j) => acc + a * x[j], 0)) * (1 + x[i % n] ** 2));
}

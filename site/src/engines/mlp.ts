// A small multilayer perceptron for Part IV (chapters 14–21): dense layers, ReLU / tanh hidden
// units, a linear output layer, forward pass with cache, and reverse-mode backpropagation.
// Pure and deterministic (seeded initialisation), sized for in-browser widgets.
import { gaussian, rng } from "./data";

export type Act = "relu" | "tanh" | "linear";
/** One dense layer: W is out × in, b has length out. */
export interface Layer { W: number[][]; b: number[] }
export type Net = Layer[];

const act = (a: Act, z: number) => (a === "relu" ? Math.max(0, z) : a === "tanh" ? Math.tanh(z) : z);
const dact = (a: Act, z: number) => (a === "relu" ? (z > 0 ? 1 : 0) : a === "tanh" ? 1 - Math.tanh(z) ** 2 : 1);

/** He-style initialisation for sizes [in, h1, …, out]: W ~ N(0, 2 / fan-in), b = 0. */
export function initNet(sizes: number[], seed = 1, gain = 2): Net {
  const rand = rng(seed);
  return sizes.slice(1).map((out, l) => {
    const fanIn = sizes[l];
    const s = Math.sqrt(gain / fanIn);
    return { W: Array.from({ length: out }, () => Array.from({ length: fanIn }, () => s * gaussian(rand))), b: new Array(out).fill(0) };
  });
}

/** Number of parameters (weights + biases) of a net with layer sizes [in, h1, …, out]. */
export const paramCount = (sizes: number[]) => sizes.slice(1).reduce((s, out, l) => s + out * (sizes[l] + 1), 0);

export interface Cache { zs: number[][]; hs: number[][] } // pre-activations per layer; hs[0] = input

/** Forward pass. Hidden layers use `a`; the last layer is linear. */
export function forward(net: Net, x: number[], a: Act = "relu"): Cache {
  const hs = [x];
  const zs: number[][] = [];
  net.forEach((L, l) => {
    const h = hs[l];
    const z = L.W.map((row, i) => row.reduce((s, w, j) => s + w * h[j], L.b[i]));
    zs.push(z);
    hs.push(l === net.length - 1 ? z : z.map((v) => act(a, v)));
  });
  return { zs, hs };
}

export const predict = (net: Net, x: number[], a: Act = "relu") => forward(net, x, a).hs.at(-1)!;

/** Backpropagation: given dLoss/dOutput, return dLoss/dW and dLoss/db for every layer. */
export function backward(net: Net, cache: Cache, dOut: number[], a: Act = "relu"): Net {
  const grads: Net = net.map((L) => ({ W: L.W.map((r) => r.map(() => 0)), b: L.b.map(() => 0) }));
  let delta = dOut; // dL/dz of the current layer
  for (let l = net.length - 1; l >= 0; l--) {
    const h = cache.hs[l];
    delta.forEach((d, i) => {
      grads[l].b[i] = d;
      h.forEach((v, j) => (grads[l].W[i][j] = d * v));
    });
    if (l === 0) break;
    const prevZ = cache.zs[l - 1];
    delta = prevZ.map((z, j) => net[l].W.reduce((s, row, i) => s + row[j] * delta[i], 0) * dact(a, z));
  }
  return grads;
}

/** Which hidden units are on (z > 0), as a string key: identifies the linear region of x. */
export const pattern = (net: Net, x: number[]) => forward(net, x).zs.slice(0, -1).map((z) => z.map((v) => (v > 0 ? 1 : 0)).join("")).join("|");

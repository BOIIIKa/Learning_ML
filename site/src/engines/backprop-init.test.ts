import { describe, expect, it } from "vitest";
import {
  NEURON, backwardFlat, fnValue, forwardFlat, initFlat, jvp, neuronGraph, neuronLoss, reluTheory, testFn, trainDeep, varianceProfile, vjp,
} from "./backprop-init";
import { backward, forward, initNet } from "./mlp";

describe("chapter 17: backpropagation and initialisation", () => {
  it("the neuron graph: backward values equal finite differences, and match the worked example", () => {
    const g = neuronGraph(NEURON);
    const h = 1e-6;
    for (const k of ["w1", "w2", "b"] as const) {
      const num = (neuronLoss({ ...NEURON, [k]: NEURON[k] + h }) - neuronLoss({ ...NEURON, [k]: NEURON[k] - h })) / (2 * h);
      expect(g.grads[k]).toBeCloseTo(num, 8);
    }
    const v = Object.fromEntries(g.nodes.map((n) => [n.id, n]));
    expect(v.s.value).toBeCloseTo(1.9, 12);
    expect(v.p.value).toBeCloseTo(0.8699, 4);
    expect(g.loss).toBeCloseTo(0.3784, 4);
    expect(v.s.grad).toBeCloseTo(0.0985, 4);
    expect(g.grads.w1).toBeCloseTo(0.1477, 4);
    expect(g.grads.w2).toBeCloseTo(-0.0492, 4);
    expect(v.x1.grad).toBeCloseTo(0.0788, 4);
    // Every forward node is revealed before any gradient; the backward order is a reverse topological order.
    expect(g.forwardOrder.at(-1)).toBe("L");
    expect(g.backwardOrder[0]).toBe("L");
  });

  it("the flat MLP's backward pass equals mlp.ts backprop", () => {
    const sizes = [2, 5, 4, 1];
    const ref = initNet(sizes, 3);
    const net = initFlat(sizes, 2, 1);
    ref.forEach((L, l) => { net.W[l] = Float64Array.from(L.W.flat()); net.b[l] = Float64Array.from(L.b); });
    const x = [0.3, -1.1];
    const gW = net.W.map((w) => new Float64Array(w.length)), gb = net.b.map((b) => new Float64Array(b.length));
    for (const a of ["relu", "tanh"] as const) {
      gW.forEach((g) => g.fill(0)); gb.forEach((g) => g.fill(0));
      backwardFlat(net, forwardFlat(net, x, a), [0.7], a, gW, gb);
      const r = backward(ref, forward(ref, x, a), [0.7], a);
      r.forEach((L, l) => {
        L.W.flat().forEach((v, k) => expect(gW[l][k]).toBeCloseTo(v, 12));
        L.b.forEach((v, k) => expect(gb[l][k]).toBeCloseTo(v, 12));
      });
    }
  });

  it("ReLU variance: each layer multiplies the second moment by gain/2 (He = 2 keeps it level)", () => {
    for (const gain of [1, 2, 3]) {
      const p = varianceProfile("relu", gain, 30);
      p.forward.forEach((v, l) => { expect(v / reluTheory(gain, l)).toBeGreaterThan(0.4); expect(v / reluTheory(gain, l)).toBeLessThan(1.2); });
    }
    // The caption's numbers: after 30 layers gain 1 vanishes (~1e-5), He stays order 1, gain 4 explodes (~1e4) — and so do the gradients at the input.
    const g1 = varianceProfile("relu", 1, 30), he = varianceProfile("relu", 2, 30), g4 = varianceProfile("relu", 4, 30);
    expect(g1.forward[29]).toBeLessThan(1e-4);
    expect(g1.backward[0]).toBeLessThan(1e-4);
    expect(he.forward[29]).toBeGreaterThan(0.3);
    expect(he.forward[29]).toBeLessThan(3);
    expect(he.backward[0]).toBeGreaterThan(0.3);
    expect(he.backward[0]).toBeLessThan(3);
    expect(g4.forward[29]).toBeGreaterThan(1e4);
    expect(g4.backward[0]).toBeGreaterThan(1e4);
    // tanh: gain 1 (Xavier / LeCun) decays slowly, gain 0.5 vanishes.
    const t1 = varianceProfile("tanh", 1, 30), t05 = varianceProfile("tanh", 0.5, 30);
    expect(t1.forward[29]).toBeGreaterThan(0.05);
    expect(t05.forward[29]).toBeLessThan(1e-4);
  });

  it("the scenario: small init stalls from 5 layers on; He trains at every depth", () => {
    for (let d = 1; d <= 10; d++) {
      const small = trainDeep(d, "small"), he = trainDeep(d, "he");
      expect(he.losses.at(-1)!).toBeLessThan(0.3);
      if (d <= 3) expect(small.losses.at(-1)!).toBeLessThan(0.25);
      if (d >= 5) expect(Math.min(...small.losses)).toBeGreaterThan(0.69); // stuck at ln 2
      if (d >= 2) expect(small.gradRms[0]).toBeLessThan(he.gradRms[0] / 5);
    }
    // At 8 layers the first layer's gradient is ~1e-6 under the small init, ~3e-2 under He.
    expect(trainDeep(8, "small").gradRms[0]).toBeLessThan(2e-6);
    expect(trainDeep(8, "he").gradRms[0]).toBeGreaterThan(2e-2);
    expect(Math.log(2)).toBeCloseTo(0.6931, 4);
  });

  it("forward mode gives columns, reverse mode rows, of the same Jacobian", () => {
    const n = 5, m = 3, A = testFn(n, m), x = [0.3, -0.7, 1.1, 0.2, -0.4];
    const h = 1e-6;
    for (let j = 0; j < n; j++) {
      const e = x.map((_, k) => (k === j ? 1 : 0));
      const col = jvp(A, x, e);
      const num = fnValue(A, x.map((v, k) => v + h * e[k])).map((v, i) => (v - fnValue(A, x.map((w, k) => w - h * e[k]))[i]) / (2 * h));
      col.forEach((v, i) => expect(v).toBeCloseTo(num[i], 7));
    }
    for (let i = 0; i < m; i++) {
      const row = vjp(A, x, Array.from({ length: m }, (_, k) => (k === i ? 1 : 0)));
      row.forEach((v, j) => expect(v).toBeCloseTo(jvp(A, x, x.map((_, k) => (k === j ? 1 : 0)))[i], 12));
    }
  });
});

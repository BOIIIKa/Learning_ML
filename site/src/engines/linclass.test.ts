import { describe, expect, it } from "vitest";
import {
  LOSSES, TRUE_SEPARATOR, accuracy, crossEntropy, entropy, fitLogistic, maxEntDie, newtonStep, novikoffBound, patients,
  perceptron, perceptronDual, sameMeanAlternative, separablePatients, sigmoid, softplusNeg,
} from "./linclass";

describe("perceptron", () => {
  // The worked example in the notes.
  const X = [[2, 2], [0, 2], [1, 0]];
  const y = [1, 1, -1];

  it("reproduces the hand-computed example", () => {
    const st = perceptron(X, y, 1);
    expect(st.map((s) => s.i)).toEqual([-1, 0, 2, 2]);
    expect(st.map((s) => [...s.w, s.b])).toEqual([[0, 0, 0], [2, 2, 1], [1, 2, 0], [0, 2, -1]]);
    expect(st.at(-1)!.alpha).toEqual([1, 0, 2]);
    expect(st.at(-1)!.mistakes).toBe(0);
  });

  it("the dual form gives the same solution using only the Gram matrix", () => {
    const d = perceptronDual(X, y, 1);
    expect(d.alpha).toEqual([1, 0, 2]);
    expect(d.b).toBe(-1);
    expect(d.w).toEqual([0, 2]);
    expect(d.gram).toEqual([[8, 4, 2], [4, 4, 0], [2, 0, 1]]);
  });

  it("separable data: converges, primal = dual, and Novikoff's bound holds", () => {
    const { X: Xs, y: ys } = separablePatients(16, 9, 0.2);
    const st = perceptron(Xs, ys, 0.5);
    const last = st.at(-1)!;
    expect(last.mistakes).toBe(0);
    const d = perceptronDual(Xs, ys, 0.5);
    expect(d.updates).toBe(st.length - 1);
    expect(d.w[0]).toBeCloseTo(last.w[0], 4);
    expect(d.w[1]).toBeCloseTo(last.w[1], 4);
    expect(d.b).toBeCloseTo(last.b, 9);
    const { bound, gamma } = novikoffBound(Xs, ys, TRUE_SEPARATOR.w, TRUE_SEPARATOR.b);
    expect(gamma).toBeGreaterThan(0);
    expect(st.length - 1).toBeLessThanOrEqual(bound);
    // A different visiting order still separates the data, usually with a different line.
    const rev = perceptron(Xs, ys, 0.5, Xs.map((_, i) => Xs.length - 1 - i));
    expect(rev.at(-1)!.mistakes).toBe(0);
  });
});

describe("logistic regression", () => {
  it("is numerically stable", () => {
    expect(sigmoid(0)).toBe(0.5);
    expect(sigmoid(800)).toBe(1);
    expect(sigmoid(-800)).toBe(0);
    expect(softplusNeg(-800)).toBeCloseTo(800, 9);
    expect(softplusNeg(0)).toBeCloseTo(Math.LN2, 12);
  });

  it("Newton's method on the five-point example (checked against numpy)", () => {
    // 1-D data x = −2..2, t = (0, 1, 0, 1, 1); the second feature is 0 (a tiny ridge keeps its weight at 0).
    const X = [-2, -1, 0, 1, 2].map((v) => [v, 0]);
    const y = [-1, 1, -1, 1, 1];
    const one = newtonStep([0, 0], 0, X, y, 1e-9);
    expect(one.w[0]).toBeCloseTo(0.8, 6);
    expect(one.b).toBeCloseTo(0.4, 6);
    expect(crossEntropy([0, 0], 0, X, y) * 5).toBeCloseTo(5 * Math.LN2, 10);
    const path = fitLogistic(X, y, 8, 1e-9);
    const f = path.at(-1)!;
    expect(f.w[0]).toBeCloseTo(1.09042556, 6);
    expect(f.b).toBeCloseTo(0.62269007, 6);
    expect(f.loss * 5).toBeCloseTo(2.42196684, 6);
    // Moment matching at the MLE: Σ p_i = Σ t_i.
    const sp = X.reduce((a, x) => a + sigmoid(f.w[0] * x[0] + f.b), 0);
    expect(sp).toBeCloseTo(3, 8);
  });

  it("converges quadratically on the patients and beats the zero model", () => {
    const { X, y } = patients();
    const path = fitLogistic(X, y, 8);
    const g = path.map((p) => p.gradNorm).filter((v) => Number.isFinite(v));
    expect(g.at(-1)!).toBeLessThan(1e-6);
    for (let k = 1; k < path.length; k++) expect(path[k].loss).toBeLessThanOrEqual(path[k - 1].loss + 1e-12);
    expect(accuracy(path.at(-1)!.w, path.at(-1)!.b, X, y)).toBeGreaterThan(0.8);
  });
});

describe("maximum entropy", () => {
  it("mean 3.5 gives the uniform die", () => {
    const r = maxEntDie(3.5);
    expect(r.lambda).toBeCloseTo(0, 9);
    for (const p of r.p) expect(p).toBeCloseTo(1 / 6, 9);
    expect(r.entropy).toBeCloseTo(Math.log(6), 9);
  });

  it("mean 4.5 matches the known exponential-family solution", () => {
    const r = maxEntDie(4.5);
    expect(r.lambda).toBeCloseTo(0.371049, 5);
    [0.054353, 0.078772, 0.11416, 0.165447, 0.239774, 0.347494].forEach((v, k) => expect(r.p[k]).toBeCloseTo(v, 5));
    for (let seed = 1; seed < 20; seed++) {
      const q = sameMeanAlternative(r.p, seed);
      expect(q.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 9);
      expect(q.reduce((a, v, k) => a + v * (k + 1), 0)).toBeCloseTo(4.5, 6);
      expect(entropy(q)).toBeLessThan(r.entropy);
    }
  });
});

describe("margin losses", () => {
  it("all surrogates except the perceptron loss pass through (0, 1) and bound the 0-1 loss", () => {
    for (const k of ["logistic", "hinge", "exponential"] as const) expect(LOSSES[k](0)).toBeCloseTo(1, 12);
    expect(LOSSES.perceptron(0)).toBe(0);
    for (let m = -3; m <= 3; m += 0.25) {
      for (const k of ["logistic", "hinge", "exponential"] as const) expect(LOSSES[k](m)).toBeGreaterThanOrEqual(LOSSES.zeroOne(m) - 1e-12);
      expect(LOSSES.exponential(m)).toBeGreaterThanOrEqual(LOSSES.logistic(m) * Math.LN2 - 1e-12);
    }
  });
});

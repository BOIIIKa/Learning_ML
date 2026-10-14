import { describe, expect, it } from "vitest";
import { adaboost } from "./adaboost";
import { blobs, moons } from "./data";
import { decisionField, kernels, smo } from "./smo";

describe("smo", () => {
  it("satisfies the KKT conditions of the soft-margin dual", () => {
    const { X, y } = blobs();
    const C = 1;
    const { alpha, b, w } = smo(X, y, { C, tol: 1e-4, maxSteps: 5000 });
    expect(Math.abs(alpha.reduce((s, a, k) => s + a * y[k], 0))).toBeLessThan(1e-8);
    X.forEach((x, k) => {
      const margin = y[k] * (w[0] * x[0] + w[1] * x[1] + b);
      expect(alpha[k]).toBeGreaterThanOrEqual(-1e-9);
      expect(alpha[k]).toBeLessThanOrEqual(C + 1e-9);
      if (alpha[k] < 1e-6) expect(margin).toBeGreaterThan(1 - 1e-2);
      else if (alpha[k] > C - 1e-6) expect(margin).toBeLessThan(1 + 1e-2);
      else expect(Math.abs(margin - 1)).toBeLessThan(1e-2);
    });
  });

  it("records one trace step per update and starts from zero", () => {
    const { X, y } = blobs();
    const { steps } = smo(X, y);
    expect(steps[0].w).toEqual([0, 0]);
    expect(steps.length).toBeGreaterThan(5);
    steps.slice(1).forEach((s) => expect(s.highlight).toHaveLength(2));
  });
});

describe("kernel smo", () => {
  it("separates the moons with an RBF kernel", () => {
    const { X, y } = moons(80);
    const kernel = kernels.rbf(2);
    const r = smo(X, y, { C: 10, kernel, maxSteps: 3000 });
    const f = decisionField(X, y, r, kernel, X);
    const acc = f.filter((v, k) => (v >= 0 ? 1 : -1) === y[k]).length / y.length;
    expect(acc).toBeGreaterThan(0.95);
  });
});

describe("adaboost", () => {
  it("reproduces the classic ten-point example", () => {
    const X = [...Array(10).keys()].map((v) => [v, 0]);
    const y = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1];
    const { model, trace } = adaboost(X, y, 3);
    expect(model.map((m) => m.alpha)).toEqual([
      expect.closeTo(0.4236, 3),
      expect.closeTo(0.6496, 3),
      expect.closeTo(0.752, 3),
    ]);
    expect(trace.steps[2].metrics?.error).toBe(0);
  });

  it("keeps training error under the exponential bound", () => {
    const { X, y } = moons();
    const { trace } = adaboost(X, y, 30);
    trace.steps.forEach((s) => expect(s.metrics!.error).toBeLessThanOrEqual(s.metrics!.bound + 1e-12));
  });
});

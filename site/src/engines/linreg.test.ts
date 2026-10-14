import { describe, expect, it } from "vitest";
import { collinearDraws, dot3, lassoCD, logGrid, project3, quadPenalized, regPath, ridge, softThreshold, standardize } from "./linreg";
import { sample } from "./generalization";

describe("linreg", () => {
  it("projection residual is orthogonal to both columns", () => {
    const a1: [number, number, number] = [1, 1, 1];
    const a2: [number, number, number] = [-1, 0, 1];
    const y: [number, number, number] = [1, 2.5, 1.2];
    const { yhat } = project3(a1, a2, y);
    const r = [0, 1, 2].map((i) => y[i] - yhat[i]) as [number, number, number];
    expect(Math.abs(dot3(r, a1))).toBeLessThan(1e-9);
    expect(Math.abs(dot3(r, a2))).toBeLessThan(1e-9);
  });

  it("soft thresholding", () => {
    expect(softThreshold(3, 1)).toBe(2);
    expect(softThreshold(-3, 1)).toBe(-2);
    expect(softThreshold(0.5, 1)).toBe(0);
  });

  it("lasso with λ = 0 matches least squares; large λ zeros everything", () => {
    const s = sample(30, 2);
    const { Z } = standardize(s.xs.map((x) => [x, x * x / 10]));
    const ybar = s.ys.reduce((a, b) => a + b) / s.ys.length;
    const yc = s.ys.map((v) => v - ybar);
    const ols = ridge(Z, yc, 0);
    const l0 = lassoCD(Z, yc, 0, undefined, 5000);
    l0.forEach((v, j) => expect(v).toBeCloseTo(ols[j], 3));
    expect(lassoCD(Z, yc, 1e4).every((v) => v === 0)).toBe(true);
  });

  it("lasso KKT: |grad_j| ≤ λ, with equality on the active set", () => {
    const s = sample(30, 2);
    const { Z } = standardize(s.xs.map((x) => [1, 2, 3, 4, 5].map((k) => ((x - 5) / 5) ** k)));
    const ybar = s.ys.reduce((a, b) => a + b) / s.ys.length;
    const yc = s.ys.map((v) => v - ybar);
    const lam = 0.5;
    const w = lassoCD(Z, yc, lam, undefined, 5000);
    const n = Z.length;
    for (let j = 0; j < w.length; j++) {
      const g = Z.reduce((acc, r, i) => acc + r[j] * (yc[i] - r.reduce((t, v, k) => t + v * w[k], 0)), 0) / n;
      if (w[j] !== 0) expect(Math.abs(Math.abs(g) - lam)).toBeLessThan(1e-4);
      else expect(Math.abs(g)).toBeLessThanOrEqual(lam + 1e-6);
    }
  });

  it("lasso path gets sparser as λ grows; CV is finite", () => {
    const s = sample(30, 2);
    const lams = logGrid(1e-3, 30, 12);
    const { ws, cv } = regPath(s.xs, s.ys, 9, "lasso", lams);
    const nnz = ws.map((w) => w.filter((v) => Math.abs(v) > 1e-10).length);
    expect(nnz[0]).toBe(0);
    expect(nnz[nnz.length - 1]).toBeGreaterThan(2);
    expect(cv.every(Number.isFinite)).toBe(true);
  });

  it("ridge stabilizes collinear estimates", () => {
    const { ols, ridge: r } = collinearDraws(0.95, 0.5);
    const spread = (v: [number, number][]) => v.reduce((s, w) => s + (w[0] - 1) ** 2, 0) / v.length;
    expect(spread(r)).toBeLessThan(spread(ols) / 3);
  });

  it("2-D lasso lands on an axis when the centre is near one", () => {
    const A = [[1, 0.3], [0.3, 0.6]];
    const w = quadPenalized(A, [2, 0.3], "lasso", 1.2);
    expect(w[1]).toBe(0);
    const r = quadPenalized(A, [2, 0.3], "ridge", 1.2);
    expect(Math.abs(r[1])).toBeGreaterThan(0);
  });
});

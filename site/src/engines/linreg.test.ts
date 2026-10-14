import { describe, expect, it } from "vitest";
import {
  camera, cvCurve, fuelData, lambdaGrid, lassoCD2, lassoFit, leverages, pathFor, polyDesign, projectOntoColumns,
  quadForm, quadLoss, regPath, ridgeFit, softThreshold, solveL1Ball, solveL2Ball, type Vec2,
} from "./linreg";

describe("least squares as projection", () => {
  // The worked example of the chapter: x = 1, 2, 3; y = 1, 3, 2.
  const cols = [[1, 1, 1], [1, 2, 3]];
  const y = [1, 3, 2];

  it("solves the three-point example by hand", () => {
    const { w, yhat, e, xte, rss } = projectOntoColumns(cols, y);
    expect(w[0]).toBeCloseTo(1, 12);
    expect(w[1]).toBeCloseTo(0.5, 12);
    [1.5, 2, 2.5].forEach((v, i) => expect(yhat[i]).toBeCloseTo(v, 12));
    [-0.5, 1, -0.5].forEach((v, i) => expect(e[i]).toBeCloseTo(v, 12));
    xte.forEach((v) => expect(v).toBeCloseTo(0, 12));
    expect(rss).toBeCloseTo(1.5, 12);
  });

  it("leverages are the hat-matrix diagonal and sum to p", () => {
    const h = leverages(cols);
    [5 / 6, 1 / 3, 5 / 6].forEach((v, i) => expect(h[i]).toBeCloseTo(v, 12));
    expect(h.reduce((a, b) => a + b, 0)).toBeCloseTo(2, 12);
  });

  it("camera axes are orthonormal and unproject inverts project on the screen plane", () => {
    const cam = camera(0.7, 0.4);
    const d = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0);
    expect(d(cam.right, cam.up)).toBeCloseTo(0, 12);
    expect(d(cam.right, cam.right)).toBeCloseTo(1, 12);
    expect(d(cam.up, cam.up)).toBeCloseTo(1, 12);
    const [sx, sy] = cam.project(cam.unproject(1.3, -0.4));
    expect(sx).toBeCloseTo(1.3, 12);
    expect(sy).toBeCloseTo(-0.4, 12);
  });
});

describe("shrinkage operators", () => {
  it("soft thresholding", () => {
    expect(softThreshold(3, 1)).toBe(2);
    expect(softThreshold(-3, 1)).toBe(-2);
    expect(softThreshold(0.5, 1)).toBe(0);
  });

  it("one centred feature: ridge shrinks by d²/(d²+λ), lasso soft-thresholds (worked example)", () => {
    // Centred x = (−1, 0, 1), y = (−1, 1, 0): Σx² = 2, Σxy = 1. With the 1/(2n) scaling used by ridgeFit,
    // (Σx²/n + λ) w = Σxy/n, so λ = 1/3 here is the same as λ = 1 in ‖y − Xw‖² + λ‖w‖².
    const Z = [[-1], [0], [1]];
    const y = [-1, 1, 0];
    expect(ridgeFit(Z, y, 1 / 3)[0]).toBeCloseTo(1 / 3, 12);
    expect(ridgeFit(Z, y, 0)[0]).toBeCloseTo(0.5, 12);
    // (1/2n)‖y − Zw‖² + λ|w| with λ = 0.5/3 equals ½‖y − Zw‖² + 0.5|w| ⇒ w = S(1, 0.5)/2 = 0.25.
    expect(lassoFit(Z, y, 0.5 / 3).w[0]).toBeCloseTo(0.25, 10);
    expect(lassoFit(Z, y, 1 / 3).w[0]).toBe(0);
  });
});

describe("constrained quadratics on the balls", () => {
  const A = quadForm(0.5, 1.5);
  const brute = (inside: (w: Vec2) => boolean, c: Vec2) => {
    let best = Infinity;
    for (let i = -400; i <= 400; i++) for (let j = -400; j <= 400; j++) {
      const w: Vec2 = [i / 200, j / 200];
      if (inside(w)) best = Math.min(best, quadLoss(A, c, w));
    }
    return best;
  };

  it("L2 ball matches brute force and lies on the circle", () => {
    for (const c of [[1.5, 0.8], [-1.2, 1.4], [0.3, -1.9]] as Vec2[]) {
      const { w } = solveL2Ball(A, c, 1);
      expect(Math.hypot(...w)).toBeCloseTo(1, 6);
      expect(quadLoss(A, c, w)).toBeLessThanOrEqual(brute((v) => Math.hypot(...v) <= 1, c) + 1e-9);
    }
  });

  it("L1 ball matches brute force; a corner solution is sparse", () => {
    for (const c of [[1.5, 0.8], [-1.2, 1.4], [0.3, -1.9], [2, 0.2]] as Vec2[]) {
      const { w } = solveL1Ball(A, c, 1);
      expect(Math.abs(w[0]) + Math.abs(w[1])).toBeCloseTo(1, 9);
      expect(quadLoss(A, c, w)).toBeLessThanOrEqual(brute((v) => Math.abs(v[0]) + Math.abs(v[1]) <= 1, c) + 1e-9);
    }
    const { w, vertex } = solveL1Ball(quadForm(0, 1), [2, 0.5], 1);
    expect(vertex).toBe(true);
    expect(w[0]).toBeCloseTo(1, 9);
    expect(w[1]).toBeCloseTo(0, 9);
  });

  it("2-D coordinate descent reaches the lasso optimum (subgradient conditions)", () => {
    const c: Vec2 = [1.2, 0.3];
    const lam = 0.4;
    const w = lassoCD2(A, c, lam, 60).at(-1)!;
    const g = [A[0][0] * (w[0] - c[0]) + A[0][1] * (w[1] - c[1]), A[1][0] * (w[0] - c[0]) + A[1][1] * (w[1] - c[1])];
    for (let j = 0; j < 2; j++) {
      if (w[j] !== 0) expect(g[j] + lam * Math.sign(w[j])).toBeCloseTo(0, 8);
      else expect(Math.abs(g[j])).toBeLessThanOrEqual(lam + 1e-9);
    }
  });
});

describe("regularization paths on the fuel data", () => {
  const { xs, ys } = fuelData();
  const lams = lambdaGrid();

  it("standardized design has zero-mean unit-variance columns", () => {
    const { Z } = polyDesign(xs);
    for (let j = 0; j < Z[0].length; j++) {
      const col = Z.map((r) => r[j]);
      expect(col.reduce((a, b) => a + b, 0) / col.length).toBeCloseTo(0, 10);
      expect(col.reduce((a, b) => a + b * b, 0) / col.length).toBeCloseTo(1, 10);
    }
  });

  it("lasso coordinate descent satisfies the KKT conditions", () => {
    const { Z } = polyDesign(xs);
    const ybar = ys.reduce((a, b) => a + b, 0) / ys.length;
    const yc = ys.map((v) => v - ybar);
    const lam = 0.02;
    const { w } = lassoFit(Z, yc, lam, undefined, 20000, 1e-12);
    const n = Z.length;
    for (let j = 0; j < w.length; j++) {
      const corr = Z.reduce((s, r, k) => s + r[j] * (yc[k] - r.reduce((t, z, i) => t + z * w[i], 0)), 0) / n;
      if (w[j] !== 0) expect(corr).toBeCloseTo(lam * Math.sign(w[j]), 5);
      else expect(Math.abs(corr)).toBeLessThanOrEqual(lam + 1e-6);
    }
  });

  it("ridge norms shrink monotonically; lasso gets sparser as λ grows", () => {
    const ridge = pathFor(xs, ys, "ridge", lams).ws.map((w) => Math.hypot(...w));
    for (let i = 1; i < ridge.length; i++) expect(ridge[i]).toBeLessThanOrEqual(ridge[i - 1] + 1e-12);
    const nnz = pathFor(xs, ys, "lasso", lams).ws.map((w) => w.filter((v) => Math.abs(v) > 1e-10).length);
    expect(nnz[0]).toBe(9);
    expect(nnz.at(-1)).toBe(0);
  });

  it("cross-validation prefers a moderate λ", () => {
    for (const pen of ["ridge", "lasso"] as const) {
      const cv = cvCurve(xs, ys, pen, lams);
      const best = cv.indexOf(Math.min(...cv));
      expect(best).toBeGreaterThan(0);
      expect(best).toBeLessThan(lams.length - 1);
      expect(cv[0]).toBeGreaterThan(3 * cv[best]);
    }
    const p = regPath("lasso");
    expect(p.train[0]).toBeLessThan(p.train.at(-1)!);
  });
});

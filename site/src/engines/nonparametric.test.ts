import { describe, expect, it } from "vitest";
import { moons } from "./data";
import { sample, truth } from "./generalization";
import { polyFit, mse } from "./generalization";
import { buildKd, contrast, dist, edgeFor, fitSpline, kdNodes, knnError, knnScore, nearest, searchKd, shellFraction } from "./nonparametric";

describe("kNN", () => {
  // Worked example: query at the origin.
  const P = [[2, 0], [1.5, 1.5], [0, 2.2]];
  const Y = [1, -1, -1];

  it("worked example: L1 and L2 pick A as the nearest neighbour, L∞ picks B", () => {
    expect(nearest(P, [0, 0], 1, "l2")).toEqual([0]);
    expect(nearest(P, [0, 0], 1, "l1")).toEqual([0]);
    expect(nearest(P, [0, 0], 1, "linf")).toEqual([1]);
    expect(dist(P[1], [0, 0], "l2")).toBeCloseTo(2.1213, 4);
    expect(knnScore(P, Y, [0, 0], 3, "l2")).toBeCloseTo(-1 / 3, 12);
  });

  it("k = 1 has zero training error; test error is U-shaped in k", () => {
    const { X, y } = moons(150, 1, 0.3);
    const T = moons(600, 2, 0.3);
    expect(knnError(X, y, X, y, 1, "l2")).toBe(0);
    const e1 = knnError(X, y, T.X, T.y, 1, "l2");
    const e15 = knnError(X, y, T.X, T.y, 15, "l2");
    const e149 = knnError(X, y, T.X, T.y, 149, "l2");
    expect(e15).toBeLessThan(e1);
    expect(e15).toBeLessThan(e149);
  });
});

describe("kd-tree", () => {
  const P = [[1, 6], [3, 2], [4, 7], [6, 3], [7, 8], [9, 5]];
  const root = buildKd(P, [0, 10, 0, 10]);

  it("worked example: root (6,3) on x, then (1,6) and (7,8) on y", () => {
    expect(root!.i).toBe(3);
    expect(root!.left!.i).toBe(0);
    expect(root!.right!.i).toBe(4);
    expect(root!.left!.left!.i).toBe(1);
    expect(root!.left!.right!.i).toBe(2);
    expect(root!.right!.left!.i).toBe(5);
  });

  it("worked example: query (2, 6.5) finds (1,6) after visiting 4 nodes and pruning the right subtree", () => {
    const { best, events } = searchKd(root, P, [2, 6.5]);
    expect(best).toBe(0);
    expect(events.filter((e) => e.kind === "visit").map((e) => e.i)).toEqual([3, 0, 2, 1]);
    expect(events.filter((e) => e.kind === "prune").map((e) => e.i)).toEqual([4]);
  });

  it("always finds the true nearest neighbour", () => {
    const { X } = moons(80, 5, 0.3);
    const r = buildKd(X, [-2, 3, -2, 2]);
    expect(kdNodes(r).length).toBe(80);
    for (const q of [[0, 0], [1.5, -0.3], [-1, 1], [2.2, 0.8]]) expect(searchKd(r, X, q).best).toBe(nearest(X, q, 1, "l2")[0]);
  });
});

describe("splines", () => {
  const { xs, ys } = sample(60, 4);
  const T = sample(400, 9);

  it("degrees of freedom = K + 4", () => {
    expect(fitSpline(xs, ys, 5).df).toBe(9);
  });

  it("with the same degrees of freedom a spline tracks the truth better than a polynomial", () => {
    const sp = fitSpline(xs, ys, 8); // 12 parameters
    const po = polyFit(xs, ys, 11, 0); // 12 parameters
    const grid = Array.from({ length: 101 }, (_, i) => 0.2 + (9.6 * i) / 100);
    expect(mse(sp.f, grid, grid.map(truth))).toBeLessThan(mse(po, grid, grid.map(truth)));
    expect(mse(sp.f, T.xs, T.ys)).toBeLessThan(mse(po, T.xs, T.ys));
  });
});

describe("curse of dimensionality", () => {
  it("the near/far distance ratio approaches 1 as d grows", () => {
    expect(contrast(2)).toBeLessThan(0.1);
    expect(contrast(100)).toBeGreaterThan(0.6);
  });
  it("shells and sub-cubes", () => {
    expect(shellFraction(100, 0.05)).toBeCloseTo(1 - 0.95 ** 100, 12);
    expect(shellFraction(100, 0.05)).toBeGreaterThan(0.99);
    expect(edgeFor(0.1, 10)).toBeCloseTo(0.794, 3);
  });
});

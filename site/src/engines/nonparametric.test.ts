import { describe, expect, it } from "vitest";
import { moons, rng } from "./data";
import {
  basisRow, bfsOrder, buildKd, deliveryData, dist, edgeFor, evenKnots, fitBasis, kdPoints, kdSearch, kNearest, knnError, knnPredict,
  knnRegress, minMaxRatio, shellFraction, solve, subtree,
} from "./nonparametric";

// The hand-worked example in the chapter notes.
const EX = { X: [[6.2, 4], [5.5, 5.5], [4, 1.5], [2, 6], [1, 3], [7, 1], [8, 6], [2.5, 1]], y: [1, -1, 1, -1, -1, 1, 1, -1], q: [4, 4] };

describe("knn", () => {
  it("computes L1 / L2 / L∞ distances", () => {
    expect(dist([0, 0], [3, 4], "l1")).toBe(7);
    expect(dist([0, 0], [3, 4], "l2")).toBe(5);
    expect(dist([0, 0], [3, -4], "linf")).toBe(4);
  });

  it("reproduces the worked example: the metric and k both change the answer", () => {
    const { X, y, q } = EX;
    expect(kNearest(X, q, 1, "l1").idx).toEqual([0]);
    expect(kNearest(X, q, 1, "l2").idx).toEqual([1]);
    expect(kNearest(X, q, 3, "l2").idx).toEqual([1, 0, 2]);
    expect(kNearest(X, q, 3, "linf").idx).toEqual([1, 3, 0]);
    expect(knnPredict(X, y, q, 1, "l1")).toBe(1);
    expect(knnPredict(X, y, q, 1, "l2")).toBe(-1);
    expect(knnPredict(X, y, q, 3, "l1")).toBe(1);
    expect(knnPredict(X, y, q, 3, "l2")).toBe(1);
    expect(knnPredict(X, y, q, 3, "linf")).toBe(-1);
  });

  it("1-NN has zero training error; moderate k generalizes better than k = 1 on moons", () => {
    const tr = moons(120, 0, 0.3);
    const te = moons(400, 99, 0.3);
    expect(knnError(tr.X, tr.y, tr.X, tr.y, 1, "l2")).toBe(0);
    expect(knnError(tr.X, tr.y, te.X, te.y, 15, "l2")).toBeLessThan(knnError(tr.X, tr.y, te.X, te.y, 1, "l2"));
  });
});

describe("kd-tree", () => {
  it("splits at the median and alternates axes", () => {
    const nodes = buildKd(EX.X, [0, 10, 0, 10]);
    expect(nodes).toHaveLength(8);
    expect(nodes[0].point).toBe(1); // (5.5, 5.5): x-median of 8 points (index 4 after sorting)
    expect(nodes[nodes[0].left!].point).toBe(4); // (1, 3): y-median of the left half
    expect(nodes[nodes[0].right!].point).toBe(0); // (6.2, 4): y-median of the right half
    expect(nodes[nodes[0].left!].axis).toBe(1);
    expect(bfsOrder(nodes)[0]).toBe(0);
    expect(subtree(nodes, 0).sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
  });

  it("finds the exact nearest neighbour and prunes on average", () => {
    const rand = rng(11);
    const P = Array.from({ length: 300 }, () => [10 * rand(), 10 * rand()]);
    const nodes = buildKd(P, [0, 10, 0, 10]);
    let visitedTotal = 0;
    for (let t = 0; t < 100; t++) {
      const q = [10 * rand(), 10 * rand()];
      const { best, r, events } = kdSearch(nodes, P, q);
      const brute = kNearest(P, q, 1, "l2");
      expect(r).toBeCloseTo(brute.d[0], 12);
      expect(dist(P[best], q, "l2")).toBeCloseTo(brute.d[0], 12);
      visitedTotal += events.at(-1)!.visited.length;
    }
    expect(visitedTotal / 100).toBeLessThan(40);
  });

  it("visited and pruned subtrees partition the tree", () => {
    const P = kdPoints();
    const nodes = buildKd(P, [0, 10, 0, 10]);
    const last = kdSearch(nodes, P, [8, 2]).events.at(-1)!;
    const covered = [...last.visited, ...last.pruned.flatMap((id) => subtree(nodes, id))].sort((a, b) => a - b);
    expect(covered).toEqual(nodes.map((n) => n.id));
  });
});

describe("curse of dimensionality", () => {
  it("the nearest / farthest ratio grows towards 1 with d", () => {
    const r1 = minMaxRatio(1);
    const r10 = minMaxRatio(10);
    const r1000 = minMaxRatio(1000);
    expect(r1).toBeLessThan(0.05);
    expect(r10).toBeGreaterThan(r1);
    expect(r1000).toBeGreaterThan(0.85);
  });

  it("shell fraction and sub-cube edge", () => {
    expect(shellFraction(1, 0.1)).toBeCloseTo(0.1, 12);
    expect(shellFraction(50, 0.05)).toBeCloseTo(1 - 0.95 ** 50, 12);
    expect(edgeFor(0.01, 10)).toBeCloseTo(0.630957, 5);
  });
});

describe("basis expansions", () => {
  it("solves a linear system", () => {
    const x = solve([[2, 1], [1, 3]], [3, 5]);
    expect(x[0]).toBeCloseTo(0.8, 12);
    expect(x[1]).toBeCloseTo(1.4, 12);
  });

  it("truncated power basis has M + 1 + K columns", () => {
    expect(basisRow(5, "cubic", 0, evenKnots(4)).row).toHaveLength(8);
    expect(basisRow(5, "linear", 0, evenKnots(3)).row).toHaveLength(5);
    expect(basisRow(5, "poly", 6, []).row).toHaveLength(7);
  });

  it("unpenalized df equals the number of basis functions; λ shrinks it", () => {
    const { xs, ys } = deliveryData();
    const f0 = fitBasis(xs, ys, "cubic", 0, evenKnots(5), 0);
    expect(f0.df).toBeCloseTo(9, 2); // up to the tiny stabilizing ridge
    const f1 = fitBasis(xs, ys, "cubic", 0, evenKnots(5), 1);
    expect(f1.df).toBeLessThan(9);
    expect(f1.df).toBeGreaterThan(4 - 1e-6);
  });

  it("reproduces a cubic exactly and a linear spline is continuous at its knots", () => {
    const xs = Array.from({ length: 30 }, (_, i) => i / 3);
    const g = (x: number) => 1 + 0.5 * x - 0.2 * x * x + 0.03 * x ** 3;
    const f = fitBasis(xs, xs.map(g), "cubic", 0, evenKnots(3), 0);
    for (const x of [0.4, 3.3, 7.7]) expect(f.predict(x)).toBeCloseTo(g(x), 5);
    const { xs: dx, ys: dy } = deliveryData();
    const lin = fitBasis(dx, dy, "linear", 0, evenKnots(4), 0);
    for (const k of evenKnots(4)) expect(lin.predict(k - 1e-7)).toBeCloseTo(lin.predict(k + 1e-7), 4);
  });

  it("kNN regression with k = n is the global mean", () => {
    const { xs, ys } = deliveryData();
    const mean = ys.reduce((a, b) => a + b, 0) / ys.length;
    expect(knnRegress(xs, ys, 3, xs.length)).toBeCloseTo(mean, 10);
  });
});

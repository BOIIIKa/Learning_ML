import { describe, expect, it } from "vitest";
import { moons } from "./data";
import { accuracy, bestSplit, candidateSplits, depthOf, fitStump1D, forest, forestProba, gini, gradientBoost1D, growTree } from "./tree";

describe("cart", () => {
  it("finds the obvious split and its exact Gini gain", () => {
    const X = [[0, 0], [1, 0], [2, 0], [3, 0]];
    const y = [-1, -1, 1, 1];
    const s = bestSplit(X, y, [0, 1, 2, 3])!;
    expect(s.feature).toBe(0);
    expect(s.threshold).toBe(1.5);
    expect(s.gain).toBeCloseTo(gini(0.5), 12);
  });

  it("never reports a negative gain", () => {
    const { X, y } = moons(60, 2);
    candidateSplits(X, y, X.map((_, i) => i)).forEach((s) => expect(s.gain).toBeGreaterThanOrEqual(-1e-12));
  });

  it("deeper trees fit the training data better and respect maxDepth", () => {
    const { X, y } = moons(120, 0);
    const accs = [1, 3, 8].map((d) => {
      const t = growTree(X, y, undefined, { maxDepth: d });
      expect(depthOf(t)).toBeLessThanOrEqual(d);
      return accuracy(t, X, y);
    });
    expect(accs[0]).toBeLessThan(accs[1]);
    expect(accs[1]).toBeLessThanOrEqual(accs[2]);
    expect(accs[2]).toBeGreaterThan(0.97);
  });

  it("a forest generalizes at least as well as one deep tree on fresh data", () => {
    const train = moons(120, 0, 0.3);
    const test = moons(400, 99, 0.3);
    const tree = growTree(train.X, train.y, undefined, { maxDepth: 10 });
    const trees = forest(train.X, train.y, 60, 3, 10);
    const forestAcc = test.X.filter((x, i) => (forestProba(trees, x) >= 0.5 ? 1 : -1) === test.y[i]).length / test.X.length;
    expect(forestAcc).toBeGreaterThanOrEqual(accuracy(tree, test.X, test.y));
  });
});

describe("gradient boosting 1-D", () => {
  it("fits the least-squares stump", () => {
    const s = fitStump1D([0, 1, 2, 3], [1, 1, 5, 5]);
    expect(s).toEqual({ threshold: 1.5, left: 1, right: 5 });
  });

  it("training MSE never increases", () => {
    const xs = [...Array(40).keys()].map((i) => i / 4);
    const ys = xs.map((x) => Math.sin(x) + 0.1 * Math.cos(7 * x));
    const { stages } = gradientBoost1D(xs, ys, 30);
    for (let m = 1; m < stages.length; m++) expect(stages[m].mse).toBeLessThanOrEqual(stages[m - 1].mse + 1e-12);
    expect(stages.at(-1)!.mse).toBeLessThan(stages[0].mse / 5);
  });
});

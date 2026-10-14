import { describe, expect, it } from "vitest";
import { agglomerate, chooseK, cut, lloyd, seedPlusPlus, seedRandom, silhouette, subgroups, wcss } from "./clustering";

describe("k-means", () => {
  it("worked example: from centres 1 and 2 the objective falls 165 → 59.2 → 39.04 → 33.25 → 25.4375 → 56/3", () => {
    const X = [[1], [2], [4], [7], [8], [12]];
    const run = lloyd(X, [[1], [2]]);
    const objs = run.map((s) => s.obj);
    [165, 165, 59.2, 39.04, 33.25, 25.4375, 56 / 3].forEach((v, i) => expect(objs[i]).toBeCloseTo(v, 9));
    const last = run.at(-1)!;
    expect(last.C.map((c) => c[0])).toEqual([7 / 3, 9]);
    expect(last.lab).toEqual([0, 0, 0, 1, 1, 1]);
  });

  it("every half-step never increases the objective", () => {
    const { X } = subgroups();
    for (const seed of [1, 2, 3, 4]) {
      const objs = lloyd(X, seedRandom(X, 3, seed)).map((s) => s.obj);
      for (let i = 1; i < objs.length; i++) expect(objs[i]).toBeLessThanOrEqual(objs[i - 1] + 1e-9);
    }
  });

  it("k-means++ seeding ends lower than random seeding on average", () => {
    const { X } = subgroups();
    const avg = (f: (s: number) => number[][]) => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].reduce((s, sd) => s + lloyd(X, f(sd)).at(-1)!.obj, 0) / 10;
    expect(avg((s) => seedPlusPlus(X, 3, s))).toBeLessThanOrEqual(avg((s) => seedRandom(X, 3, s)) + 1e-9);
  });

  it("some random seed gets stuck in a worse local optimum than k-means++", () => {
    const { X } = subgroups();
    const best = Math.min(...[1, 2, 3, 4, 5].map((s) => lloyd(X, seedPlusPlus(X, 3, s)).at(-1)!.obj));
    const worst = Math.max(...Array.from({ length: 30 }, (_, s) => lloyd(X, seedRandom(X, 3, s)).at(-1)!.obj));
    expect(worst).toBeGreaterThan(best * 1.2);
  });
});

describe("hierarchical clustering", () => {
  const X = [[1], [2], [4], [7], [11]];
  it("worked example: single linkage merges at 1, 2, 3, 4 (chaining)", () => {
    expect(agglomerate(X, "single").map((m) => m.h)).toEqual([1, 2, 3, 4]);
  });
  it("worked example: complete linkage merges at 1, 3, 6, 10", () => {
    expect(agglomerate(X, "complete").map((m) => m.h)).toEqual([1, 3, 6, 10]);
    expect(cut(5, agglomerate(X, "complete"), 4)).toEqual([0, 0, 1, 1, 2]);
  });
});

describe("choosing k", () => {
  it("silhouette peaks at the true k = 3; WCSS keeps falling", () => {
    const { X } = subgroups();
    const ks = [2, 3, 4, 5, 6];
    const { wcss: w, sil } = chooseK(X, ks);
    for (let i = 1; i < w.length; i++) expect(w[i]).toBeLessThan(w[i - 1]);
    expect(ks[sil.indexOf(Math.max(...sil))]).toBe(3);
  });
  it("silhouette of a perfect split is high", () => {
    expect(silhouette([[0], [0.1], [10], [10.1]], [0, 0, 1, 1])).toBeGreaterThan(0.95);
    expect(wcss([[0], [2]], [0, 0], [[1]])).toBe(2);
  });
});

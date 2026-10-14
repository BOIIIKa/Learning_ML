import { describe, expect, it } from "vitest";
import { sample } from "./generalization";
import { forward } from "./mlp";
import { countPieces, deepTentNet, deepTentParams, knotsFor, maxRegions2d, patternGrid, randomUnits, regionsInWindow, relu, shallowFit, shallowParams, tent, tentPow } from "./neural-networks";

describe("chapter 14: ReLU networks", () => {
  it("worked example: 1 + 2 relu(x) − 3 relu(x−1) + 2 relu(x−2) has 4 pieces and values 1, 3, 2, 3", () => {
    const f = (x: number) => 1 + 2 * relu(x) - 3 * relu(x - 1) + 2 * relu(x - 2);
    expect([0, 1, 2, 3].map(f)).toEqual([1, 3, 2, 3]);
    expect(countPieces(f, -1, 3)).toBe(4);
  });

  it("a shallow net with D hinge units has at most D + 1 pieces, and fits better as D grows", () => {
    const { xs, ys } = sample(40, 11);
    const mse = (D: number) => { const m = shallowFit(xs, ys, knotsFor(D)); return xs.reduce((s, x, i) => s + (m.f(x) - ys[i]) ** 2, 0) / xs.length; };
    for (const D of [1, 2, 3, 5, 8, 12]) expect(countPieces(shallowFit(xs, ys, knotsFor(D)).f, -1, 11)).toBeLessThanOrEqual(D + 1);
    expect(mse(12)).toBeLessThan(mse(4));
    expect(mse(4)).toBeLessThan(mse(1));
  });

  it("D lines make at most 1 + D + C(D,2) regions; the default layer (seed 12) reaches it, the grid sees no more", () => {
    expect([1, 2, 3, 4, 5, 6].map(maxRegions2d)).toEqual([2, 4, 7, 11, 16, 22]);
    for (const D of [1, 2, 3, 4, 5, 6, 8]) {
      const units = randomUnits(D, 4);
      const exact = regionsInWindow(units, 2.5);
      expect(exact).toBeLessThanOrEqual(maxRegions2d(D));
      expect(patternGrid(units, 2.5, 160).distinct).toBeLessThanOrEqual(exact);
    }
    for (const D of [2, 3, 4, 5]) expect(regionsInWindow(randomUnits(D, 12), 2.5)).toBe(maxRegions2d(D));
  });

  it("composing the tent map L times gives 2^L pieces; the deep net computes exactly t∘…∘t", () => {
    expect(tent(0.25)).toBe(0.5);
    expect(tentPow(2, 0.25)).toBe(1);
    for (const L of [1, 2, 3, 4, 5]) {
      expect(countPieces((x) => tentPow(L, x), 0, 1, 8192)).toBe(2 ** L);
      const net = deepTentNet(L);
      for (const x of [0.1, 0.37, 0.5, 0.81]) expect(forward(net, [x]).hs.at(-1)![0]).toBeCloseTo(tentPow(L, x), 12);
    }
  });

  it("same parameters: deep 2^L pieces vs shallow ≤ 2L + 1 (depth wins from L = 3)", () => {
    for (const L of [1, 2, 3, 4, 5]) expect(deepTentParams(L)).toBe(shallowParams(2 * L));
    expect(2 ** 3).toBeGreaterThan(2 * 3 + 1);
    expect(2 ** 2).toBeLessThan(2 * 2 + 1);
    // The widget's shallow competitor: 2L hinge units least-squares fitted to the sawtooth.
    const xs = Array.from({ length: 801 }, (_, i) => i / 800);
    for (const L of [1, 2, 3, 4, 5, 6]) {
      const f = shallowFit(xs, xs.map((x) => tentPow(L, x)), knotsFor(2 * L, 0, 1)).f;
      expect(countPieces(f, 0, 1, 8192)).toBeLessThanOrEqual(2 * L + 1);
    }
  });
});

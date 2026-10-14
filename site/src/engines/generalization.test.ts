import { describe, expect, it } from "vitest";
import {
  biasVariance, cohort, crossValidate, cvSimulation, doubleDescent, hoeffdingEps, hoursDesign, kfoldSplits, legendre, linspace,
  meanSd, mse, polyFit, polyPredict, randomFeatures, rfFit, rfPredict, toU,
} from "./generalization";

describe("polynomial fits", () => {
  it("Legendre polynomials match their closed forms", () => {
    const [p0, p1, p2, p3] = legendre(0.5, 3);
    expect([p0, p1]).toEqual([1, 0.5]);
    expect(p2).toBeCloseTo((3 * 0.25 - 1) / 2, 12); // −0.125
    expect(p3).toBeCloseTo((5 * 0.125 - 3 * 0.5) / 2, 12); // −0.4375
  });

  it("least squares recovers a noise-free cubic exactly", () => {
    const xs = hoursDesign(9);
    const g = (h: number) => 1 + 2 * toU(h) - toU(h) ** 3;
    const w = polyFit(xs, xs.map(g), 3);
    for (const h of [0.7, 3.3, 8.8]) expect(polyPredict(w, h)).toBeCloseTo(g(h), 9);
  });
});

describe("bias–variance", () => {
  it("fixed design: the variance averaged over the training inputs is σ²(d+1)/n", () => {
    const n = 12;
    const sigma = 8;
    for (const d of [1, 3, 6]) {
      const bv = biasVariance(d, n, sigma, 4000, 2, hoursDesign(n));
      expect(bv.variance / ((sigma * sigma * (d + 1)) / n)).toBeGreaterThan(0.94);
      expect(bv.variance / ((sigma * sigma * (d + 1)) / n)).toBeLessThan(1.06);
    }
  });

  it("bias² falls and variance rises with the degree", () => {
    const grid = linspace(0.5, 9.5, 60);
    const runs = [0, 1, 2, 3, 4].map((d) => biasVariance(d, 12, 8, 300, 1, grid));
    for (let d = 1; d < runs.length; d++) {
      expect(runs[d].bias2).toBeLessThan(runs[d - 1].bias2);
      expect(runs[d].variance).toBeGreaterThan(runs[d - 1].variance);
    }
  });
});

describe("cross-validation", () => {
  it("k folds partition the indices into nearly equal parts", () => {
    const folds = kfoldSplits(10, 3, 1);
    expect(folds.map((f) => f.length)).toEqual([4, 3, 3]);
    expect(folds.flat().sort((a, b) => a - b)).toEqual([...Array(10).keys()]);
  });

  it("leave-one-out of the mean is (n/(n−1))² × training MSE — the worked example gives 76", () => {
    const ys = [52, 60, 64, 70];
    const xs = [1, 2, 3, 4];
    const loo = crossValidate(xs, ys, 0, 4, 1).estimate;
    const m = ys.reduce((a, b) => a + b, 0) / 4;
    const train = mse(ys.map(() => m), ys);
    expect(train).toBeCloseTo(42.75, 10);
    expect(loo).toBeCloseTo(76, 10);
    expect(loo).toBeCloseTo((4 / 3) ** 2 * train, 10);
  });

  it("small k is pessimistic and noisier; LOO is nearly unbiased", () => {
    const sim = cvSimulation(24, 8, 3, [2, 24], 120, 5);
    const [k2, loo] = sim.estimates.map(meanSd);
    expect(k2.mean).toBeGreaterThan(loo.mean);
    expect(k2.sd).toBeGreaterThan(loo.sd);
    expect(Math.abs(loo.mean - sim.trueError) / sim.trueError).toBeLessThan(0.1);
  });
});

describe("double descent", () => {
  const data = cohort(20, 4, 1);

  it("beyond p = n the minimum-norm fit interpolates the training set", () => {
    const m = rfFit(randomFeatures(100, 3), data.xs, data.ys);
    expect(mse(data.xs.map((x) => rfPredict(m, x)), data.ys)).toBeLessThan(1e-6);
  });

  it("test error peaks near p = n and falls again; ridge removes the peak", () => {
    const curve = doubleDescent(data.xs, data.ys, 4, 0, [12, 22, 1000], 9);
    const [under, peak, over] = curve.map((c) => c.test);
    expect(peak).toBeGreaterThan(10 * under);
    expect(peak).toBeGreaterThan(10 * over);
    const ridge = doubleDescent(data.xs, data.ys, 4, 0.01, [22], 9)[0].test;
    expect(ridge).toBeLessThan(peak / 10);
  });
});

it("Hoeffding + union bound: |H| = 1000, n = 1000, δ = 0.05 gives ε ≈ 0.0728", () => {
  expect(hoeffdingEps(1000, 1000, 0.05)).toBeCloseTo(0.07279, 5);
});

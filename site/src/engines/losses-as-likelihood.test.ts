import { describe, expect, it } from "vitest";
import { DISTS, avgNll, coverage, crossEntropy, entropy, fitHetero, fitHomo, heteroData, heteroTruth, kl, softplus } from "./losses-as-likelihood";

describe("chapter 15: losses as likelihood", () => {
  it("each loss's derivative matches finite differences, and its minimiser predicts the observed mean", () => {
    const cases: [keyof typeof DISTS, number][] = [["gaussian", 1.7], ["bernoulli", 1], ["bernoulli", 0], ["categorical", 0], ["categorical", 1], ["laplace", -0.4], ["poisson", 3]];
    for (const [d, y] of cases) for (const f of [-1.3, 0.2, 2.1]) {
      const h = 1e-6, num = (DISTS[d].nll(y, f + h) - DISTS[d].nll(y, f - h)) / (2 * h);
      expect(DISTS[d].grad(y, f)).toBeCloseTo(num, 5);
    }
    // Gradient zero ⇔ predicted mean = observation (Gaussian, Poisson; Bernoulli with a soft label).
    expect(DISTS.gaussian.grad(1.7, 1.7)).toBe(0);
    expect(DISTS.poisson.grad(3, Math.log(3))).toBeCloseTo(0, 12);
    expect(DISTS.bernoulli.grad(0.3, Math.log(0.3 / 0.7))).toBeCloseTo(0, 12);
    // A hard label (y = 1, or the correct class) has no finite minimiser: the loss keeps falling.
    for (let f = -5; f < 5; f += 0.5) {
      expect(DISTS.bernoulli.nll(1, f + 0.5)).toBeLessThan(DISTS.bernoulli.nll(1, f));
      expect(DISTS.categorical.nll(0, f + 0.5)).toBeLessThan(DISTS.categorical.nll(0, f));
    }
  });

  it("worked numbers: BCE 0.1269 / 2.1269 at logit 2, softmax CE 0.4076, Gaussian NLL 3.8084, KL 0.0823", () => {
    expect(DISTS.bernoulli.nll(1, 2)).toBeCloseTo(0.126928, 6);
    expect(DISTS.bernoulli.nll(0, 2)).toBeCloseTo(2.126928, 6);
    expect(softplus(2) - softplus(-2)).toBeCloseTo(2, 12);
    const ce = -2 + Math.log(Math.exp(2) + Math.exp(1) + 1);
    expect(ce).toBeCloseTo(0.407606, 6);
    expect((78 - 70) ** 2 / (2 * 25) + Math.log(5 * Math.sqrt(2 * Math.PI))).toBeCloseTo(3.808376, 6);
    expect(entropy(0.7)).toBeCloseTo(0.610864, 6);
    expect(crossEntropy(0.7, 0.5)).toBeCloseTo(0.693147, 6);
    expect(kl(0.7, 0.5)).toBeCloseTo(0.082283, 6);
  });

  it("cross-entropy = entropy + KL, KL ≥ 0, and cross-entropy is smallest at q = p", () => {
    for (const p of [0.1, 0.3, 0.7, 0.95]) for (const q of [0.05, 0.2, 0.5, 0.8, 0.99]) {
      expect(crossEntropy(p, q)).toBeCloseTo(entropy(p) + kl(p, q), 12);
      expect(kl(p, q)).toBeGreaterThanOrEqual(0);
      expect(crossEntropy(p, q)).toBeGreaterThanOrEqual(crossEntropy(p, p) - 1e-12);
    }
  });

  it("the heteroscedastic fit recovers the growing spread, beats the constant-σ fit, and covers both halves", () => {
    const { xs, ys } = heteroData();
    const homo = fitHomo(xs, ys), het = fitHetero(xs, ys);
    expect(avgNll(het, xs, ys)).toBeLessThan(avgNll(homo, xs, ys) - 0.1);
    expect(het.sd(9)).toBeGreaterThan(2.5 * het.sd(1));
    expect(Math.abs(het.sd(5) - heteroTruth.sd(5)) / heteroTruth.sd(5)).toBeLessThan(0.3);
    const ch = coverage(homo, xs, ys), ce = coverage(het, xs, ys);
    expect(ch.low).toBe(1); // a constant band is far too wide where the noise is small …
    expect(ch.high).toBeLessThan(0.93); // … and too narrow where it is large
    expect(ce.low).toBeGreaterThan(0.9);
    expect(ce.high).toBeGreaterThan(0.9);
  });
});

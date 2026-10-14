import { describe, expect, it } from "vitest";
import { biasVariance, cvError, doubleDescentCurve, folds, legendre, mse, polyFit, sample } from "./generalization";

describe("generalization", () => {
  it("legendre matches closed forms", () => {
    const [p0, p1, p2, p3] = legendre(3, 0.3);
    expect([p0, p1]).toEqual([1, 0.3]);
    expect(p2).toBeCloseTo((3 * 0.09 - 1) / 2, 12);
    expect(p3).toBeCloseTo((5 * 0.027 - 3 * 0.3) / 2, 12);
  });

  it("training error falls with degree; degree n−1 interpolates", () => {
    const { xs, ys } = sample(10, 1);
    const errs = [0, 2, 5, 9].map((d) => mse(polyFit(xs, ys, d), xs, ys));
    for (let i = 1; i < errs.length; i++) expect(errs[i]).toBeLessThanOrEqual(errs[i - 1] + 1e-9);
    expect(errs[3]).toBeLessThan(0.1);
  });

  it("bias falls and variance rises with degree", () => {
    const lo = biasVariance(1, 25);
    const hi = biasVariance(5, 25);
    expect(hi.bias2).toBeLessThan(lo.bias2);
    expect(hi.variance).toBeGreaterThan(lo.variance);
  });

  it("every index lands in exactly one fold and CV picks a moderate degree", () => {
    const f = folds(30, 5);
    expect(new Set(f).size).toBe(5);
    const { xs, ys } = sample(30, 2);
    const cv = [1, 3, 5, 9].map((d) => cvError(xs, ys, d, f, 5));
    expect(cv[1]).toBeLessThan(cv[0]);
    expect(cv[1]).toBeLessThan(cv[3]);
  });

  it("double descent: test error peaks near p = n and falls again", () => {
    const { ps, errs } = doubleDescentCurve(30, 100, 12);
    const at = (p: number) => errs[ps.indexOf(p)];
    expect(at(30)).toBeGreaterThan(3 * at(10));
    expect(at(30)).toBeGreaterThan(3 * at(100));
    expect(Math.max(...errs)).toBe(Math.max(at(29), at(30), at(31)));
  });
});

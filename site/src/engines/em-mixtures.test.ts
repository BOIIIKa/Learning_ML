import { describe, expect, it } from "vitest";
import { elbo1, emRun, loglik1, mStep1, oneD, resp1, shapedGroups, softAssign, type Gmm } from "./em-mixtures";

describe("worked example (1-D, unit variances, equal weights)", () => {
  it("responsibilities σ(2x − 2) and one M-step to (−0.4875, 3.2683)", () => {
    const xs = [-1, 0, 3, 4];
    const r2 = xs.map((x) => softAssign(x, 0, 2, 1));
    [0.018, 0.1192, 0.982, 0.9975].forEach((v, i) => expect(r2[i]).toBeCloseTo(v, 3));
    const m1 = xs.reduce((s, x, i) => s + (1 - r2[i]) * x, 0) / r2.reduce((s, r) => s + 1 - r, 0);
    const m2 = xs.reduce((s, x, i) => s + r2[i] * x, 0) / r2.reduce((s, r) => s + r, 0);
    expect(m1).toBeCloseTo(-0.4875, 3);
    expect(m2).toBeCloseTo(3.2683, 3);
  });
});

describe("EM for a 2-D Gaussian mixture", () => {
  const X = shapedGroups();
  const g0: Gmm = { w: [1 / 3, 1 / 3, 1 / 3], mu: [[-0.5, 0.5], [0.5, 0.5], [0, -0.5]], S: [[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]] };
  const { ll, r } = emRun(X, g0, 60);
  it("the log-likelihood never decreases", () => {
    for (let t = 1; t < ll.length; t++) expect(ll[t]).toBeGreaterThanOrEqual(ll[t - 1] - 1e-8);
  });
  it("responsibilities sum to one", () => {
    r.at(-1)!.forEach((ri) => expect(ri.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12));
  });
});

describe("the ELBO picture (1-D θ)", () => {
  const xs = oneD();
  it("the ELBO touches log p at θ_t and lies below it elsewhere", () => {
    const th = 0.8;
    const q = resp1(xs, th);
    expect(elbo1(xs, q, th)).toBeCloseTo(loglik1(xs, th), 9);
    for (const t of [-1, 0, 0.5, 1.5, 2.5, 4]) expect(elbo1(xs, q, t)).toBeLessThanOrEqual(loglik1(xs, t) + 1e-9);
  });
  it("the M-step maximizes the ELBO and raises log p", () => {
    let th = 0.5;
    for (let k = 0; k < 15; k++) {
      const q = resp1(xs, th);
      const next = mStep1(xs, q);
      for (const d of [-0.1, 0.1]) expect(elbo1(xs, q, next)).toBeGreaterThanOrEqual(elbo1(xs, q, next + d));
      expect(loglik1(xs, next)).toBeGreaterThanOrEqual(loglik1(xs, th) - 1e-12);
      th = next;
    }
    expect(th).toBeGreaterThan(2);
  });
});

describe("hard-assignment limit", () => {
  it("as σ² → 0 responsibilities become 0/1 by nearest centre (k-means)", () => {
    expect(softAssign(1.2, 0, 2, 1e-3)).toBeCloseTo(1, 9);
    expect(softAssign(0.8, 0, 2, 1e-3)).toBeCloseTo(0, 9);
    expect(softAssign(1.2, 0, 2, 4)).toBeLessThan(0.6);
  });
});

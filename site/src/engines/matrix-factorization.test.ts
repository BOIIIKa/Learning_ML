import { describe, expect, it } from "vitest";
import { correlated, jacobiEig, lowRank, panel, pca, principalAngle, projStats, scanImage, svd } from "./matrix-factorization";

describe("PCA in 2-D", () => {
  it("worked example: Σ = [[2,1],[1,2]] has eigenvalues 3, 1 along (1,1)/√2", () => {
    const { values, vectors } = jacobiEig([[2, 1], [1, 2]]);
    expect(values[0]).toBeCloseTo(3, 12);
    expect(values[1]).toBeCloseTo(1, 12);
    expect(Math.abs(vectors[0][0])).toBeCloseTo(Math.SQRT1_2, 12);
    expect(values[0] / (values[0] + values[1])).toBeCloseTo(0.75, 12);
  });
  it("projected variance + reconstruction error = total variance, for every direction", () => {
    const P = correlated();
    for (const t of [0, 0.3, 1, 2]) {
      const s = projStats(P, t);
      expect(s.proj + s.resid).toBeCloseTo(s.total, 12);
    }
  });
  it("the principal angle maximizes the projected variance", () => {
    const P = correlated();
    const t0 = principalAngle(P);
    for (const d of [-0.2, -0.05, 0.05, 0.2]) expect(projStats(P, t0).proj).toBeGreaterThan(projStats(P, t0 + d).proj);
  });
});

describe("SVD and Eckart–Young", () => {
  const A = scanImage();
  const d = svd(A);
  const fro2 = (M: number[][]) => M.reduce((s, r) => s + r.reduce((t, v) => t + v * v, 0), 0);
  it("full rank reconstructs the image", () => {
    const R = lowRank(d, d.s.length);
    expect(Math.sqrt(fro2(A.map((r, i) => r.map((v, j) => v - R[i][j]))))).toBeLessThan(1e-8);
  });
  it("the rank-k error equals the tail of the singular values", () => {
    for (const k of [1, 3, 8]) {
      const R = lowRank(d, k);
      const err2 = fro2(A.map((r, i) => r.map((v, j) => v - R[i][j])));
      expect(err2).toBeCloseTo(d.s.slice(k).reduce((s, v) => s + v * v, 0), 6);
    }
  });
  it("singular values are sorted and non-negative", () => {
    for (let i = 1; i < d.s.length; i++) expect(d.s[i]).toBeLessThanOrEqual(d.s[i - 1] + 1e-9);
  });
});

describe("scree", () => {
  it("two latent factors: the first two components explain most of the variance", () => {
    const { values } = pca(panel().X);
    const tot = values.reduce((s, v) => s + v, 0);
    expect((values[0] + values[1]) / tot).toBeGreaterThan(0.8);
    expect(values[2] / tot).toBeLessThan(0.05);
  });
});

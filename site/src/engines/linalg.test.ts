import { describe, expect, it } from "vitest";
import { eigSym, fromSvd, gaussianLogLik, gaussianMle, mul, projectHalfplane, svd2, type Mat } from "./linalg";

const close = (A: Mat, B: Mat) => A.flat().forEach((v, i) => expect(v).toBeCloseTo(B.flat()[i], 9));

describe("linalg", () => {
  it("eigSym satisfies S v = λ v", () => {
    const S: Mat = [[2, 0.8], [0.8, 0.5]];
    const { values, vectors } = eigSym(S);
    vectors.forEach((v, k) => mul(S, v).forEach((x, i) => expect(x).toBeCloseTo(values[k] * v[i], 12)));
    expect(values[0]).toBeGreaterThanOrEqual(values[1]);
  });

  it.each([
    [[[1, 0], [0, 1]]], [[[2, 1], [0, 1]]], [[[0, -1], [1, 0]]], [[[1, 2], [2, 4]]], [[[1, 0], [0, -1]]], [[[-1.5, 0.3], [0.7, 0.2]]],
  ] as [Mat][])("svd2 reconstructs %j", (A) => {
    const s = svd2(A);
    expect(s.sigma[0]).toBeGreaterThanOrEqual(Math.abs(s.sigma[1]) - 1e-12);
    close(fromSvd(s), A);
  });

  it("MLE maximizes the log-likelihood", () => {
    const X = [[0, 0], [1, 2], [2, 1], [3, 3], [1, 1]];
    const { mean, cov } = gaussianMle(X);
    const best = gaussianLogLik(X, mean, cov);
    for (const [dm, dc] of [[0.1, 0], [0, 0.1], [-0.2, -0.05]]) {
      expect(gaussianLogLik(X, [mean[0] + dm, mean[1]], [[cov[0][0] + dc, cov[0][1]], [cov[1][0], cov[1][1] + dc]])).toBeLessThan(best);
    }
  });

  it("half-plane projection satisfies KKT", () => {
    const a: [number, number] = [1, 2];
    const { x, lambda, active } = projectHalfplane([3, 3], a, 2);
    expect(active).toBe(true);
    expect(a[0] * x[0] + a[1] * x[1]).toBeCloseTo(2, 12);
    // stationarity 2(x − t) + λa = 0
    expect(2 * (x[0] - 3) + lambda * a[0]).toBeCloseTo(0, 12);
    expect(2 * (x[1] - 3) + lambda * a[1]).toBeCloseTo(0, 12);
    expect(projectHalfplane([0, 0], a, 2)).toEqual({ x: [0, 0], lambda: 0, active: false });
  });
});

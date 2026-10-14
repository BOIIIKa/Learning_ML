import { describe, expect, it } from "vitest";
import {
  covFromAxes, fitClasses, gaussianNbLinear, lda2, learningCurves, linearRuleError, logisticNewton, nbLogScores, normCdf,
  placeGaussian, qda2, redundantPopulation, sigmoid, softmaxLog, standardNormals, wordProbs,
} from "./generative";
import type { Vec } from "./linalg";

// A small two-class set; reference log-odds come from scikit-learn's LinearDiscriminantAnalysis /
// QuadraticDiscriminantAnalysis decision_function (identical to the maximum-likelihood formulas).
const X = [[0, 0], [1, 2], [2, 1], [1, 0.5], [3, 3], [4, 2], [3, 4], [5, 4.5], [4, 3.5]];
const y = [-1, -1, -1, -1, 1, 1, 1, 1, 1];
const pts: Vec[] = [[2, 2], [0, 3], [4, 0]];

describe("Gaussian discriminant analysis", () => {
  it("fits priors, means and MLE covariances", () => {
    const [c0, c1] = fitClasses(X, y);
    expect(c0.prior).toBeCloseTo(4 / 9, 12);
    expect(c1.mean[0]).toBeCloseTo(3.8, 12);
    expect(c0.cov[0][0]).toBeCloseTo(0.5, 12); // xs 0,1,2,1: mean 1, squared deviations 1,0,1,0
  });

  it("LDA log-odds match scikit-learn", () => {
    const { w, b } = lda2(X, y);
    const ref = [-1.82513231, -7.87685645, 1.74383321];
    pts.forEach((p, i) => expect(w[0] * p[0] + w[1] * p[1] + b).toBeCloseTo(ref[i], 6));
  });

  it("QDA log-odds match scikit-learn", () => {
    const f = qda2(X, y);
    const ref = [-1.9982574, -4.2945537, 6.77893864];
    pts.forEach((p, i) => expect(f(p)).toBeCloseTo(ref[i], 6));
  });

  it("QDA reduces to LDA when the two classes share a covariance", () => {
    const z = standardNormals(200, 3);
    const A = placeGaussian(z, [-1, 0], 1.2, 0.5, 0.4);
    const B = placeGaussian(z, [1.5, 1], 1.2, 0.5, 0.4); // same shape, shifted: identical sample covariances
    const XX = [...A, ...B];
    const yy = [...A.map(() => -1), ...B.map(() => 1)];
    const { w, b } = lda2(XX, yy);
    const f = qda2(XX, yy);
    for (const p of [[0, 0], [2, -1], [-3, 2]] as Vec[]) expect(f(p)).toBeCloseTo(w[0] * p[0] + w[1] * p[1] + b, 8);
  });

  it("covFromAxes has the requested eigen-structure", () => {
    const C = covFromAxes(2, 0.5, Math.PI / 6);
    expect(C[0][0] + C[1][1]).toBeCloseTo(4.25, 12);
    expect(C[0][0] * C[1][1] - C[0][1] * C[1][0]).toBeCloseTo(1, 12);
  });
});

describe("logistic regression", () => {
  it("Newton's method matches scikit-learn LogisticRegression(C = 1/λ)", () => {
    const XX = [...X, [2.5, 2.5]];
    const yy = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0];
    const { w, b } = logisticNewton(XX, yy, 0.5);
    expect(w[0]).toBeCloseTo(1.42741039, 5);
    expect(w[1]).toBeCloseTo(0.95874449, 5);
    expect(b).toBeCloseTo(-6.1315142, 5);
  });
});

describe("multinomial naive Bayes", () => {
  // Words: fever, pain, refill, schedule. Rows: urgent, routine (the chapter's worked example).
  const counts = [[4, 3, 0, 1], [1, 1, 4, 6]];
  const priors = [2 / 5, 3 / 5];
  const msg = [1, 1, 1, 0];

  it("Laplace smoothing gives (N_kj + 1)/(N_k + V)", () => {
    expect(wordProbs(counts, 1)[0]).toEqual([5 / 12, 4 / 12, 1 / 12, 2 / 12]);
    expect(wordProbs(counts, 0)[0][2]).toBe(0);
  });

  it("reproduces the worked example: P(urgent | fever pain refill) = 0.6124 with α = 1", () => {
    expect(softmaxLog(nbLogScores(counts, priors, 1, msg))[0]).toBeCloseTo(1024 / 1672, 10);
  });

  it("without smoothing one unseen word vetoes the class", () => {
    const s = nbLogScores(counts, priors, 0, msg);
    expect(s[0]).toBe(-Infinity);
    expect(softmaxLog(s)).toEqual([0, 1]);
  });
});

describe("generative vs discriminative learning curves", () => {
  it("normCdf and sigmoid are accurate", () => {
    expect(normCdf(1.959964)).toBeCloseTo(0.975, 6);
    expect(sigmoid(0)).toBe(0.5);
    expect(sigmoid(-800)).toBe(0);
  });

  it("the exact linear-rule error matches the 1-D formula Φ(−Δ/2)", () => {
    expect(linearRuleError([1], -0.5, [1], [[1]])).toBeCloseTo(normCdf(-0.5), 10);
  });

  it("naive Bayes is asymptotically optimal only without redundant features", () => {
    for (const rho of [0, 0.6]) {
      const { mu, S } = redundantPopulation(rho);
      const nbW = mu; // diag(S) = I, so naive Bayes in the limit uses w = mu
      const nbErr = linearRuleError(nbW, -mu.reduce((s, v) => s + v * v, 0) / 2, mu, S);
      const c = learningCurves(rho, 2, [512]);
      expect(c.nbInf).toBeCloseTo(nbErr, 10);
      if (rho === 0) expect(c.nbInf).toBeCloseTo(c.bayes, 8);
      else expect(c.nbInf - c.bayes).toBeGreaterThan(0.02);
    }
  });

  it("the curves cross: naive Bayes wins with little data, logistic regression with a lot", () => {
    const c = learningCurves(0.6);
    expect(c.sizes[1]).toBe(16);
    expect(c.nb[1]).toBeLessThan(c.lr[1]); // m = 16
    expect(c.nb.at(-1)!).toBeGreaterThan(c.lr.at(-1)! + 0.01); // m = 512
    expect(c.lr.at(-1)!).toBeGreaterThan(c.bayes);
  });

  it("Gaussian naive Bayes with shared variances is LDA with a diagonal covariance", () => {
    const XX = [[0, 1], [1, 0], [2, 2], [3, 1], [4, 3], [5, 5]];
    const yy = [0, 0, 0, 1, 1, 1];
    const { w } = gaussianNbLinear(XX, yy);
    // class means (1, 1) and (4, 3); pooled (shared) per-feature variances:
    const v0 = (1 + 0 + 1 + 1 + 0 + 1) / 6;
    const v1 = (0 + 1 + 1 + 4 + 0 + 4) / 6;
    expect(w[0]).toBeCloseTo(3 / v0, 6);
    expect(w[1]).toBeCloseTo(2 / v1, 6);
  });
});

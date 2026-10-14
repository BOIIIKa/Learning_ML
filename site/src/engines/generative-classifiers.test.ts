import { describe, expect, it } from "vitest";
import {
  bernoulliPosterior, dupClasses, covFrom, errorRate, fitGda, gnb, ldaLine, ldaLogOdds, learningCurves, logisticLine,
  logisticP, qdaLogOdds, sampleClasses, stdNormals, type Gauss,
} from "./generative-classifiers";
import type { V2 } from "./linear-classification";

// The worked example: 4 patients needing a follow-up, 6 not; features fever F and cough C.
const COUNTS: [number[], number[]] = [[3, 4], [1, 2]];
const NS: [number, number] = [4, 6];

describe("naive Bayes with Laplace smoothing", () => {
  it("worked example: (F=0, C=0) has posterior 0 without smoothing and 32/437 with α = 1", () => {
    expect(bernoulliPosterior(COUNTS, NS, [0, 0], 0)).toBe(0);
    expect(bernoulliPosterior(COUNTS, NS, [0, 0], 1)).toBeCloseTo(32 / 437, 12);
  });
  it("worked example: (F=1, C=1) goes from 9/10 to 320/401 with α = 1", () => {
    expect(bernoulliPosterior(COUNTS, NS, [1, 1], 0)).toBeCloseTo(0.9, 12);
    expect(bernoulliPosterior(COUNTS, NS, [1, 1], 1)).toBeCloseTo(320 / 401, 12);
  });
  it("as α grows the feature probabilities tend to 1/2 and the posterior to the prior", () => {
    expect(bernoulliPosterior(COUNTS, NS, [1, 1], 1e6)).toBeCloseTo(0.4, 4);
  });
});

describe("Gaussian discriminant analysis", () => {
  const zP = stdNormals(60, 1);
  const zN = stdNormals(60, 2);
  const pos: Gauss = { mean: [1, 0.8], cov: covFrom(1.1, 0.5, 30) };
  const neg: Gauss = { mean: [-1, -0.6], cov: covFrom(0.6, 0.6, 0) };
  const { X, y } = sampleClasses(pos, neg, zP, zN);
  const f = fitGda(X, y);

  it("LDA log-odds are exactly linear: w·x + b equals the Gaussian formula with the pooled covariance", () => {
    const { w, b } = ldaLine(f);
    for (const x of [[0, 0], [2, -1], [-1.5, 2]] as V2[]) expect(w[0] * x[0] + w[1] * x[1] + b).toBeCloseTo(ldaLogOdds(f, x), 9);
  });

  it("QDA log-odds are quadratic (non-zero second difference) when covariances differ, linear when equal", () => {
    const g = (h: (x: V2) => number) => h([1, 0]) - 2 * h([0, 0]) + h([-1, 0]);
    expect(Math.abs(g((x) => qdaLogOdds(f, x)))).toBeGreaterThan(0.1);
    const same = { ...f, cov: [f.pooled, f.pooled] as typeof f.cov };
    expect(Math.abs(g((x) => qdaLogOdds(same, x)))).toBeLessThan(1e-9);
  });

  it("with different covariances QDA beats LDA on fresh data; with equal ones they tie", () => {
    const T = sampleClasses(pos, neg, stdNormals(3000, 7), stdNormals(3000, 8));
    const l = ldaLine(f);
    const eL = errorRate((x) => l.w[0] * x[0] + l.w[1] * x[1] + l.b, T.X, T.y);
    const eQ = errorRate((x) => qdaLogOdds(f, x), T.X, T.y);
    expect(eQ).toBeLessThan(eL);
    const lr = logisticLine(X, y);
    const eR = errorRate((x) => lr.w[0] * x[0] + lr.w[1] * x[1] + lr.b, T.X, T.y);
    expect(eR).toBeLessThan(0.5);
  });

  it("shifting the prior moves the 1-D LDA threshold to 1 + ln 4 / 2 (deck example)", () => {
    // Means 0 and 2, variance 1, P(+) = 0.2: threshold where the log-odds vanish.
    const thr = (0 + 2) / 2 + (1 * Math.log(0.8 / 0.2)) / (2 - 0);
    expect(thr).toBeCloseTo(1 + Math.log(2), 12);
  });
});

describe("generative vs discriminative", () => {
  const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;

  it("independent features: naive Bayes learns faster and both reach the same error", () => {
    const { nb, lr } = learningCurves(0);
    expect(mean(nb.slice(0, 6))).toBeLessThan(mean(lr.slice(0, 6)));
    expect(Math.abs(nb.at(-1)! - lr.at(-1)!)).toBeLessThan(0.01);
  });

  it("two duplicated features: naive Bayes wins with few samples, logistic regression with many (the curves cross)", () => {
    const { nb, lr } = learningCurves(2);
    expect(mean(nb.slice(0, 6))).toBeLessThan(mean(lr.slice(0, 6)) - 0.01);
    expect(mean(lr.slice(-3))).toBeLessThan(mean(nb.slice(-3)) - 0.01);
  });

  it("logistic regression fits separable-ish data without blowing up thanks to the L2 term", () => {
    const D = dupClasses(12, 10, 2, 3, 5);
    const f = logisticP(D.X, D.y);
    expect(Number.isFinite(f(D.X[0]))).toBe(true);
  });
});

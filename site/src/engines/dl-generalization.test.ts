import { describe, expect, it } from "vitest";
import {
  ES, GROK, LAND, MN, SNAP_EVERY, alphaFor, firstStep, gdAt, gdFilter, gdUnder, ridgeAt, ridgeFilter, ringLoss, trainLab, truthMse, underLimit,
} from "./dl-generalization";

describe("chapter 18: the regulariser lab", () => {
  const none = trainLab("wd", 0);
  it("without a regulariser the network memorises the 12 students and the test error climbs", () => {
    expect(none.train.at(-1)!).toBeLessThan(3); // caption: 2.4
    expect(none.train.at(-1)!).toBeCloseTo(2.4, 1);
    expect(none.test.at(-1)!).toBeCloseTo(184, 0);
    // Early stopping: the lowest test error comes at step 50, at 62.
    expect(none.best * SNAP_EVERY).toBe(50);
    expect(none.test[none.best]).toBeCloseTo(62, 0);
    expect(truthMse).toBeCloseTo(55, 0); // the noise floor
  });
  it("each regulariser brings the final test error down; too much weight decay underfits", () => {
    const wd3 = trainLab("wd", 3), wd10 = trainLab("wd", 4), drop = trainLab("dropout", 3), aug = trainLab("augment", 2);
    expect(wd3.test.at(-1)!).toBeCloseTo(61, 0);
    expect(drop.test.at(-1)!).toBeCloseTo(68, 0);
    expect(aug.test.at(-1)!).toBeCloseTo(64, 0);
    for (const r of [wd3, drop, aug]) expect(r.train.at(-1)!).toBeGreaterThan(10); // no longer memorised
    expect(wd10.train.at(-1)!).toBeGreaterThan(250);
    expect(wd10.test.at(-1)!).toBeCloseTo(287, 0);
  });
  it("is deterministic", () => {
    expect(trainLab("dropout", 2).test).toEqual(trainLab("dropout", 2).test);
  });
});

describe("chapter 18: early stopping ≈ ridge", () => {
  it("the filters agree at both ends and differ in between (worked example: η = 0.1, t = 100)", () => {
    const eta = 0.1, t = 100, alpha = 1 / (eta * t);
    expect(gdFilter(0.001, eta, t)).toBeCloseTo(0.00995, 5);
    expect(ridgeFilter(0.001, alpha)).toBeCloseTo(0.0099, 4);
    expect(gdFilter(0.1, eta, t)).toBeCloseTo(0.634, 3);
    expect(ridgeFilter(0.1, alpha)).toBeCloseTo(0.5, 12);
    expect(gdFilter(10, eta, t)).toBe(1);
    expect(ridgeFilter(10, alpha)).toBeCloseTo(0.990, 3);
  });
  it("the widget's numbers: t = 50 gives α = 0.04 and 0.636 / 0.5 along λ = 0.04", () => {
    expect(alphaFor(50)).toBeCloseTo(0.04, 12);
    expect(gdFilter(ES.lam[1], ES.eta, 50)).toBeCloseTo(0.636, 3);
    expect(ridgeFilter(ES.lam[1], alphaFor(50))).toBeCloseTo(0.5, 12);
  });
  it("the two paths nearly coincide; matched by α = 1/(ηt) the points differ by at most a quarter of ‖w*‖", () => {
    const norm = Math.hypot(...ES.wStar);
    const ridge = Array.from({ length: 20001 }, (_, i) => ridgeAt(10 ** (4 - (8 * i) / 20000)));
    let path = 0, matched = 0;
    for (let t = 0; t <= 400; t++) {
      const g = gdAt(t);
      path = Math.max(path, Math.min(...ridge.map((r) => Math.hypot(g[0] - r[0], g[1] - r[1]))) / norm);
      if (t > 0) { const r = ridgeAt(alphaFor(t)); matched = Math.max(matched, Math.hypot(g[0] - r[0], g[1] - r[1]) / norm); }
    }
    expect(path).toBeLessThan(0.11);
    expect(matched).toBeLessThan(0.25);
    expect(matched).toBeGreaterThan(0.2);
    // Both end at w*.
    expect(Math.hypot(gdAt(400)[0] - ES.wStar[0], gdAt(400)[1] - ES.wStar[1])).toBeLessThan(1e-3);
    expect(Math.hypot(ridgeAt(1e-6)[0] - ES.wStar[0], ridgeAt(1e-6)[1] - ES.wStar[1])).toBeLessThan(1e-3);
  });
});

describe("chapter 18: gradient descent on an underdetermined problem", () => {
  it("from the origin it converges to the minimum-norm solution (0.6, 1.2)", () => {
    const end = gdUnder([0, 0], 200).at(-1)!;
    expect(end[0]).toBeCloseTo(0.6, 9);
    expect(end[1]).toBeCloseTo(1.2, 9);
    expect(Math.hypot(...end)).toBeCloseTo(1.342, 3);
  });
  it("from any start it walks a straight line along a to the nearest solution", () => {
    for (const w0 of [[2, 2], [2.5, 2.5], [-1.5, 0.5]] as [number, number][]) {
      const path = gdUnder(w0, 200), lim = underLimit(w0);
      expect(path.at(-1)![0]).toBeCloseTo(lim[0], 9);
      // every step moves parallel to a = (1, 2): (w − w0) × a = 0
      for (const w of path) expect((w[0] - w0[0]) * MN.a[1] - (w[1] - w0[1]) * MN.a[0]).toBeCloseTo(0, 12);
      expect(MN.a[0] * lim[0] + MN.a[1] * lim[1]).toBeCloseTo(MN.y, 12);
    }
    expect(underLimit([2, 2])[0]).toBeCloseTo(1.4, 12);
    expect(underLimit([2, 2])[1]).toBeCloseTo(0.8, 12);
  });
});

describe("chapter 18: precomputed loss landscapes", () => {
  const c = (LAND.K - 1) / 2, half = c / 2;
  it("has a K × K grid per run, three seeds per configuration", () => {
    for (const runs of Object.values(LAND.runs)) {
      expect(runs).toHaveLength(3);
      for (const r of runs) { expect(r.loss).toHaveLength(LAND.K); r.loss.forEach((row) => expect(row).toHaveLength(LAND.K)); }
    }
  });
  it("skip connections: tens to hundreds at distance 1 without them, about 2–3 with them (every seed)", () => {
    LAND.runs.plain.forEach((p, s) => {
      const r = LAND.runs.residual[s];
      expect(ringLoss(p.loss, c)).toBeGreaterThan(50);
      expect(ringLoss(p.loss, c)).toBeLessThan(600);
      expect(ringLoss(r.loss, c)).toBeGreaterThan(1.9);
      expect(ringLoss(r.loss, c)).toBeLessThan(3);
    });
  });
  it("the widget and course quote seed 0: 55.8 without skips, 2.8 with them", () => {
    expect(ringLoss(LAND.runs.plain[0].loss, c)).toBeCloseTo(55.8, 1);
    expect(ringLoss(LAND.runs.residual[0].loss, c)).toBeCloseTo(2.8, 1);
  });
  it("small batches end flatter than full batch, at nearly the same test accuracy (every seed)", () => {
    LAND.runs.small.forEach((sm, s) => {
      const lg = LAND.runs.large[s];
      expect(ringLoss(lg.loss, half)).toBeGreaterThan(4 * ringLoss(sm.loss, half));
      expect(ringLoss(lg.loss, c)).toBeGreaterThan(2.5 * ringLoss(sm.loss, c));
      expect(Math.abs(sm.testAcc - lg.testAcc)).toBeLessThan(0.02);
      expect(sm.testAcc).toBeGreaterThan(0.94);
      expect(lg.testAcc).toBeGreaterThan(0.94);
    });
  });
});

describe("chapter 18: precomputed grokking curves", () => {
  const byWd = (wd: number) => GROK.runs[GROK.wds.indexOf(wd)];
  it("has the documented setup and aligned arrays", () => {
    expect(GROK.p).toBe(31);
    expect(GROK.nTrain).toBe(480);
    expect(GROK.wds).toEqual([0, 0.1, 0.3, 1, 3]);
    for (const r of GROK.runs) for (const k of ["train", "test", "norm"] as const) expect(r[k]).toHaveLength(r.step.length);
  });
  it("every run memorises the training sums within 100 steps", () => {
    for (const r of GROK.runs) expect(firstStep(r, "train", 0.99)).toBeLessThanOrEqual(100);
  });
  it("without weight decay the test accuracy stays near 0 and the weights grow", () => {
    for (const r of [byWd(0), ...GROK.seeds.wd0]) {
      expect(Math.max(...r.test)).toBeLessThan(0.05);
      expect(r.norm.at(-1)!).toBeGreaterThan(r.norm[0]);
    }
  });
  it("with weight decay 1 the test accuracy reaches 95% thousands of steps later, as the norm falls (every seed)", () => {
    const r = byWd(1);
    expect(firstStep(r, "train", 0.99)).toBe(69);
    expect(firstStep(r, "test", 0.5)).toBeGreaterThan(2000);
    expect(firstStep(r, "test", 0.5)).toBeLessThan(3000);
    expect(Math.round(firstStep(r, "test", 0.95) / 100) * 100).toBe(9200);
    expect(firstStep(r, "test", 0.95)).toBeGreaterThan(5000);
    expect(firstStep(r, "test", 0.95)).toBeLessThan(12000);
    expect(r.norm.at(-1)!).toBeLessThan(0.8 * r.norm[0]);
    for (const s of GROK.seeds.wd1) {
      expect(firstStep(s, "test", 0.95)).toBeLessThan(GROK.steps);
      expect(firstStep(s, "test", 0.95) / firstStep(s, "train", 0.99)).toBeGreaterThan(50);
    }
  });
  it("too little or too much decay: 0.1 has not grokked by 20k steps, 3 stalls below 70%", () => {
    expect(firstStep(byWd(0.1), "test", 0.95)).toBe(Infinity);
    expect(byWd(0.1).test.at(-1)!).toBeCloseTo(0.74, 2);
    expect(byWd(3).test.at(-1)!).toBeCloseTo(0.63, 2);
    expect(firstStep(byWd(0.3), "test", 0.95)).toBeLessThan(GROK.steps);
  });
});

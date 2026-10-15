import { describe, expect, it } from "vitest";
import { backward, forward, initNet } from "./mlp";
import {
  DIAL_STEPS, QUAD_START, detCos, RACERS, SCHED_PEAKS, SCHED_SIZES, SCHEDULES, SURFACES, design, dial, flatGrad, flatNet, gdRate, heavyBall,
  lrAt, moonData, optimalEta, optimizer, quadGrad, quadRun, race, run, stepsTo, trainSchedule,
} from "./optimization";

describe("chapter 16: optimization", () => {
  it("GD on a stretched quadratic contracts by max|1 − ηλ| per step; the best step is 2/(L + μ)", () => {
    for (const k of [10, 100]) {
      // η = 1/L kills the stiff coordinate in one step; the slow one then shrinks by 1 − 1/κ per step.
      const n0 = Math.hypot(...QUAD_START);
      expect(stepsTo(quadRun(k, 1, "gd", 2000))).toBe(Math.ceil(Math.log((1e-3 * n0) / 3) / Math.log(gdRate(k, 1))));
      // With 2/(L + μ) both coordinates shrink by the same factor, so the whole distance does.
      expect(stepsTo(quadRun(k, optimalEta(k), "gd", 2000))).toBe(Math.ceil(Math.log(1e-3) / Math.log(gdRate(k, optimalEta(k)))));
      expect(gdRate(k, optimalEta(k))).toBeCloseTo((k - 1) / (k + 1), 12);
      // No step size beats 2/(L + μ).
      for (let e = 0.05; e < 2; e += 0.05) expect(gdRate(k, e)).toBeGreaterThanOrEqual(gdRate(k, optimalEta(k)) - 1e-12);
    }
    // Worked example (κ = 10): 66 steps with η = 1/L, 35 with the best η, 15 with the best heavy ball.
    expect(stepsTo(quadRun(10, 1, "gd", 2000))).toBe(66);
    expect(stepsTo(quadRun(10, optimalEta(10), "gd", 2000))).toBe(35);
    expect(stepsTo(quadRun(10, 0, "momentum", 2000))).toBe(15);
    // The widget's "η = 2/(L+μ)" button rounds to the slider's 0.01 grid: same counts.
    expect(stepsTo(quadRun(10, +optimalEta(10).toFixed(2), "gd", 2000))).toBe(35);
    expect(stepsTo(quadRun(100, +optimalEta(100).toFixed(2), "gd", 2000))).toBe(346);
    // κ = 100: 683 → 346 → 52; momentum's rate (√κ − 1)/(√κ + 1) is the GD rate of √κ.
    expect([stepsTo(quadRun(100, 1, "gd", 2000)), stepsTo(quadRun(100, optimalEta(100), "gd", 2000)), stepsTo(quadRun(100, 0, "momentum", 2000))]).toEqual([683, 346, 52]);
    expect(heavyBall(10).eta).toBeCloseTo(2.309, 3);
    expect(heavyBall(10).beta).toBeCloseTo(0.270, 3);
    expect(heavyBall(100).rate).toBeCloseTo(gdRate(10, optimalEta(10)), 12);
    // Above 2/L the stiff direction blows up.
    expect(Math.abs(quadRun(10, 2.05, "gd", 300).at(-1)![1])).toBeGreaterThan(Math.abs(QUAD_START[1]));
  });

  it("strong convexity: GD with η = 1/L shrinks f − f* by at least (1 − μ/L) per step", () => {
    const k = 25, f = ([x, y]: number[]) => 0.5 * (x * x / k + y * y);
    const path = run("gd", { lr: 1 }, quadGrad(k), [2, -1.5], 50);
    for (let t = 0; t + 1 < path.length; t++) expect(f(path[t + 1])).toBeLessThanOrEqual((1 - 1 / k) * f(path[t]) + 1e-15);
  });

  it("Adam's bias correction makes the first step exactly the learning rate; without it the step is 3.16× too big", () => {
    for (const g of [0.01, 100]) {
      const x1 = optimizer("adam", { lr: 0.1, eps: 0 })([0], () => [g]);
      expect(x1[0]).toBeCloseTo(-0.1, 12);
      const raw = optimizer("adam", { lr: 0.1, eps: 0, bias: false })([0], () => [g]);
      expect(raw[0] / x1[0]).toBeCloseTo(0.1 / Math.sqrt(0.001), 9);
    }
    // With a constant gradient, m_t = (1 − β₁ᵗ) g before correction.
    let m = 0;
    for (let t = 1; t <= 7; t++) { m = 0.9 * m + 0.1 * 3; expect(m).toBeCloseTo((1 - 0.9 ** t) * 3, 12); }
  });

  it("AdamW decays weights outside the adaptive ratio; inside Adam an L2 penalty loses its strength", () => {
    // Zero loss gradient: AdamW shrinks by (1 − ηλ) per step.
    let x = [1];
    const step = optimizer("adamw", { lr: 0.01, wd: 0.1 });
    for (let t = 0; t < 10; t++) x = step(x, () => [0]);
    expect(x[0]).toBeCloseTo((1 - 0.001) ** 10, 12);
    // L2 inside Adam: the penalty gradient λw is divided by √v ≈ |λw|, so λ drops out — a 100× stronger
    // penalty gives the same step, of size lr.
    for (const lam of [0.001, 0.1]) {
      const y = optimizer("adam", { lr: 0.01, eps: 0 })([1], (w) => [lam * w[0]]);
      expect(y[0]).toBeCloseTo(0.99, 12);
    }
    // AdamW's shrink is proportional to λ.
    const a = optimizer("adamw", { lr: 0.01, wd: 0.001 })([1], () => [0]), b = optimizer("adamw", { lr: 0.01, wd: 0.1 })([1], () => [0]);
    expect((1 - b[0]) / (1 - a[0])).toBeCloseTo(100, 6);
  });

  it("momentum and Nesterov follow their definitions", () => {
    const g = (x: number[]) => [2 * x[0]];
    const mom = optimizer("momentum", { lr: 0.1, beta: 0.5 });
    const x1 = mom([1], g), x2 = mom(x1, g);
    expect(x1[0]).toBeCloseTo(0.8, 12); // v = −0.2
    expect(x2[0]).toBeCloseTo(0.8 + (0.5 * -0.2 - 0.1 * 1.6), 12);
    const nes = optimizer("nesterov", { lr: 0.1, beta: 0.5 });
    const y1 = nes([1], g), y2 = nes(y1, g);
    expect(y1[0]).toBeCloseTo(0.8, 12);
    expect(y2[0]).toBeCloseTo(0.8 + (0.5 * -0.2 - 0.1 * 2 * (0.8 - 0.1)), 12); // gradient at the look-ahead point
    // A constant gradient: the velocity tends to −η g/(1 − β), a 1/(1 − β) longer step.
    const c = optimizer("momentum", { lr: 0.1, beta: 0.9 });
    let x = [0], prev = 0;
    for (let t = 0; t < 300; t++) { prev = x[0]; x = c(x, () => [1]); }
    expect(x[0] - prev).toBeCloseTo(-0.1 / (1 - 0.9), 9);
  });

  it("the race: what each widget caption claims, at the default learning rates", () => {
    const at = (k: keyof typeof SURFACES) => Object.fromEntries(race(SURFACES[k], 1).map((r) => [r.name, r]));
    const conv = (fs: number[], fmin: number) => fs.findIndex((f) => f - fmin < 1e-3);
    // Valley: GD crawls along the floor; momentum and Nesterov get there first.
    const v = at("valley");
    expect(conv(v.gd.fs, 0)).toBeGreaterThan(3 * conv(v.momentum.fs, 0));
    expect(conv(v.nesterov.fs, 0)).toBeLessThan(conv(v.momentum.fs, 0));
    // Rosenbrock: GD does not reach the minimum in 150 steps; the others do.
    const r = at("rosenbrock");
    expect(conv(r.gd.fs, 0)).toBe(-1);
    for (const n of ["momentum", "nesterov", "adam"]) expect(conv(r[n].fs, 0)).toBeGreaterThan(0);
    // Saddle: GD lingers near the origin; Adam's per-coordinate scaling escapes first.
    const s = at("saddle");
    const esc = (p: number[][]) => p.findIndex((q) => Math.abs(q[1]) > 0.5);
    const e = Object.fromEntries(RACERS.map((n) => [n, esc(s[n].path)]));
    expect(e.gd).toBeGreaterThan(60);
    expect(e.adam).toBe(10);
    expect(e.gd).toBe(67); // worked example 3: the y-coordinate alone, 1.1× per step from 0.001
    expect(e.adam).toBeLessThan(e.nesterov);
    expect(e.nesterov).toBeLessThan(e.gd);
    // ×4 on the valley: Nesterov blows up first, GD bounces across the stiff direction for ever.
    const big = Object.fromEntries(race(SURFACES.valley, 4).map((r) => [r.name, r]));
    expect(big.nesterov.path.length).toBeLessThan(151);
    expect(big.gd.fs.at(-1)!).toBeGreaterThan(0.5);
    expect(big.momentum.fs.at(-1)!).toBeLessThan(1e-3);
    const banana = Object.fromEntries(race(SURFACES.rosenbrock, 4).map((r) => [r.name, r]));
    expect(banana.nesterov.path.length).toBeLessThan(151);
    for (const n of ["gd", "momentum", "adam"]) { expect(banana[n].path.length).toBe(151); expect(big[n].path.length).toBe(151); }
    // Every surface's gradient matches finite differences.
    for (const S of Object.values(SURFACES)) for (const p of [[0.3, -0.7], [-1.1, 0.4]]) {
      const h = 1e-6, g = S.grad(p);
      expect(g[0]).toBeCloseTo((S.f([p[0] + h, p[1]]) - S.f([p[0] - h, p[1]])) / (2 * h), 5);
      expect(g[1]).toBeCloseTo((S.f([p[0], p[1] + h]) - S.f([p[0], p[1] - h])) / (2 * h), 5);
    }
  });

  it("the scenario: raw hours make a badly conditioned bowl; standardising fixes it", () => {
    const raw = design(false), std = design(true);
    expect(raw.kappa).toBeGreaterThan(150);
    expect(std.kappa).toBeCloseTo(1, 9);
    expect(raw.best).toBeCloseTo(std.best, 9); // same model, same best loss
    // Raw: GD is stable only below 2/L ≈ 0.059, and even the best stable rate is far from done after 100 steps.
    expect(2 / raw.L).toBeLessThan(0.06);
    expect(dial(false, 0.06).diverged || dial(false, 0.06).losses.at(-1)! > 1e4).toBe(true);
    expect(dial(false, 0.05).losses[DIAL_STEPS]).toBeGreaterThan(1.25 * raw.best);
    expect(dial(false, 0.05).hit).toBe(-1);
    // The largest stable slider value (10^−1.25 ≈ 0.056) still leaves the intercept well below 24.4.
    const best = dial(false, 10 ** -1.25);
    expect(best.diverged).toBe(false);
    expect(best.path[DIAL_STEPS][0]).toBeLessThan(18);
    expect(raw.wStar[0]).toBeCloseTo(24.36, 2);
    expect(dial(false, 10 ** -1.2).losses[DIAL_STEPS]).toBeGreaterThan(1e6);
    // The deck's "too small" beat (η = 10^−2.5 ≈ 0.003): loss still ~3× the best, intercept barely moved.
    const slow = dial(false, 10 ** -2.5);
    expect(slow.losses[DIAL_STEPS] / raw.best).toBeGreaterThan(2.8);
    expect(slow.losses[DIAL_STEPS] / raw.best).toBeLessThan(3.1);
    expect(slow.path[DIAL_STEPS][0]).toBeLessThan(4); // vs 24.4
    // Standardised: η = 1 lands on the least-squares line in one step.
    expect(dial(true, 1).hit).toBe(1);
    expect(dial(true, 2.1).diverged || dial(true, 2.1).losses.at(-1)! > 1e4).toBe(true);
  });

  it("the flat MLP's gradient equals mlp.ts backpropagation", () => {
    const net = initNet(SCHED_SIZES, 4, 1);
    const p = flatNet(net);
    const x = moonData.X[5];
    const cache = forward(net, x, "tanh");
    const f = cache.hs.at(-1)![0];
    const ref = backward(net, cache, [1 / (1 + Math.exp(-f)) - 1], "tanh");
    const g = new Float64Array(p.length);
    flatGrad(p, x, 1, g);
    expect(Array.from(g).every((v, i) => Math.abs(v - flatNet(ref)[i]) < 1e-12)).toBe(true);
  });

  it("schedules: decay helps at a large peak rate, warm-up tames the first epochs, decay hurts at a small one", () => {
    for (let x = 0; x <= Math.PI + 1e-9; x += 0.01) expect(Math.abs(detCos(x) - Math.cos(x))).toBeLessThan(1e-14);
    expect(lrAt("step", 1, 0, 300)).toBe(1);
    expect(lrAt("step", 1, 100, 300)).toBeCloseTo(0.1, 12);
    expect(lrAt("cosine", 1, 150, 300)).toBeCloseTo(0.5, 12);
    expect(lrAt("warmup", 1, 0, 300)).toBeCloseTo(1 / 30, 12);
    expect(lrAt("warmup", 1, 29, 300)).toBeCloseTo(1, 12);
    const runs = Object.fromEntries(SCHED_PEAKS.map((pk) => [pk, Object.fromEntries(SCHEDULES.map((s) => [s, trainSchedule(s, pk)]))]));
    const big = runs[0.3], small = runs[0.03];
    expect(big.constant.at(-1)!).toBeGreaterThan(5 * big.cosine.at(-1)!);
    expect(big.warmup.at(-1)!).toBeLessThan(big.constant.at(-1)! / 5);
    expect(Math.max(...big.warmup.slice(1, 6))).toBeLessThan(0.7 * Math.max(...big.constant.slice(1, 6)));
    expect(small.constant.at(-1)!).toBeLessThan(small.step.at(-1)!);
    expect(small.constant.at(-1)!).toBeLessThan(small.cosine.at(-1)!);
  });
});

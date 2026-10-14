import { describe, expect, it } from "vitest";
import {
  LOSS_IDS, overlapping, ceGrad, ceHessian, crossEntropy, dualToPrimal, entropy, fromPolar, gdIterations, loss, maxentDie, mistakes,
  newton, newtonIterations, novikoff, perceptron, pull, screening, shuffled, toPolar, twoFaceDie, type Theta, type V2,
} from "./linear-classification";

// The worked example of the notes, deck and recap.
const EX: V2[] = [[1, 3], [4, 1], [1, 0]];
const EY = [1, 1, -1];

describe("perceptron", () => {
  it("reproduces the worked example: 6 updates to w = (1, 4), b = −2, α = (1, 1, 4)", () => {
    const { steps, converged } = perceptron(EX, EY);
    expect(converged).toBe(true);
    expect(steps.slice(1).map((s) => s.i)).toEqual([0, 2, 2, 1, 2, 2]);
    expect(steps.slice(1).map((s) => [...s.w, s.b])).toEqual([[1, 3, 1], [0, 3, 0], [-1, 3, -1], [3, 4, 0], [2, 4, -1], [1, 4, -2]]);
    expect(steps.at(-1)!.alpha).toEqual([1, 1, 4]);
  });

  it("the example's Novikoff bound is R²/γ² = 18 · 221/81 = 442/9 ≥ 6", () => {
    const { R, gamma, bound } = novikoff(EX, EY);
    expect(R ** 2).toBeCloseTo(18, 9);
    expect(gamma ** 2).toBeCloseTo(81 / 221, 5);
    expect(bound).toBeCloseTo(442 / 9, 3);
  });

  it("the dual coefficients reproduce w and b at every step", () => {
    const { X, y } = screening();
    for (const s of perceptron(X, y).steps) {
      const d = dualToPrimal(X, y, s.alpha);
      expect(d.w[0]).toBeCloseTo(s.w[0], 9);
      expect(d.w[1]).toBeCloseTo(s.w[1], 9);
      expect(d.b).toBeCloseTo(s.b, 9);
    }
  });

  it("the widget data is separable; every visiting order converges within Novikoff's bound", () => {
    const { X, y } = screening();
    const { bound, gamma } = novikoff(X, y);
    expect(gamma).toBeGreaterThan(0);
    for (const seed of [0, 1, 2, 3, 4, 5, 6, 7]) {
      const { steps, converged } = perceptron(X, y, 1, seed === 0 ? undefined : shuffled(X.length, seed));
      expect(converged).toBe(true);
      expect(steps.length - 1).toBeLessThanOrEqual(bound);
      const last = steps.at(-1)!;
      expect(mistakes(X, y, last.w, last.b)).toBe(0);
    }
  });

  it("different visiting orders end at different separating lines (the solution is not unique)", () => {
    const { X, y } = screening();
    const angle = (seed: number) => {
      const w = perceptron(X, y, 1, seed === 0 ? undefined : shuffled(X.length, seed)).steps.at(-1)!.w;
      return Math.atan2(w[1], w[0]);
    };
    const angles = [0, 1, 2, 3].map(angle);
    expect(Math.max(...angles) - Math.min(...angles)).toBeGreaterThan(0.05);
  });

  it("from zero, the learning rate only rescales (w, b): same mistakes, same line", () => {
    const { X, y } = screening();
    const a = perceptron(X, y, 1).steps;
    const b = perceptron(X, y, 0.1).steps;
    expect(b.map((s) => s.i)).toEqual(a.map((s) => s.i));
    b.forEach((s, k) => {
      expect(s.w[0]).toBeCloseTo(0.1 * a[k].w[0], 9);
      expect(s.b).toBeCloseTo(0.1 * a[k].b, 9);
    });
  });

  it("on non-separable data it never converges", () => {
    const X: V2[] = [[0, 0], [1, 1], [0, 1], [1, 0]];
    expect(perceptron(X, [1, 1, -1, -1], 1, undefined, 400).converged).toBe(false);
    expect(novikoff(X, [1, 1, -1, -1]).gamma).toBe(0);
  });
});

describe("logistic regression", () => {
  const { X, y } = overlapping();
  const t0: Theta = [0.3, -0.2, 0.1];

  it("the gradient matches finite differences", () => {
    const g = ceGrad(X, y, t0);
    for (let k = 0; k < 3; k++) {
      const e = [0, 0, 0] as Theta;
      e[k] = 1e-6;
      const fd = (crossEntropy(X, y, t0.map((v, j) => v + e[j]) as Theta) - crossEntropy(X, y, t0.map((v, j) => v - e[j]) as Theta)) / 2e-6;
      expect(g[k]).toBeCloseTo(fd, 6);
    }
  });

  it("the Hessian is positive semidefinite (the loss is convex)", () => {
    for (const t of [t0, [2, 1, -9], [0, 0, 0]] as Theta[]) {
      const H = ceHessian(X, t);
      for (let k = 0; k < 50; k++) {
        const v = [Math.sin(k), Math.cos(3 * k), Math.sin(7 * k + 1)];
        const q = v.reduce((s, vi, i) => s + vi * v.reduce((s2, vj, j) => s2 + H[i][j] * vj, 0), 0);
        expect(q).toBeGreaterThanOrEqual(-1e-12);
      }
    }
  });

  it("overlapping data: Newton converges in a handful of steps, gradient descent needs hundreds", () => {
    const path = newton(X, y);
    const last = path.at(-1)!;
    expect(Math.hypot(...ceGrad(X, y, last))).toBeLessThan(1e-8);
    const nNewton = newtonIterations(X, y);
    const nGd = gdIterations(X, y, 0.5);
    expect(nNewton).toBeLessThanOrEqual(8);
    expect(newton(X, y, 50, 1e-12).length - 1).toBeLessThanOrEqual(8);
    expect(nGd).toBeGreaterThan(200);
    expect(nGd).toBeGreaterThan(20 * nNewton);
    expect(novikoff(X, y).gamma).toBe(0);
  });

  it("separable data: there is no finite MLE; ‖w‖ keeps growing and the loss goes to 0", () => {
    const { X: Xs, y: ys } = screening();
    const path = newton(Xs, ys, 25, 0);
    const norms = path.map((t) => Math.hypot(t[0], t[1]));
    expect(norms[25]).toBeGreaterThan(norms[10]);
    expect(norms[10]).toBeGreaterThan(norms[5]);
    expect(crossEntropy(Xs, ys, path[25])).toBeLessThan(1e-3);
  });

  it("polar parametrisation round-trips", () => {
    const t = fromPolar(30, 2, 1.5);
    const p = toPolar(t);
    expect(p.deg).toBeCloseTo(30, 9);
    expect(p.s).toBeCloseTo(2, 9);
    expect(p.c).toBeCloseTo(1.5, 9);
  });
});

describe("loss zoo", () => {
  const ms = Array.from({ length: 121 }, (_, i) => -3 + i * 0.05);

  it("logistic (base 2), hinge and exponential losses upper-bound the 0-1 loss; the perceptron loss does not", () => {
    for (const m of ms) {
      for (const id of ["logistic", "hinge", "exp"] as const) expect(loss[id](m)).toBeGreaterThanOrEqual(loss.zeroOne(m) - 1e-12);
    }
    expect(loss.perceptron(-0.1)).toBeLessThan(loss.zeroOne(-0.1));
  });

  it("all are non-increasing in the margin, and pull = −ℓ′ (checked by finite differences)", () => {
    for (const id of LOSS_IDS) {
      for (const m of ms) {
        expect(loss[id](m + 0.01)).toBeLessThanOrEqual(loss[id](m) + 1e-12);
        if (id !== "zeroOne" && Math.abs(m) > 0.02 && Math.abs(m - 1) > 0.02) {
          const fd = -(loss[id](m + 1e-6) - loss[id](m - 1e-6)) / 2e-6;
          expect(pull[id](m)).toBeCloseTo(fd, 4);
        }
      }
    }
  });

  it("confident correct points still pull a little under logistic loss, not at all under hinge; exp explodes on mistakes", () => {
    expect(pull.logistic(3)).toBeGreaterThan(0);
    expect(pull.hinge(3)).toBe(0);
    expect(pull.exp(-3) / pull.logistic(-3)).toBeGreaterThan(10);
  });
});

describe("maximum entropy on a die", () => {
  it("mean 3.5 gives the uniform distribution", () => {
    const { p, lambda } = maxentDie(3.5);
    p.forEach((q) => expect(q).toBeCloseTo(1 / 6, 9));
    expect(lambda).toBeCloseTo(0, 9);
  });

  it("matches the mean and beats other distributions with the same mean", () => {
    for (const mu of [1.8, 2.7, 4.5, 5.2]) {
      const { p, H } = maxentDie(mu);
      expect(p.reduce((s, q, k) => s + q * (k + 1), 0)).toBeCloseTo(mu, 8);
      const other = twoFaceDie(mu);
      expect(other.reduce((s, q, k) => s + q * (k + 1), 0)).toBeCloseTo(mu, 9);
      expect(H).toBeGreaterThan(entropy(other));
      // Small perturbations that keep total mass and the mean lower the entropy.
      const d = [1, -2, 1, 0, 0, 0].map((v) => v * 1e-3);
      expect(entropy(p.map((q, k) => q + d[k]))).toBeLessThan(H);
    }
  });
});

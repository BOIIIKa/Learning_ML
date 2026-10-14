// Linear soft-margin SVM by SMO — the same algorithm as exercises/ch07/smo.py.
import type { Trace, TraceStep } from "./trace";

export type Kernel = (a: number[], c: number[]) => number;

export const kernels = {
  linear: (): Kernel => (a, c) => a[0] * c[0] + a[1] * c[1],
  poly: (degree: number): Kernel => (a, c) => (1 + a[0] * c[0] + a[1] * c[1]) ** degree,
  rbf: (gamma: number): Kernel => (a, c) => Math.exp(-gamma * ((a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2)),
};

export interface SmoOptions {
  C?: number;
  tol?: number;
  maxSteps?: number;
  /** Defaults to the linear kernel; `w` in the result is only meaningful for it. */
  kernel?: Kernel;
}

export interface SmoResult {
  alpha: number[];
  b: number;
  w: [number, number];
  steps: TraceStep[];
}

export function smo(X: number[][], y: number[], options: SmoOptions = {}): SmoResult {
  const { C = 1, tol = 1e-3, maxSteps = 500, kernel = kernels.linear() } = options;
  const n = y.length;
  const K = X.map((a) => X.map((c) => kernel(a, c)));
  const alpha = new Array(n).fill(0);
  let b = 0;
  const steps: TraceStep[] = [];

  const weightVector = (): [number, number] => {
    let w0 = 0;
    let w1 = 0;
    for (let k = 0; k < n; k++) {
      w0 += alpha[k] * y[k] * X[k][0];
      w1 += alpha[k] * y[k] * X[k][1];
    }
    return [w0, w1];
  };
  const record = (highlight: number[], note: string) => {
    let dual = 0;
    for (let i = 0; i < n; i++) {
      dual += alpha[i];
      for (let j = 0; j < n; j++) dual -= 0.5 * alpha[i] * alpha[j] * y[i] * y[j] * K[i][j];
    }
    const support = alpha.flatMap((a, k) => (a > 1e-6 ? [k] : []));
    steps.push({ w: weightVector(), b, highlight, support, metrics: { dual }, note });
  };

  record([], "init");
  for (let step = 0; step < maxSteps; step++) {
    const E = y.map((_, k) => {
      let f = b;
      for (let m = 0; m < n; m++) f += alpha[m] * y[m] * K[m][k];
      return f - y[k];
    });
    const violation = E.map((e, k) => {
      const r = y[k] * e;
      return (r < -tol && alpha[k] < C) || (r > tol && alpha[k] > 0) ? Math.abs(r) : 0;
    });
    if (Math.max(...violation) <= 0) break;

    let progressed = false;
    const outer = [...violation.keys()].sort((p, q) => violation[q] - violation[p]);
    for (const i of outer) {
      if (violation[i] <= 0) break;
      const inner = [...E.keys()].sort((p, q) => Math.abs(E[q] - E[i]) - Math.abs(E[p] - E[i]));
      for (const j of inner) {
        if (j === i) continue;
        const [ai, aj] = [alpha[i], alpha[j]];
        const [L, H] =
          y[i] !== y[j] ? [Math.max(0, aj - ai), Math.min(C, C + aj - ai)] : [Math.max(0, ai + aj - C), Math.min(C, ai + aj)];
        const eta = K[i][i] + K[j][j] - 2 * K[i][j];
        if (L >= H || eta <= 1e-12) continue;
        const ajNew = Math.min(H, Math.max(L, aj + (y[j] * (E[i] - E[j])) / eta));
        if (Math.abs(ajNew - aj) < 1e-8) continue;
        const aiNew = ai + y[i] * y[j] * (aj - ajNew);
        const [di, dj] = [aiNew - ai, ajNew - aj];
        const b1 = b - E[i] - y[i] * di * K[i][i] - y[j] * dj * K[i][j];
        const b2 = b - E[j] - y[i] * di * K[i][j] - y[j] * dj * K[j][j];
        b = aiNew > 0 && aiNew < C ? b1 : ajNew > 0 && ajNew < C ? b2 : (b1 + b2) / 2;
        alpha[i] = aiNew;
        alpha[j] = ajNew;
        record([i, j], "step");
        progressed = true;
        break;
      }
      if (progressed) break;
    }
    if (!progressed) break;
  }
  return { alpha, b, w: weightVector(), steps };
}

export function smoTrace(X: number[][], y: number[], C = 1): Trace {
  return { kind: "classifier2d", X, y, steps: smo(X, y, { C }).steps };
}

/** f(p) = Σ α_k y_k K(x_k, p) + b at each point (row-major grid order). */
export function decisionField(X: number[][], y: number[], r: SmoResult, kernel: Kernel, points: number[][]): number[] {
  const sv = r.alpha.flatMap((a, k) => (a > 1e-8 ? [k] : []));
  return points.map((p) => sv.reduce((f, k) => f + r.alpha[k] * y[k] * kernel(X[k], p), r.b));
}

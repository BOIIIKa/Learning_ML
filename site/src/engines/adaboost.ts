// AdaBoost with decision stumps — the same algorithm as exercises/ch08/adaboost.py.
import type { Trace } from "./trace";
import { gridFor } from "./data";

export interface Stump {
  alpha: number;
  feature: number;
  threshold: number;
  polarity: number;
}

const stumpPredict = (x: number[], s: Omit<Stump, "alpha">) => s.polarity * (x[s.feature] <= s.threshold ? 1 : -1);

export function bestStump(X: number[][], y: number[], w: number[]) {
  let best = { err: Infinity, feature: 0, threshold: 0, polarity: 1 };
  for (let feature = 0; feature < X[0].length; feature++) {
    const values = [...new Set(X.map((x) => x[feature]))].sort((p, q) => p - q);
    for (let k = 0; k + 1 < values.length; k++) {
      const threshold = (values[k] + values[k + 1]) / 2;
      for (const polarity of [1, -1]) {
        let err = 0;
        for (let i = 0; i < y.length; i++) if (stumpPredict(X[i], { feature, threshold, polarity }) !== y[i]) err += w[i];
        if (err < best.err) best = { err, feature, threshold, polarity };
      }
    }
  }
  return best;
}

export function adaboost(X: number[][], y: number[], rounds = 30) {
  const n = y.length;
  let w = new Array(n).fill(1 / n);
  const model: Stump[] = [];
  const grid = gridFor(X);
  const field = new Array(grid.points.length).fill(0);
  const score = new Array(n).fill(0);
  let bound = 1;
  const trace: Trace = { kind: "classifier2d", X, y, grid: { x: grid.x, y: grid.y }, steps: [] };

  for (let m = 0; m < rounds; m++) {
    const { err: rawErr, ...stump } = bestStump(X, y, w);
    const err = Math.min(Math.max(rawErr, 1e-10), 1 - 1e-10);
    const alpha = 0.5 * Math.log((1 - err) / err);
    model.push({ alpha, ...stump });
    const pred = X.map((x) => stumpPredict(x, stump));
    pred.forEach((p, i) => (score[i] += alpha * p));
    grid.points.forEach((p, k) => (field[k] += alpha * stumpPredict(p, stump)));
    bound *= 2 * Math.sqrt(err * (1 - err));
    const trainErr = score.filter((s, i) => (s >= 0 ? 1 : -1) !== y[i]).length / n;
    trace.steps.push({
      field: [...field],
      weights: w.map((v) => v * n),
      highlight: pred.flatMap((p, i) => (p !== y[i] ? [i] : [])),
      metrics: { error: trainErr, bound, alpha },
      note: `x${stump.feature + 1} ≤ ${stump.threshold.toFixed(2)}`,
    });
    w = w.map((v, i) => v * Math.exp(-alpha * y[i] * pred[i]));
    const Z = w.reduce((s, v) => s + v, 0);
    w = w.map((v) => v / Z);
  }
  return { model, trace };
}

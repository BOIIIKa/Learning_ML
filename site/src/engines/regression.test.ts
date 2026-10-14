import { describe, expect, it } from "vitest";
import { empiricalAbsRisk, expAbsRisk, expSample, gradientDescent, leastSquares, median, randomSearch, risk, studyData } from "./regression";

describe("regression", () => {
  const { xs, ys } = studyData();

  it("least squares is the minimum of the squared risk and GD converges to it", () => {
    const w = leastSquares("linear", xs, ys);
    const best = risk("squared", "linear", w, xs, ys);
    for (const dw of [[0.5, 0], [0, 0.5], [-0.3, 0.2]]) expect(risk("squared", "linear", [w[0] + dw[0], w[1] + dw[1]], xs, ys)).toBeGreaterThan(best);
    const gd = gradientDescent("squared", "linear", xs, ys, 400);
    expect(gd.ws.at(-1)![0]).toBeCloseTo(w[0], 2);
    expect(gd.ws.at(-1)![1]).toBeCloseTo(w[1], 2);
  });

  it("constant model: squared → mean, absolute → median", () => {
    const sq = gradientDescent("squared", "constant", xs, ys, 300).ws.at(-1)![0];
    expect(sq).toBeCloseTo(ys.reduce((a, b) => a + b, 0) / ys.length, 3);
    // Any point between the two middle values is optimal, so compare risks rather than locations.
    const ab = gradientDescent("absolute", "constant", xs, ys, 3000).ws.at(-1)!;
    expect(risk("absolute", "constant", ab, xs, ys)).toBeCloseTo(risk("absolute", "constant", [median(ys)], xs, ys), 1);
  });

  it("absolute loss is robust to the absent students", () => {
    const ls = leastSquares("linear", xs, ys);
    const ab = gradientDescent("absolute", "linear", xs, ys, 4000).ws.at(-1)!;
    // true slope is 5.2 per hour = 26 per scaled unit
    expect(Math.abs(ab[1] - 26)).toBeLessThan(Math.abs(ls[1] - 26));
  });

  it("random search never increases its best risk", () => {
    const { risks } = randomSearch("squared", "linear", xs, ys, 100);
    for (let t = 1; t < risks.length; t++) expect(risks[t]).toBeLessThanOrEqual(risks[t - 1]);
  });

  it("empirical absolute risk approaches E|Y−c| for Exp(1)", () => {
    const ysBig = expSample(20000, 3);
    for (const c of [0, 0.5, Math.LN2, 1.5]) expect(empiricalAbsRisk(ysBig, c)).toBeCloseTo(expAbsRisk(c), 1);
    expect(median(ysBig)).toBeCloseTo(Math.LN2, 1);
  });
});

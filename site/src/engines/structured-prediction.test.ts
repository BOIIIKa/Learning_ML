import { describe, expect, it } from "vitest";
import { EXAMPLE_OBS, HMM, backward, baumWelch, bruteForce, crfBrute, crfLogZ, forward, sampleHmm, viterbi, type Hmm } from "./structured-prediction";

describe("HMM worked example", () => {
  it("forward variables and P(O) = 0.10007", () => {
    const al = forward(HMM, EXAMPLE_OBS);
    [[0.54, 0.08], [0.041, 0.168], [0.00959, 0.09048]].forEach((row, t) => row.forEach((v, i) => expect(al[t][i]).toBeCloseTo(v, 12)));
    expect(al[2][0] + al[2][1]).toBeCloseTo(0.10007, 12);
  });
  it("backward gives the same P(O)", () => {
    const be = backward(HMM, EXAMPLE_OBS);
    const p = HMM.pi.reduce((s, q, i) => s + q * HMM.B[i][EXAMPLE_OBS[0]] * be[0][i], 0);
    expect(p).toBeCloseTo(0.10007, 12);
  });
  it("Viterbi: δ values, back-pointers and best path stable → deteriorating → deteriorating", () => {
    const v = viterbi(HMM, EXAMPLE_OBS);
    [[0.54, 0.08], [0.0378, 0.1296], [0.005184, 0.062208]].forEach((row, t) => row.forEach((x, i) => expect(v.delta[t][i]).toBeCloseTo(x, 12)));
    expect(v.back).toEqual([[0, 0], [1, 1]]);
    expect(v.path).toEqual([0, 1, 1]);
    expect(v.prob).toBeCloseTo(0.062208, 12);
  });
  it("dynamic programming matches brute-force enumeration", () => {
    const { O } = sampleHmm(HMM, 8, 3);
    const b = bruteForce(HMM, O);
    expect(forward(HMM, O).at(-1)!.reduce((a, c) => a + c, 0)).toBeCloseTo(b.p, 12);
    expect(viterbi(HMM, O).path).toEqual(b.best);
  });
});

describe("Baum–Welch", () => {
  const { O } = sampleHmm(HMM, 400, 7);
  const init: Hmm = { pi: [0.5, 0.5], A: [[0.6, 0.4], [0.5, 0.5]], B: [[0.7, 0.3], [0.4, 0.6]] };
  const { models, loglik } = baumWelch(O, init, 40);
  it("never decreases the log-likelihood (EM)", () => {
    for (let k = 1; k < loglik.length; k++) expect(loglik[k]).toBeGreaterThanOrEqual(loglik[k - 1] - 1e-9);
  });
  it("recovers the emission probabilities roughly", () => {
    const B = models.at(-1)!.B;
    expect(Math.abs(B[0][0] - 0.9)).toBeLessThan(0.12);
    expect(Math.abs(B[1][1] - 0.8)).toBeLessThan(0.12);
  });
});

describe("linear-chain CRF", () => {
  it("the forward recursion gives the exact partition function", () => {
    const U = [[0.3, -0.2, 0.5], [1, 0, -1], [-0.4, 0.2, 0.1], [0.7, -0.3, 0]];
    const Tr = [[0.5, -0.1, 0], [0.2, 0.3, -0.6], [-0.2, 0.4, 0.1]];
    expect(crfLogZ(U, Tr)).toBeCloseTo(crfBrute(U, Tr), 10);
  });
});

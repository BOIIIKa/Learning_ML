import { describe, expect, it } from "vitest";
import { corpus, lag1, ldaGibbs, metropolis, pagerank, purity, surfer } from "./markov-chains";

const LINKS = [[1, 2], [2], [0]]; // A→B, A→C, B→C, C→A

describe("PageRank", () => {
  it("worked example: (686, 380, 703)/1769 with damping 0.85", () => {
    const r = pagerank(LINKS);
    [686 / 1769, 380 / 1769, 703 / 1769].forEach((v, i) => expect(r[i]).toBeCloseTo(v, 10));
  });
  it("the random surfer's visit frequencies converge to PageRank", () => {
    const f = surfer(LINKS, 0.85, 200000, 3);
    pagerank(LINKS).forEach((v, i) => expect(Math.abs(f[i] - v)).toBeLessThan(0.01));
  });
});

describe("Metropolis–Hastings", () => {
  it("tiny steps accept almost everything but mix slowly; huge steps reject almost everything", () => {
    const small = metropolis(4000, 0.05, 1), mid = metropolis(4000, 0.9, 1), big = metropolis(4000, 8, 1);
    expect(small.acc).toBeGreaterThan(0.9);
    expect(big.acc).toBeLessThan(0.1);
    expect(mid.acc).toBeGreaterThan(0.2);
    expect(mid.acc).toBeLessThan(0.7);
    expect(lag1(small.path)).toBeGreaterThan(lag1(mid.path));
  });
  it("a moderate step visits both bumps", () => {
    const { path } = metropolis(6000, 0.9, 2);
    const right = path.filter((p) => p[0] > 0.3).length / path.length;
    expect(right).toBeGreaterThan(0.15);
    expect(right).toBeLessThan(0.85);
  });
});

describe("LDA collapsed Gibbs", () => {
  it("topics go from mixed to pure", () => {
    const docs = corpus();
    const states = ldaGibbs(docs, 2, 8, 40, 5);
    expect(purity(states[0])).toBeLessThan(0.65);
    expect(purity(states.at(-1)!)).toBeGreaterThan(0.85);
    const tot = states.at(-1)!.nk.reduce((a, b) => a + b, 0);
    expect(tot).toBe(docs.flat().length);
  });
});

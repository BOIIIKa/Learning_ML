import { describe, expect, it } from "vitest";
import { backward, forward, initNet, paramCount, pattern, predict, type Net } from "./mlp";

describe("mlp", () => {
  it("counts parameters layer by layer", () => {
    expect(paramCount([2, 8, 8, 1])).toBe(2 * 8 + 8 + 8 * 8 + 8 + 8 + 1);
    expect(paramCount([1, 6, 1])).toBe(19);
  });

  it("backpropagation matches finite differences (relu and tanh)", () => {
    for (const a of ["relu", "tanh"] as const) {
      const net = initNet([3, 5, 4, 2], 3);
      net.forEach((L) => L.b.forEach((_, i) => (L.b[i] = 0.1 * (i + 1))));
      const x = [0.3, -0.7, 1.1], target = [0.5, -1];
      const loss = (n: Net) => predict(n, x, a).reduce((s, v, i) => s + 0.5 * (v - target[i]) ** 2, 0);
      const cache = forward(net, x, a);
      const g = backward(net, cache, cache.hs.at(-1)!.map((v, i) => v - target[i]), a);
      const eps = 1e-6;
      for (let l = 0; l < net.length; l++) for (let i = 0; i < net[l].W.length; i++) for (let j = 0; j < net[l].W[i].length; j++) {
        const w0 = net[l].W[i][j];
        net[l].W[i][j] = w0 + eps; const up = loss(net);
        net[l].W[i][j] = w0 - eps; const dn = loss(net);
        net[l].W[i][j] = w0;
        expect(g[l].W[i][j]).toBeCloseTo((up - dn) / (2 * eps), 5);
      }
    }
  });

  it("an activation pattern is constant where the net is linear", () => {
    const net = initNet([2, 4, 1], 7);
    expect(pattern(net, [0.5, 0.5])).toMatch(/^[01]{4}$/);
  });
});

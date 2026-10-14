import { describe, expect, it } from "vitest";
import bundle from "../../generated/exercises.json";
import { equivalent } from "./DerivationExercise";

const items = Object.values(bundle.chapters).flatMap((c: any) => c.derivation);

describe("derivation answers", () => {
  it.each(items.map((d: any) => [d.id, d]))("%s: reference answer is self-consistent and finite somewhere", (_id, d: any) => {
    expect(equivalent(d.answer, d.answer, d.vars)).toBe(true);
  });

  it("accepts algebraically equivalent forms and rejects wrong ones", () => {
    const vars = { e: [0.05, 0.45] as [number, number], M: [1, 10] as [number, number] };
    expect(equivalent("(4*e*(1-e))^(M/2)", "(2*sqrt(e*(1 - e)))^M", vars)).toBe(true);
    expect(equivalent("(2*e*(1-e))^M", "(2*sqrt(e*(1 - e)))^M", vars)).toBe(false);
    expect(equivalent("log(sqrt((1-e)/e))", "0.5*log((1 - e)/e)", { e: [0.05, 0.45] })).toBe(true);
    expect(equivalent("x1*z1*x1*z1 + 2*x1*x2*z1*z2 + (x2*z2)^2", "(x1*z1 + x2*z2)^2",
      { x1: [-2, 2], x2: [-2, 2], z1: [-2, 2], z2: [-2, 2] })).toBe(true);
  });
});

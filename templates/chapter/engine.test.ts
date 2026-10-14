import { describe, expect, it } from "vitest";
import { fit, toyData } from "./{{SLUG}}";

// Test every claim a widget caption or the notes make about this engine.
describe("{{SLUG}} engine", () => {
  it("TODO(scaffold): recovers the generating parameters", () => {
    const { xs, ys } = toyData(400, 3);
    const { slope } = fit(xs, ys);
    expect(slope).toBeCloseTo(2, 1);
  });
});

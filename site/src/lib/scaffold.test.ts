import { describe, expect, it } from "vitest";

// Scaffolded chapters (scripts/new_chapter.py) carry TODO(scaffold) markers. Everything matched
// here is published, so no marker may survive into a merge.
const files = import.meta.glob(
  [
    "../content/**/*.mdx", "../slides/**/*.mdx", "../recaps/**/*.mdx", "../handbook/**/*.mdx",
    "../components/chapters/**/*.tsx", "../engines/*.ts", "../../../exercises/**/*.{py,yaml}",
  ],
  { query: "?raw", import: "default", eager: true },
) as Record<string, string>;

describe("no unfinished scaffolding", () => {
  it("scans the published folders", () => expect(Object.keys(files).length).toBeGreaterThan(50));
  it("has no TODO(scaffold) markers left", () => {
    const left = Object.entries(files).filter(([, text]) => text.includes("TODO(scaffold)")).map(([f]) => f);
    expect(left).toEqual([]);
  });
});

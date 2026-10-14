import katex from "katex";

const escapeHtml = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

const math = (tex: string, displayMode: boolean) =>
  katex.renderToString(tex, { displayMode, throwOnError: false, strict: "ignore" });

/** Minimal rich text: $$display$$, $inline$, `code`, **bold**. Everything else is escaped. */
export function renderRich(text: string): string {
  return text
    .split(/(\$\$[\s\S]+?\$\$|\$[^$\n]+?\$|`[^`\n]+`)/g)
    .map((part) => {
      if (part.startsWith("$$")) return math(part.slice(2, -2), true);
      if (part.startsWith("$") && part.length > 1) return math(part.slice(1, -1), false);
      if (part.startsWith("`")) return `<code>${escapeHtml(part.slice(1, -1))}</code>`;
      return escapeHtml(part).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    })
    .join("");
}

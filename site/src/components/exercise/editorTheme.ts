import { EditorView } from "@codemirror/view";
import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { tags } from "@lezer/highlight";

const highlight = HighlightStyle.define([
  { tag: tags.keyword, color: "var(--accent-3)", fontWeight: "600" },
  { tag: [tags.string, tags.special(tags.string)], color: "var(--success)" },
  { tag: tags.comment, color: "var(--text-muted)", fontStyle: "italic" },
  { tag: [tags.number, tags.bool, tags.null], color: "var(--warning)" },
  { tag: [tags.function(tags.variableName), tags.function(tags.propertyName)], color: "var(--accent)" },
  { tag: tags.definition(tags.variableName), color: "var(--accent)" },
  { tag: tags.operator, color: "var(--accent-2)" },
]);

export const editorTheme = [
  EditorView.theme({
    "&": { backgroundColor: "var(--code-bg)", color: "var(--text)", fontSize: "13.5px", borderRadius: "10px" },
    ".cm-content": { fontFamily: "var(--mono)", caretColor: "var(--accent)" },
    ".cm-gutters": { backgroundColor: "var(--code-bg)", color: "var(--text-muted)", border: "none" },
    ".cm-activeLine, .cm-activeLineGutter": { backgroundColor: "color-mix(in srgb, var(--accent) 6%, transparent)" },
    "&.cm-focused .cm-cursor": { borderLeftColor: "var(--accent)" },
    "&.cm-focused .cm-selectionBackground, .cm-selectionBackground": { backgroundColor: "color-mix(in srgb, var(--accent) 25%, transparent) !important" },
    ".cm-scroller": { maxHeight: "520px" },
    ".cm-tooltip": { border: "none", backgroundColor: "transparent" },
  }),
  syntaxHighlighting(highlight),
];

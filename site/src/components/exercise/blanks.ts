// CodeMirror extension for fill-in-the-blank exercises.
//
// Each blank is a `# TODO(id): …` line plus its `raise NotImplementedError("id")` line in the
// skeleton. Blanks are tracked as mark decorations so they follow the student's edits. Pressing
// Tab inside a blank shows a hint; pressing it again within DOUBLE_TAB_MS types the reference
// answer in with a glow and a small particle burst.
import { indentMore } from "@codemirror/commands";
import { StateEffect, StateField, type EditorState, type Range } from "@codemirror/state";
import { Decoration, type DecorationSet, EditorView, keymap, showTooltip, type Tooltip } from "@codemirror/view";
import { burst } from "../ui/fx";

export interface Blank {
  id: string;
  answer: string;
}

const DOUBLE_TAB_MS = 650;

const removeBlank = StateEffect.define<string>();
const setFilling = StateEffect.define<{ from: number; to: number } | null>();
const addFlash = StateEffect.define<{ from: number; to: number }>();
const clearFlash = StateEffect.define<null>();
const setHint = StateEffect.define<number | null>();

const blankMark = (id: string) => Decoration.mark({ class: "cm-blank", blankId: id, inclusive: true });
const fillingMark = Decoration.mark({ class: "cm-filling" });
const flashMark = Decoration.mark({ class: "cm-filled" });

/** Ranges of the TODO blocks in a freshly loaded document. */
function findBlanks(state: EditorState): Range<Decoration>[] {
  const out: Range<Decoration>[] = [];
  for (let n = 1; n < state.doc.lines; n++) {
    const line = state.doc.line(n);
    const m = /^\s*# TODO\(([\w-]+)\)/.exec(line.text);
    if (!m) continue;
    const next = state.doc.line(n + 1);
    if (next.text.includes(`NotImplementedError("${m[1]}")`)) out.push(blankMark(m[1]).range(line.from, next.to));
  }
  return out;
}

const blankField = StateField.define<DecorationSet>({
  create: (state) => Decoration.set(findBlanks(state)),
  update(deco, tr) {
    deco = deco.map(tr.changes);
    for (const e of tr.effects) {
      if (e.is(removeBlank)) deco = deco.update({ filter: (_f, _t, d) => d.spec.blankId !== e.value });
    }
    // A full document replacement (reset) re-scans the blanks.
    if (tr.isUserEvent("reset")) {
      deco = Decoration.set(findBlanks(tr.state));
    }
    return deco;
  },
  provide: (f) => EditorView.decorations.from(f),
});

const fillingField = StateField.define<DecorationSet>({
  create: () => Decoration.none,
  update(deco, tr) {
    deco = deco.map(tr.changes);
    for (const e of tr.effects) {
      if (e.is(setFilling)) deco = e.value && e.value.to > e.value.from ? Decoration.set([fillingMark.range(e.value.from, e.value.to)]) : Decoration.none;
      if (e.is(addFlash)) deco = Decoration.set([flashMark.range(e.value.from, e.value.to)]);
      if (e.is(clearFlash)) deco = Decoration.none;
    }
    return deco;
  },
  provide: (f) => EditorView.decorations.from(f),
});

function hintField(text: string) {
  return StateField.define<Tooltip | null>({
    create: () => null,
    update(tip, tr) {
      for (const e of tr.effects) {
        if (e.is(setHint)) {
          return e.value === null ? null : {
            pos: e.value,
            above: true,
            create: () => {
              const dom = document.createElement("div");
              dom.className = "cm-tab-hint";
              dom.textContent = text;
              return { dom };
            },
          };
        }
      }
      return tr.docChanged || tr.selection ? null : tip;
    },
    provide: (f) => showTooltip.from(f),
  });
}

/** The blank (id, from, to) containing `pos`, if any. */
export function blankAt(state: EditorState, pos: number) {
  let found: { id: string; from: number; to: number } | null = null;
  state.field(blankField).between(pos, pos, (from, to, d) => {
    found = { id: d.spec.blankId, from, to };
  });
  return found as { id: string; from: number; to: number } | null;
}

export function remainingBlanks(state: EditorState): string[] {
  const ids: string[] = [];
  state.field(blankField).between(0, state.doc.length, (_f, _t, d) => {
    ids.push(d.spec.blankId);
  });
  return ids;
}

const nextFrame = () => new Promise((r) => requestAnimationFrame(r));

/** Type the reference answer into blank `id` with the glow animation. Resolves when done. */
export async function fillBlank(view: EditorView, blanks: Blank[], id: string) {
  const answer = blanks.find((b) => b.id === id)?.answer;
  let range: { from: number; to: number } | null = null;
  view.state.field(blankField).between(0, view.state.doc.length, (from, to, d) => {
    if (d.spec.blankId === id) range = { from, to };
  });
  if (!answer || !range) return;
  const { from, to } = range as { from: number; to: number };

  view.dispatch({ changes: { from, to, insert: "" }, effects: [removeBlank.of(id), setHint.of(null)], userEvent: "input.fill" });
  const chunk = Math.max(1, Math.ceil(answer.length / 36));
  let pos = from;
  for (let i = 0; i < answer.length; i += chunk) {
    await nextFrame();
    const piece = answer.slice(i, i + chunk);
    view.dispatch({ changes: { from: pos, insert: piece }, effects: setFilling.of({ from, to: pos + piece.length }), userEvent: "input.fill" });
    pos += piece.length;
  }
  view.dispatch({ effects: addFlash.of({ from, to: pos }), selection: { anchor: pos } });
  const c = view.coordsAtPos(pos);
  if (c) burst(c.right, c.top + 8, { count: 26, power: 0.7, gravity: 0.12 });
  setTimeout(() => view.dispatch({ effects: clearFlash.of(null) }), 1400);
}

export function blanksExtension(blanks: Blank[], tabAgainText: string, onChange: (left: string[]) => void) {
  let last = { time: 0, id: "" };
  const tab = (view: EditorView) => {
    const head = view.state.selection.main.head;
    const blank = blankAt(view.state, head);
    if (!blank) return indentMore(view);
    const now = performance.now();
    if (last.id === blank.id && now - last.time < DOUBLE_TAB_MS) {
      last = { time: 0, id: "" };
      fillBlank(view, blanks, blank.id);
    } else {
      last = { time: now, id: blank.id };
      view.dispatch({ effects: setHint.of(head) });
    }
    return true;
  };
  return [
    blankField,
    fillingField,
    hintField(tabAgainText),
    keymap.of([{ key: "Tab", run: tab }]),
    EditorView.updateListener.of((u) => {
      if (u.docChanged || u.transactions.some((tr) => tr.effects.some((e) => e.is(removeBlank)))) onChange(remainingBlanks(u.state));
    }),
  ];
}

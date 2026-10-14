import katex from "katex";
import { compile, parse } from "mathjs";
import { useMemo, useRef, useState } from "react";
import { pick, t, type Bilingual, type Lang } from "../../i18n/ui";
import { renderRich } from "../../lib/rich";
import { burstFrom } from "../ui/fx";
import "./exercise.css";

export interface DerivationItem {
  id: string;
  title: Bilingual;
  prompt: Bilingual;
  answer: string;
  vars: Record<string, [number, number]>;
  solution: Bilingual;
}

type Verdict = "idle" | "correct" | "wrong" | "error" | "revealed";

const DOUBLE_TAB_MS = 650;

/** Numerically compare two expressions at random points of the variable box. */
export function equivalent(student: string, reference: string, vars: Record<string, [number, number]>, trials = 32) {
  const f = compile(student);
  const g = compile(reference);
  const names = Object.keys(vars);
  let compared = 0;
  for (let k = 0; k < trials * 3 && compared < trials; k++) {
    const scope = Object.fromEntries(names.map((n) => [n, vars[n][0] + Math.random() * (vars[n][1] - vars[n][0])]));
    const want = g.evaluate({ ...scope });
    if (typeof want !== "number" || !Number.isFinite(want)) continue;
    const got = f.evaluate({ ...scope });
    if (typeof got !== "number" || Math.abs(got - want) > 1e-6 * (1 + Math.abs(want))) return false;
    compared++;
  }
  return compared > 0;
}

function unknownSymbols(expr: string, allowed: string[]) {
  const known = new Set([...allowed, "pi", "e"]);
  const out = new Set<string>();
  parse(expr).traverse((node: any, _path: string, parent: any) => {
    const isCallee = parent?.type === "FunctionNode" && parent.fn === node;
    if (node.type === "SymbolNode" && !isCallee && !known.has(node.name)) out.add(node.name);
  });
  return [...out];
}

function toTex(expr: string) {
  try {
    return katex.renderToString(parse(expr).toTex({ parenthesis: "auto" }), { throwOnError: false });
  } catch {
    return null;
  }
}

export default function DerivationExercise({ item, lang, index }: { item: DerivationItem; lang: Lang; index: number }) {
  const [input, setInput] = useState("");
  const [verdict, setVerdict] = useState<Verdict>("idle");
  const [message, setMessage] = useState("");
  const [open, setOpen] = useState(false);
  const [hint, setHint] = useState(false);
  const [typing, setTyping] = useState(false);
  const lastTab = useRef(0);
  const checkBtn = useRef<HTMLButtonElement>(null);
  const preview = useMemo(() => (input.trim() ? toTex(input) : null), [input]);

  const check = () => {
    try {
      const unknown = unknownSymbols(input, Object.keys(item.vars));
      if (unknown.length) {
        setVerdict("error");
        setMessage(`${t(lang, "parseError")}: ${unknown.join(", ")}`);
        return;
      }
      const ok = equivalent(input, item.answer, item.vars);
      setVerdict(ok ? "correct" : "wrong");
      setMessage(t(lang, ok ? "correct" : "incorrect"));
      if (ok) burstFrom(checkBtn.current, { count: 60, power: 1.1 });
    } catch {
      setVerdict("error");
      setMessage(t(lang, "parseError"));
    }
  };

  const typeAnswer = async () => {
    setTyping(true);
    setHint(false);
    const text = item.answer;
    for (let i = 1; i <= text.length; i++) {
      setInput(text.slice(0, i));
      await new Promise((r) => setTimeout(r, 18));
    }
    setTyping(false);
    setVerdict("revealed");
    setMessage("");
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") check();
    if (e.key !== "Tab" || e.shiftKey) return;
    e.preventDefault();
    const now = performance.now();
    if (now - lastTab.current < DOUBLE_TAB_MS) {
      lastTab.current = 0;
      typeAnswer();
    } else {
      lastTab.current = now;
      setHint(true);
      setTimeout(() => setHint(false), DOUBLE_TAB_MS + 200);
    }
  };

  return (
    <article className={`deriv-ex card verdict-${verdict}`}>
      <header className="ex-head">
        <span className="ex-num">{String(index + 1).padStart(2, "0")}</span>
        <h4>{pick(item.title, lang)}</h4>
      </header>
      <p className="deriv-prompt" dangerouslySetInnerHTML={{ __html: renderRich(pick(item.prompt, lang)) }} />

      <div className="deriv-input-row">
        <div className={`deriv-input-wrap${typing ? " typing" : ""}`}>
          <input
            className="deriv-input"
            value={input}
            placeholder={t(lang, "yourAnswer")}
            spellCheck={false}
            onChange={(e) => { setInput(e.target.value); setVerdict("idle"); }}
            onKeyDown={onKeyDown}
          />
          {hint && <span className="deriv-tab-hint">{t(lang, "tabAgain")}</span>}
        </div>
        <button ref={checkBtn} className="btn btn-primary" onClick={check} disabled={!input.trim() || typing}>{t(lang, "check")}</button>
      </div>
      <div className="deriv-preview">
        {preview && <span className="deriv-tex" dangerouslySetInnerHTML={{ __html: preview }} />}
        {message && <span className={`deriv-verdict ${verdict}`}>{message}</span>}
      </div>

      <button className="deriv-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? "▾" : "▸"} {t(lang, open ? "hideSolution" : "showSolution")}
      </button>
      <div className={`deriv-solution${open ? " open" : ""}`}>
        <div>
          <p dangerouslySetInnerHTML={{ __html: renderRich(pick(item.solution, lang)) }} />
        </div>
      </div>
    </article>
  );
}

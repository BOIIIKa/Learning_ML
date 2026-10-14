import { python } from "@codemirror/lang-python";
import { EditorView, basicSetup } from "codemirror";
import { useEffect, useRef, useState } from "react";
import { pick, t, type Bilingual, type Lang } from "../../i18n/ui";
import { renderRich } from "../../lib/rich";
import { runCheck, stop, type CheckResult, type Status } from "../../runtime/python";
import { burstFrom } from "../ui/fx";
import TracePlayer from "../viz/TracePlayer";
import { blanksExtension, fillBlank, remainingBlanks, type Blank } from "./blanks";
import { editorTheme } from "./editorTheme";
import "./exercise.css";

export interface CodeItem {
  id: string;
  kind: "implement" | "library";
  title: Bilingual;
  intro: Bilingual;
  packages: string[];
  skeleton: Bilingual;
  blanks: (Blank & { hint: Bilingual })[];
  check: string;
  metrics?: { key: string; label: Bilingual }[];
}

const METRIC_COLORS = ["var(--accent)", "var(--accent-2)", "var(--warning)"];

const draftKey = (id: string) => `learnml:draft:${id}`;
const loadDraft = (id: string) => {
  try { return localStorage.getItem(draftKey(id)); } catch { return null; }
};
const saveDraft = (id: string, code: string | null) => {
  try { code === null ? localStorage.removeItem(draftKey(id)) : localStorage.setItem(draftKey(id), code); } catch { /* storage unavailable */ }
};

export default function CodeExercise({ item, runtime, lang }: { item: CodeItem; runtime: Record<string, string>; lang: Lang }) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const runBtn = useRef<HTMLButtonElement>(null);
  const [left, setLeft] = useState<string[]>(item.blanks.map((b) => b.id));
  const [filling, setFilling] = useState(false);
  const [status, setStatus] = useState<Status | "idle">("idle");
  const [result, setResult] = useState<CheckResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const doc = loadDraft(item.id) ?? item.skeleton[lang];
    const v = new EditorView({
      parent: host.current!,
      doc,
      extensions: [
        basicSetup,
        python(),
        editorTheme,
        blanksExtension(item.blanks, t(lang, "tabAgain"), (l, f) => { setLeft(l); setFilling(f); }),
        EditorView.updateListener.of((u) => u.docChanged && saveDraft(item.id, u.state.doc.toString())),
      ],
    });
    view.current = v;
    setLeft(remainingBlanks(v.state));
    return () => v.destroy();
  }, [item, lang]);

  const reset = () => {
    const v = view.current!;
    v.dispatch({ changes: { from: 0, to: v.state.doc.length, insert: item.skeleton[lang] }, userEvent: "reset" });
    saveDraft(item.id, null);
    setResult(null);
    setError(null);
  };

  const fillAll = async () => {
    for (const id of remainingBlanks(view.current!.state)) await fillBlank(view.current!, item.blanks, id);
  };

  const run = async () => {
    setError(null);
    setStatus("booting");
    try {
      const r = await runCheck(
        { packages: item.packages, files: runtime, student: view.current!.state.doc.toString(), check: item.check },
        setStatus,
      );
      setResult(r);
      if (r.tests.length && r.tests.every((x) => x.passed)) {
        setTimeout(() => burstFrom(runBtn.current, { count: 120, power: 1.6 }), 250);
      }
    } catch (e) {
      if (String(e).includes("stopped")) setError(null);
      else setError(String(e));
    } finally {
      setStatus("idle");
    }
  };

  const passed = result?.tests.filter((x) => x.passed).length ?? 0;
  const allPassed = !!result && result.tests.length > 0 && passed === result.tests.length;
  const busy = status !== "idle";

  return (
    <article className={`code-ex card${allPassed ? " solved" : ""}`}>
      <header className="ex-head">
        <span className={`ex-kind ${item.kind}`}>{t(lang, item.kind === "implement" ? "implementEx" : "libraryEx")}</span>
        <h4>{pick(item.title, lang)}</h4>
        <p className="muted" dangerouslySetInnerHTML={{ __html: renderRich(pick(item.intro, lang)) }} />
      </header>

      <div className="ex-toolbar">
        <span className="ex-blanks" title={t(lang, "tabHint")}>
          <kbd>Tab</kbd><kbd>Tab</kbd> {t(lang, "tabHint")}
        </span>
        <span className={`ex-left${left.length === 0 && !filling ? " done" : ""}`}>{t(lang, "blanksLeft", { n: left.length })}</span>
      </div>
      <div className="ex-editor" ref={host} />

      <div className="ex-actions">
        <button ref={runBtn} className="btn btn-primary" onClick={run} disabled={busy || filling}>
          {busy ? <span className="spinner" /> : "▶"} {busy ? t(lang, "running") : t(lang, "run")}
        </button>
        {busy && <button className="btn" onClick={stop}>■ Stop</button>}
        <button className="btn" onClick={fillAll} disabled={busy || filling || left.length === 0}>✦ {t(lang, "fillAll")}</button>
        <button className="btn" onClick={reset} disabled={busy}>↺ {t(lang, "reset")}</button>
      </div>

      {status === "booting" && <p className="ex-status">{t(lang, "loadingPython")}</p>}
      {error && <p className="ex-error">{t(lang, "runtimeError")}: {error}</p>}

      {result && (
        <div className="ex-results">
          <div className={`ex-summary${allPassed ? " good" : ""}`}>
            {allPassed ? `🎉 ${t(lang, "allPassed")}` : t(lang, "passedCount", { k: passed, n: result.tests.length })}
          </div>
          <ul className="ex-tests">
            {result.tests.map((x, i) => (
              <li key={`${i}-${x.passed}-${x.message}`} className={x.passed ? "pass" : "fail"} style={{ animationDelay: `${i * 70}ms` }}>
                <svg viewBox="0 0 24 24" className="ex-icon" aria-hidden="true">
                  {x.passed ? <path d="M5 12.5l4.5 4.5L19 7.5" /> : <path d="M7 7l10 10M17 7L7 17" />}
                </svg>
                <span>{pick(x.name, lang)}</span>
                {x.message && <code className="ex-msg">{x.message}</code>}
              </li>
            ))}
          </ul>
          {result.traces.length > 0 && (
            <div className="ex-trace">
              <h5>{t(lang, "traceTitle")}</h5>
              <TracePlayer
                traces={result.traces.slice(0, 2)}
                lang={lang}
                autoplay
                metrics={(item.metrics ?? []).map((m, i) => ({ key: m.key, label: pick(m.label, lang), color: METRIC_COLORS[i % 3], dashed: i > 0 }))}
              />
            </div>
          )}
        </div>
      )}
    </article>
  );
}

import { lazy, Suspense, useEffect, useRef, useState, type ReactNode } from "react";
import { t, type Lang } from "../../i18n/ui";
import type { CodeItem } from "./CodeExercise";
import type { DerivationItem } from "./DerivationExercise";

// Split chunks: mathjs (derivations) and CodeMirror (code) load only when their exercise nears
// the viewport, instead of one 1 MB bundle when the practice section appears.
const DerivationExercise = lazy(() => import("./DerivationExercise"));
const CodeExercise = lazy(() => import("./CodeExercise"));

/** Renders a placeholder of `minHeight` until the slot is close to the viewport. */
function WhenNear({ minHeight, children }: { minHeight: number; children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [near, setNear] = useState(false);
  useEffect(() => {
    const io = new IntersectionObserver((es) => es.some((e) => e.isIntersecting) && (setNear(true), io.disconnect()), { rootMargin: "600px 0px" });
    io.observe(ref.current!);
    return () => io.disconnect();
  }, []);
  const ph = <div className="ex-placeholder card" style={{ minHeight }} aria-busy="true" />;
  return <div ref={ref}>{near ? <Suspense fallback={ph}>{children}</Suspense> : ph}</div>;
}

/** The chapter's practice section: derivation drills first, then the hands-on training. */
export default function Practice({ lang, derivation, code, runtime }: {
  lang: Lang;
  derivation: DerivationItem[];
  code: CodeItem[];
  runtime: Record<string, string>;
}) {
  return (
    <div className="practice">
      <h3>{t(lang, "derivationEx")}</h3>
      <p className="muted"><kbd>Tab</kbd><kbd>Tab</kbd> {t(lang, "tabHint")}</p>
      {derivation.map((item, i) => (
        <WhenNear key={item.id} minHeight={170}><DerivationExercise item={item} lang={lang} index={i} /></WhenNear>
      ))}
      {code.map((item) => (
        <WhenNear key={item.id} minHeight={520}><CodeExercise item={item} runtime={runtime} lang={lang} /></WhenNear>
      ))}
    </div>
  );
}

import { t, type Lang } from "../../i18n/ui";
import CodeExercise, { type CodeItem } from "./CodeExercise";
import DerivationExercise, { type DerivationItem } from "./DerivationExercise";

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
      {derivation.map((item, i) => <DerivationExercise key={item.id} item={item} lang={lang} index={i} />)}
      {code.map((item) => <CodeExercise key={item.id} item={item} runtime={runtime} lang={lang} />)}
    </div>
  );
}

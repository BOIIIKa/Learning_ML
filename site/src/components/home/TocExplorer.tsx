import { useState } from "react";
import { href, pick, t, type Bilingual, type Lang } from "../../i18n/ui";
import { renderRich } from "../../lib/rich";
import "./toc.css";

interface Ch {
  id: string;
  slug: string;
  title: Bilingual;
  paradigm: string;
  tags: { task: string[]; strategy: string[]; algorithm: string[] };
  triad?: { model: Bilingual; strategy: Bilingual; algorithm: Bilingual };
}

type View = "part" | "task" | "strategy" | "algorithm";

/** The course map. Same chapters, regrouped by part or by the triad tags. */
export default function TocExplorer({ lang, parts, tags, ready, courses = [] }: {
  lang: Lang;
  parts: { id: string; title: Bilingual; summary: Bilingual; chapters: Ch[] }[];
  tags: Record<"task" | "strategy" | "algorithm", Record<string, Bilingual>>;
  ready: string[];
  /** Slugs that also have an animated course. */
  courses?: string[];
}) {
  const [view, setView] = useState<View>("part");
  const chapters = parts.flatMap((p) => p.chapters);

  const groups: { key: string; title: string; summary?: string; items: Ch[] }[] =
    view === "part"
      ? parts.map((p) => ({ key: p.id, title: pick(p.title, lang), summary: pick(p.summary, lang), items: p.chapters }))
      : Object.entries(tags[view])
          .map(([k, label]) => ({ key: k, title: pick(label, lang), items: chapters.filter((c) => c.tags[view].includes(k)) }))
          .filter((g) => g.items.length > 0);

  const views: [View, Parameters<typeof t>[1]][] = [["part", "byPart"], ["task", "byTask"], ["strategy", "byStrategy"], ["algorithm", "byAlgorithm"]];

  return (
    <div className="toc">
      <div className="toc-tabs" role="tablist">
        {views.map(([v, key]) => (
          <button key={v} role="tab" aria-selected={view === v} className={`toc-tab${view === v ? " on" : ""}`} onClick={() => setView(v)}>
            {t(lang, key)}
          </button>
        ))}
      </div>
      <div className="toc-groups" key={view}>
        {groups.map((g, gi) => (
          <section key={g.key} className="toc-group" style={{ animationDelay: `${gi * 40}ms` }}>
            <h3>{g.title}</h3>
            {g.summary && <p className="muted toc-summary">{g.summary}</p>}
            <div className="toc-cards">
              {g.items.map((c) => {
                const isReady = ready.includes(c.slug);
                const body = (
                  <>
                    <span className="toc-id">{c.id}</span>
                    <span className="toc-title">{pick(c.title, lang)}</span>
                    {c.triad && view !== "part" && (
                      <span className="toc-triad" dangerouslySetInnerHTML={{ __html: renderRich(pick(c.triad[view === "task" ? "model" : view], lang)) }} />
                    )}
                    <span className={`toc-badge${isReady ? " ready" : ""}`}>{t(lang, isReady ? "available" : "planned")}</span>
                  </>
                );
                return isReady ? (
                  <div key={c.id} className="toc-card ready">
                    <a className="toc-main" href={href(`${lang}/chapters/${c.slug}/`)} aria-label={pick(c.title, lang)} />
                    {body}
                    {courses.includes(c.slug) && (
                      <a className="toc-course" href={href(`${lang}/course/${c.slug}/`)} title={t(lang, "course")}>▶ {t(lang, "course")}</a>
                    )}
                  </div>
                ) : (
                  <div key={c.id} className="toc-card">{body}</div>
                );
              })}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}

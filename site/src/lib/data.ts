// Typed access to the generated outline / exercise bundles and the authored chapter content.
import outlineJson from "../generated/outline.json";
import exercisesJson from "../generated/exercises.json";
import type { Bilingual, Lang } from "../i18n/ui";

export interface Chapter {
  id: string;
  slug: string;
  title: Bilingual;
  paradigm: string;
  sources: Record<string, string[]>;
  prereqs: string[];
  tags: { task: string[]; strategy: string[]; algorithm: string[] };
  goals: Bilingual[];
  triad?: { model: Bilingual; strategy: Bilingual; algorithm: Bilingual };
  derivations: Bilingual[];
  visuals: { id: string; title: Bilingual; interaction: Bilingual }[];
  frontier: { ref: string; note: Bilingual }[];
}

export interface Part {
  id: string;
  title: Bilingual;
  summary: Bilingual;
  chapters: Chapter[];
}

export const outline = outlineJson as unknown as {
  meta: { title: Bilingual; audience: Bilingual; license: Bilingual };
  books: Record<string, { title: string; authors: string; url: string }>;
  tags: Record<"task" | "strategy" | "algorithm", Record<string, Bilingual>>;
  parts: Part[];
};

export const exercises = exercisesJson as unknown as {
  runtime: Record<string, string>;
  chapters: Record<string, { derivation: any[]; code: any[] }>;
};

export const allChapters: Chapter[] = outline.parts.flatMap((p) => p.chapters);

/** Authored chapter pages: src/content/<lang>/<slug>.mdx */
const pages = import.meta.glob("../content/*/*.mdx", { eager: true }) as Record<string, any>;

export function chapterPage(lang: Lang, slug: string) {
  return pages[`../content/${lang}/${slug}.mdx`];
}

const decks = import.meta.glob("../slides/*/*.mdx", { eager: true }) as Record<string, any>;
const recaps = import.meta.glob("../recaps/*/*.mdx", { eager: true }) as Record<string, any>;

export const deck = (lang: Lang, slug: string) => decks[`../slides/${lang}/${slug}.mdx`];
export const recap = (lang: Lang, slug: string) => recaps[`../recaps/${lang}/${slug}.mdx`];

/** Slugs that have content in every language. */
export const readySlugs: string[] = allChapters
  .map((c) => c.slug)
  .filter((slug) => chapterPage("zh", slug) && chapterPage("en", slug));

export const readyChapters = allChapters.filter((c) => readySlugs.includes(c.slug));

/** Handbook entries: src/handbook/<lang>/<slug>.mdx with frontmatter { title, summary, chapters }. */
const handbookPages = import.meta.glob("../handbook/*/*.mdx", { eager: true }) as Record<string, any>;

export interface HandbookEntry {
  slug: string;
  title: string;
  summary: string;
  chapters: string[];
  Content: any;
}

export function handbook(lang: Lang): HandbookEntry[] {
  return Object.entries(handbookPages)
    .filter(([path]) => path.startsWith(`../handbook/${lang}/`))
    .map(([path, mod]) => ({
      slug: path.split("/").pop()!.replace(/\.mdx$/, ""),
      title: mod.frontmatter.title,
      summary: mod.frontmatter.summary,
      chapters: mod.frontmatter.chapters ?? [],
      Content: mod.default,
    }));
}

/** Slugs that also have an animated course deck in every language. */
export const courseSlugs: string[] = readySlugs.filter((slug) => deck("zh", slug) && deck("en", slug));

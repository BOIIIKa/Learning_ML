# Chapter templates

`python3 scripts/new_chapter.py <id>` copies these files into place for chapter `<id>` (two digits,
as in `curriculum/outline.yaml`), filling in `{{ID}}`, `{{SLUG}}`, `{{TITLE_ZH}}`, `{{TITLE_EN}}`,
`{{CH}}` (= `ch<id>`), `{{Engine}}` and `{{Widget}}`.

Every scaffolded file contains `TODO(scaffold)` markers. `site/src/lib/scaffold.test.ts` fails while
any marker remains in published content, so a half-written chapter cannot be merged by accident.

| Template | Goes to |
|---|---|
| `notes.zh.mdx`, `notes.en.mdx` | `site/src/content/{zh,en}/<slug>.mdx` (the notes page; makes the chapter "ready") |
| `deck.zh.mdx`, `deck.en.mdx` | `site/src/slides/{zh,en}/<slug>.mdx` (the animated course) |
| `recap.zh.mdx`, `recap.en.mdx` | `site/src/recaps/{zh,en}/<slug>.mdx` (one-page summary) |
| `handbook.zh.mdx`, `handbook.en.mdx` | `site/src/handbook/{zh,en}/<slug>.mdx` |
| `engine.ts`, `engine.test.ts` | `site/src/engines/<slug>.ts` (+ test) — pure TS maths for the widgets |
| `Widget.tsx` | `site/src/components/chapters/ch<id>/<Widget>.tsx` |
| `exercises.yaml`, `impl.py`, `impl_check.py`, `library.py`, `library_check.py` | `exercises/ch<id>/` |

See `.claude/skills/new-chapter/SKILL.md` for the full workflow.

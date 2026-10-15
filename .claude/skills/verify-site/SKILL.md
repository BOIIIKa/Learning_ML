---
name: verify-site
description: How to run and visually verify the Learning_ML site locally or in a cloud sandbox - dev server with a local Pyodide, the Playwright verify scripts for notes pages, animated courses and themes, and the full pre-push check list. Use before committing site changes, when debugging hydration/console errors, or when asked to check how something looks.
---

# Verifying the site

## Dev server

```bash
cd site
npm ci                                   # once
python3 ../scripts/build_exercises.py    # regenerate exercises.json after editing exercises/
npx astro dev --port 4321 --host 127.0.0.1   # Astro 7 daemonizes; stop with `npx astro dev stop`
```

- First time on a machine: `bash scripts/setup_local.sh` (`.venv`, `npm ci`, Playwright's Chromium,
  first checks); see `docs/LOCAL_DEV.md`.
- **Pyodide**: by default it loads from jsDelivr. Offline or in sandboxes without that host, run
  `bash scripts/fetch_pyodide.sh` (downloads 0.29.3 into `.cache/` and links `site/public/pyodide`)
  and start the server with `PUBLIC_PYODIDE_URL=/pyodide/`. **Remove the link before `astro build`
  or committing.**
- After `astro build`, the dev server's Vite cache can go stale ("504 Outdated Optimize Dep",
  islands fail to hydrate): `npx astro dev stop; rm -rf node_modules/.vite`, restart.

## Scripts (`site/scripts/verify/`; Playwright is a dev dependency, or set `PLAYWRIGHT_MODULE`)

| Command | Checks |
|---|---|
| `node scripts/verify/chapter.mjs <slug> [zh] [dark] [outDir]` | screenshots every widget; fills and runs every code exercise in Pyodide; exits non-zero on failed tests or console errors |
| `node scripts/verify/course.mjs walk zh <slug…>` | steps through every beat of the decks; console must stay clean |
| `node scripts/verify/course.mjs film <slug> zh "#5" "2,6,10"` | plays from slide 5 and screenshots at those seconds (morphs, drives, pacing) |
| `node scripts/verify/pages.mjs <slug>` | home, handbook, chapter in both themes; scrolls to hydrate every island; mobile overflow at 390 px |
| `node scripts/verify/hydration.mjs /zh/chapters/<slug>/ [dark]` | prints React's hydration diffs (+ client / − server) and the island they occur in |
| `node scripts/verify/mathdiff.mjs` | compares `Math.*` bit for bit between Node and Chromium (cos, sin, fractional pow differ) |

Screenshots go to `verify-out/` (git-ignored). **Open and look at them** — layout overflow,
unreadable colours and wrong widget states don't show up as errors.

## Before pushing

```bash
cd site && npx tsc --noEmit -p . && npx vitest run && npx astro build
cd .. && python -m pytest -q tests && python3 scripts/build_outline.py --check
```

CI (`.github/workflows/ci.yml`) runs the same; Pages (`pages.yml`) deploys `main` with
`SITE_BASE=/Learning_ML/`. github.io is not reachable from the cloud sandbox, so check the deploy
via the Actions run status.

## Common errors and their causes

| Symptom | Cause / fix |
|---|---|
| "A tree hydrated but some attributes … didn't match" | unrounded float in SVG, `Math.random()` in render, multi-value or custom-property `style` in SSR, DOM mutated before hydration, or a chaotic computation amplifying a last-bit `Math.cos` difference (run `hydration.mjs`, then `mathdiff.mjs`) |
| Display math renders inline | single-line `$$…$$`; use a multi-line block |
| MDX compile error near `<` or `{` | escape in prose (`$p \lt n$`), no `style=""` strings |
| KaTeX glyphs misaligned | katex CSS version ≠ rehype-katex's katex (pinned 0.16.x) |
| Run button executes half-filled code | it is disabled while the fill animation runs; keep that guard |

#!/usr/bin/env python3
"""Scaffold a new chapter from templates/chapter/.

    python3 scripts/new_chapter.py 04                  # widget name defaults to <Slug>Scenario
    python3 scripts/new_chapter.py 04 --widget LogisticPlayground
    python3 scripts/new_chapter.py 04 --dry-run

Titles and the slug come from curriculum/outline.yaml. Existing files are never overwritten
(unless --force). Every generated file carries TODO(scaffold) markers; the site's test suite
fails while any remain in published content (site/src/lib/scaffold.test.ts).
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TPL = ROOT / "templates" / "chapter"


def find_chapter(cid: str) -> dict:
    outline = yaml.safe_load((ROOT / "curriculum" / "outline.yaml").read_text(encoding="utf-8"))
    for part in outline["parts"]:
        for ch in part["chapters"]:
            if ch["id"] == cid:
                return ch
    sys.exit(f"chapter {cid!r} not found in curriculum/outline.yaml")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id", help="two-digit chapter id, e.g. 04")
    ap.add_argument("--widget", help="name of the first widget component (PascalCase)")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ch = find_chapter(args.id.zfill(2))
    cid, slug = ch["id"], ch["slug"]
    pascal = "".join(w.capitalize() for w in slug.split("-"))
    subs = {
        "{{ID}}": cid,
        "{{SLUG}}": slug,
        "{{TITLE_ZH}}": ch["title"]["zh"],
        "{{TITLE_EN}}": ch["title"]["en"],
        "{{CH}}": f"ch{cid}",
        "{{Widget}}": args.widget or f"{pascal}Scenario",
        "{{Engine}}": slug,
    }
    site = ROOT / "site" / "src"
    plan = {
        "notes.zh.mdx": site / "content" / "zh" / f"{slug}.mdx",
        "notes.en.mdx": site / "content" / "en" / f"{slug}.mdx",
        "deck.zh.mdx": site / "slides" / "zh" / f"{slug}.mdx",
        "deck.en.mdx": site / "slides" / "en" / f"{slug}.mdx",
        "recap.zh.mdx": site / "recaps" / "zh" / f"{slug}.mdx",
        "recap.en.mdx": site / "recaps" / "en" / f"{slug}.mdx",
        "handbook.zh.mdx": site / "handbook" / "zh" / f"{slug}.mdx",
        "handbook.en.mdx": site / "handbook" / "en" / f"{slug}.mdx",
        "engine.ts": site / "engines" / f"{slug}.ts",
        "engine.test.ts": site / "engines" / f"{slug}.test.ts",
        "Widget.tsx": site / "components" / "chapters" / f"ch{cid}" / f"{subs['{{Widget}}']}.tsx",
        "exercises.yaml": ROOT / "exercises" / f"ch{cid}" / "exercises.yaml",
        "impl.py": ROOT / "exercises" / f"ch{cid}" / "impl.py",
        "impl_check.py": ROOT / "exercises" / f"ch{cid}" / "impl_check.py",
        "library.py": ROOT / "exercises" / f"ch{cid}" / "library.py",
        "library_check.py": ROOT / "exercises" / f"ch{cid}" / "library_check.py",
    }
    for src, dst in plan.items():
        text = (TPL / src).read_text(encoding="utf-8")
        for k, v in subs.items():
            text = text.replace(k, v)
        rel = dst.relative_to(ROOT)
        if dst.exists() and not args.force:
            print(f"skip   {rel} (exists)")
            continue
        print(f"{'would write' if args.dry_run else 'write'}  {rel}")
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text, encoding="utf-8")
    print(f"\nNext: follow .claude/skills/new-chapter/SKILL.md for chapter {cid} ({ch['title']['en']}).")


if __name__ == "__main__":
    main()

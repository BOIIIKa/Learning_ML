"""Validate curriculum/outline.yaml and render OUTLINE.zh.md / OUTLINE.en.md.

Usage:
    python3 scripts/build_outline.py           # validate + write
    python3 scripts/build_outline.py --check   # validate + fail if outputs are stale
"""

import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "curriculum" / "outline.yaml"
LANGS = ("zh", "en")

PARADIGMS = {
    "foundation": {"zh": "基础", "en": "Foundation"},
    "supervised": {"zh": "监督", "en": "Supervised"},
    "unsupervised": {"zh": "无监督", "en": "Unsupervised"},
    "deep": {"zh": "深度", "en": "Deep"},
    "generative": {"zh": "生成", "en": "Generative"},
    "rl": {"zh": "强化", "en": "RL"},
    "frontier": {"zh": "前沿", "en": "Frontier"},
    "meta": {"zh": "方法", "en": "Methodology"},
}

UI = {
    "toc": {"zh": "目录", "en": "Table of Contents"},
    "chapter": {"zh": "第 {id} 章", "en": "Ch. {id}"},
    "paradigm": {"zh": "范式", "en": "Paradigm"},
    "sources": {"zh": "对应原书章节", "en": "Source chapters"},
    "prereqs": {"zh": "前置章节", "en": "Prerequisites"},
    "goals": {"zh": "学习目标", "en": "Learning goals"},
    "triad": {"zh": "三要素", "en": "The triad"},
    "model": {"zh": "模型", "en": "Model"},
    "strategy": {"zh": "策略", "en": "Strategy"},
    "algorithm": {"zh": "算法", "en": "Algorithm"},
    "derivations": {"zh": "严谨推导", "en": "Rigorous derivations"},
    "visuals": {"zh": "可视化组件", "en": "Interactive visuals"},
    "frontier": {"zh": "前沿跟进", "en": "Frontier reading"},
    "none": {"zh": "无", "en": "none"},
    "self": {"zh": "自编（无对应原书章节）", "en": "Original (no source-book chapter)"},
    "index": {"zh": "交叉索引", "en": "Cross-indices"},
    "by_task": {"zh": "按任务", "en": "By task"},
    "by_strategy": {"zh": "按策略（损失）", "en": "By strategy (loss)"},
    "by_algorithm": {"zh": "按算法（优化 / 推断）", "en": "By algorithm (optimization / inference)"},
    "coverage": {"zh": "原书覆盖表", "en": "Source-book coverage"},
    "book_ch": {"zh": "原书章", "en": "Book chapter"},
    "our_ch": {"zh": "本课程章节", "en": "Our chapters"},
    "omitted": {"zh": "未纳入", "en": "Not included"},
    "stats": {
        "zh": "共 {parts} 个部分、{chapters} 章、{visuals} 个可视化组件、{papers} 篇前沿文献。",
        "en": "{parts} parts, {chapters} chapters, {visuals} interactive visuals, {papers} frontier papers.",
    },
    "generated": {
        "zh": "> 本文件由 `scripts/build_outline.py` 从 `curriculum/outline.yaml` 自动生成，请勿手改。[English](OUTLINE.en.md)",
        "en": "> Generated from `curriculum/outline.yaml` by `scripts/build_outline.py` — do not edit by hand. [中文](OUTLINE.zh.md)",
    },
}


def fail(msg):
    sys.exit(f"outline.yaml: {msg}")


def check_bilingual(obj, where):
    if not isinstance(obj, dict) or any(not obj.get(lang) for lang in LANGS):
        fail(f"{where} must have non-empty zh and en")


def validate(data):
    tags = data["tags"]
    seen_ids, seen_viz = [], set()
    for book in data["books"]:
        if not data["books"][book].get("title"):
            fail(f"book {book} has no title")
    for part in data["parts"]:
        check_bilingual(part["title"], f"{part['id']}.title")
        check_bilingual(part["summary"], f"{part['id']}.summary")
        for ch in part["chapters"]:
            cid = ch["id"]
            if cid in seen_ids:
                fail(f"duplicate chapter id {cid}")
            check_bilingual(ch["title"], f"{cid}.title")
            if ch["paradigm"] not in PARADIGMS:
                fail(f"{cid}: unknown paradigm {ch['paradigm']}")
            for book in ch["sources"]:
                if book not in data["books"]:
                    fail(f"{cid}: unknown book {book}")
            for pre in ch["prereqs"]:
                if pre not in seen_ids:
                    fail(f"{cid}: prerequisite {pre} must be an earlier chapter")
            for kind, values in ch["tags"].items():
                for v in values:
                    if v not in tags[kind]:
                        fail(f"{cid}: unknown {kind} tag {v}")
            for i, item in enumerate(ch["goals"] + ch["derivations"]):
                check_bilingual(item, f"{cid} item {i}")
            for key in ("model", "strategy", "algorithm"):
                if "triad" in ch:
                    check_bilingual(ch["triad"][key], f"{cid}.triad.{key}")
            for viz in ch["visuals"]:
                if viz["id"] in seen_viz:
                    fail(f"duplicate visual id {viz['id']}")
                seen_viz.add(viz["id"])
                check_bilingual(viz["title"], f"{viz['id']}.title")
                check_bilingual(viz["interaction"], f"{viz['id']}.interaction")
            for ref in ch["frontier"]:
                check_bilingual(ref["note"], f"{cid} frontier note")
            seen_ids.append(cid)
    # Every numbered chapter of every source book is either used or explicitly omitted.
    for book, meta in data["books"].items():
        used = {c for ch in chapters(data) for c in ch["sources"].get(book, [])}
        used |= set(meta.get("omitted", {}))
        missing = [str(n) for n in range(1, meta["chapters"] + 1) if str(n) not in used]
        if missing:
            fail(f"{book} chapters {', '.join(missing)} are neither used nor omitted")


def chapters(data):
    for part in data["parts"]:
        yield from part["chapters"]


def anchor(ch):
    return f"ch-{ch['id']}"


def ch_link(ch, lang):
    return f"[{ch['id']} {ch['title'][lang]}](#{anchor(ch)})"


def sources_text(ch, data, lang):
    if not ch["sources"]:
        return UI["self"][lang]
    return "; ".join(f"{book} {', '.join(chs)}" for book, chs in ch["sources"].items())


def book_chapter_key(label):
    # Numbered chapters first in numeric order, then appendices.
    return (0, int(label), "") if label.isdigit() else (1, 0, label)


def render(data, lang):
    by_id = {ch["id"]: ch for ch in chapters(data)}
    out = [f"# {data['meta']['title'][lang]}", "", UI["generated"][lang], ""]
    out += [f"- **{'受众' if lang == 'zh' else 'Audience'}**: {data['meta']['audience'][lang]}"]
    out += [f"- **{'许可' if lang == 'zh' else 'License'}**: {data['meta']['license'][lang]}"]
    out += ["- **" + ("底本" if lang == "zh" else "Source books") + "**: "
            + "; ".join(f"`{k}` [{b['title']}]({b['url']})" for k, b in data["books"].items())]
    n_viz = sum(len(ch["visuals"]) for ch in chapters(data))
    n_ref = sum(len(ch["frontier"]) for ch in chapters(data))
    out += ["", UI["stats"][lang].format(parts=len(data["parts"]), chapters=len(by_id),
                                         visuals=n_viz, papers=n_ref), ""]

    out += [f"## {UI['toc'][lang]}", ""]
    for part in data["parts"]:
        out.append(f"- **{part['title'][lang]}**")
        for ch in part["chapters"]:
            out.append(f"  - {ch_link(ch, lang)} · `{PARADIGMS[ch['paradigm']][lang]}`")
    out += [f"- **[{UI['index'][lang]}](#index)**", ""]

    for part in data["parts"]:
        out += [f"## {part['title'][lang]}", "", f"*{part['summary'][lang]}*", ""]
        for ch in part["chapters"]:
            out += [f'<a id="{anchor(ch)}"></a>', "",
                    f"### {ch['id']} {ch['title'][lang]}", ""]
            prereqs = ", ".join(ch_link(by_id[p], lang) for p in ch["prereqs"]) or UI["none"][lang]
            out += [f"- **{UI['paradigm'][lang]}**: {PARADIGMS[ch['paradigm']][lang]}",
                    f"- **{UI['sources'][lang]}**: {sources_text(ch, data, lang)}",
                    f"- **{UI['prereqs'][lang]}**: {prereqs}", ""]
            out += [f"**{UI['goals'][lang]}**", ""]
            out += [f"- {g[lang]}" for g in ch["goals"]] + [""]
            if "triad" in ch:
                out += [f"**{UI['triad'][lang]}**", "",
                        f"| {UI['model'][lang]} | {UI['strategy'][lang]} | {UI['algorithm'][lang]} |",
                        "|---|---|---|",
                        "| " + " | ".join(ch["triad"][k][lang] for k in ("model", "strategy", "algorithm")) + " |",
                        ""]
            out += [f"**{UI['derivations'][lang]}**", ""]
            out += [f"{i}. {d[lang]}" for i, d in enumerate(ch["derivations"], 1)] + [""]
            out += [f"**{UI['visuals'][lang]}**", ""]
            out += [f"- `{v['id']}` **{v['title'][lang]}** — {v['interaction'][lang]}" for v in ch["visuals"]]
            out.append("")
            if ch["frontier"]:
                out += [f"**{UI['frontier'][lang]}**", ""]
                out += [f"- {r['ref']} — {r['note'][lang]}" for r in ch["frontier"]] + [""]

    out += ['<a id="index"></a>', "", f"## {UI['index'][lang]}", ""]
    for kind, title in (("task", "by_task"), ("strategy", "by_strategy"), ("algorithm", "by_algorithm")):
        out += [f"### {UI[title][lang]}", ""]
        for key, label in data["tags"][kind].items():
            hits = [ch_link(ch, lang) for ch in chapters(data) if key in ch["tags"][kind]]
            if hits:
                out.append(f"- **{label[lang]}**: {', '.join(hits)}")
        out.append("")

    out += [f"### {UI['coverage'][lang]}", ""]
    for book, meta in data["books"].items():
        cover = defaultdict(list)
        for ch in chapters(data):
            for bch in ch["sources"].get(book, []):
                cover[bch].append(ch_link(ch, lang))
        for bch, reason in meta.get("omitted", {}).items():
            cover[bch].append(f"*{UI['omitted'][lang]}: {reason[lang]}*")
        out += [f"**{book}**", "", f"| {UI['book_ch'][lang]} | {UI['our_ch'][lang]} |", "|---|---|"]
        out += [f"| {bch} | {', '.join(cover[bch])} |" for bch in sorted(cover, key=book_chapter_key)]
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main():
    data = yaml.safe_load(SRC.read_text(encoding="utf-8"))
    validate(data)
    stale = []
    for lang in LANGS:
        path = ROOT / "curriculum" / f"OUTLINE.{lang}.md"
        text = render(data, lang)
        if "--check" in sys.argv:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(path.name)
        else:
            path.write_text(text, encoding="utf-8")
    if stale:
        sys.exit(f"stale: {', '.join(stale)} — run python3 scripts/build_outline.py")


if __name__ == "__main__":
    main()

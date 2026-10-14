"""Bundle the outline and the exercises into JSON for the site.

Reads curriculum/outline.yaml and exercises/ch*/exercises.yaml (plus the .py files they
reference), splits each code exercise into skeleton / solution / blanks with
learnml.blanks, and writes site/src/generated/{outline,exercises}.json.
"""

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from learnml import blanks  # noqa: E402

OUT = ROOT / "site" / "src" / "generated"


def fail(msg):
    sys.exit(f"build_exercises: {msg}")


def runtime_files():
    """The learnml package, shipped into Pyodide's file system."""
    pkg = ROOT / "python" / "learnml"
    return {f"learnml/{p.name}": p.read_text(encoding="utf-8") for p in sorted(pkg.glob("*.py"))}


def build_chapter(chapter_dir, chapter_ids):
    cid = chapter_dir.name.removeprefix("ch")
    if cid not in chapter_ids:
        fail(f"{chapter_dir.name} does not match any chapter id in outline.yaml")
    spec = yaml.safe_load((chapter_dir / "exercises.yaml").read_text(encoding="utf-8"))
    derivation = spec.get("derivation", [])
    for item in derivation:
        for key in ("title", "prompt", "solution"):
            if set(item.get(key, {})) != {"zh", "en"}:
                fail(f"{chapter_dir.name}/{item.get('id')}: {key} needs zh and en")
        if not item.get("answer") or not item.get("vars"):
            fail(f"{chapter_dir.name}/{item['id']}: answer and vars are required")
    code = []
    for item in spec.get("code", []):
        if item["kind"] not in ("implement", "library"):
            fail(f"{chapter_dir.name}/{item['id']}: kind must be implement or library")
        source = (chapter_dir / item["file"]).read_text(encoding="utf-8")
        check = (chapter_dir / item["file"].replace(".py", "_check.py")).read_text(encoding="utf-8")
        code.append({**item, **blanks.parse(source), "check": check})
    return cid, {"derivation": derivation, "code": code}


def main():
    outline = yaml.safe_load((ROOT / "curriculum" / "outline.yaml").read_text(encoding="utf-8"))
    chapter_ids = {ch["id"] for part in outline["parts"] for ch in part["chapters"]}
    chapters = dict(
        build_chapter(d, chapter_ids)
        for d in sorted((ROOT / "exercises").glob("ch*"))
        if (d / "exercises.yaml").exists()
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "outline.json").write_text(json.dumps(outline, ensure_ascii=False), encoding="utf-8")
    bundle = {"runtime": runtime_files(), "chapters": chapters}
    (OUT / "exercises.json").write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

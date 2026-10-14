"""Every exercise: the reference solution passes all checks; the blank skeleton passes none of the blank tests."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from learnml import blanks, checking  # noqa: E402

EXERCISES = sorted(p for p in (ROOT / "exercises").glob("ch*/*.py") if not p.stem.endswith("_check"))


@pytest.mark.parametrize("path", EXERCISES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_solution_passes(path):
    parsed = blanks.parse(path.read_text(encoding="utf-8"))
    check_src = path.with_name(f"{path.stem}_check.py").read_text(encoding="utf-8")
    result = checking.run(parsed["solution"], check_src)
    failed = [(t["name"]["en"], t["message"]) for t in result["tests"] if not t["passed"]]
    assert not failed
    assert result["traces"], "solution should produce at least one trace"


@pytest.mark.parametrize("path", EXERCISES, ids=lambda p: f"{p.parent.name}/{p.name}")
@pytest.mark.parametrize("lang", blanks.LANGS)
def test_skeleton_fails(path, lang):
    parsed = blanks.parse(path.read_text(encoding="utf-8"))
    check_src = path.with_name(f"{path.stem}_check.py").read_text(encoding="utf-8")
    result = checking.run(parsed["skeleton"][lang], check_src)
    assert result["tests"] and not any(t["passed"] for t in result["tests"])

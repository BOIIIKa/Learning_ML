"""Single-source fill-in-the-blank exercises.

An exercise is an ordinary, runnable Python file (the reference solution) in which
some regions are marked as blanks:

    # @blank eta
    # zh: 计算 η = K_ii + K_jj − 2K_ij
    # en: Compute η = K_ii + K_jj − 2K_ij
    eta = K[i, i] + K[j, j] - 2 * K[i, j]
    # @end

`parse` turns that file into the reference solution (markers stripped), a skeleton per
language (each blank replaced by a TODO comment plus `raise NotImplementedError`), and
the list of blanks with their answers, so the editor can fill them on double-Tab.
"""

import re

LANGS = ("zh", "en")
_OPEN = re.compile(r"^(\s*)# @blank ([\w-]+)\s*$")
_CLOSE = re.compile(r"^\s*# @end\s*$")
_HINT = re.compile(r"^\s*# (zh|en): (.*)$")


def parse(source):
    lines = source.splitlines()
    solution, skeletons, blanks = [], {lang: [] for lang in LANGS}, []
    i = 0
    while i < len(lines):
        m = _OPEN.match(lines[i])
        if not m:
            solution.append(lines[i])
            for lang in LANGS:
                skeletons[lang].append(lines[i])
            i += 1
            continue
        indent, blank_id = m.groups()
        i += 1
        hints = {}
        while i < len(lines) and _HINT.match(lines[i]):
            lang, text = _HINT.match(lines[i]).groups()
            hints[lang] = text
            i += 1
        answer = []
        while i < len(lines) and not _CLOSE.match(lines[i]):
            answer.append(lines[i])
            i += 1
        if i == len(lines):
            raise ValueError(f"blank {blank_id!r} has no closing '# @end'")
        if set(hints) != set(LANGS) or not answer:
            raise ValueError(f"blank {blank_id!r} needs zh/en hints and a non-empty answer")
        i += 1  # skip '# @end'
        solution.extend(answer)
        for lang in LANGS:
            skeletons[lang].append(f"{indent}# TODO({blank_id}): {hints[lang]}")
            skeletons[lang].append(f'{indent}raise NotImplementedError("{blank_id}")')
        blanks.append({"id": blank_id, "hint": hints, "answer": "\n".join(answer)})
    if len({b["id"] for b in blanks}) != len(blanks):
        raise ValueError("duplicate blank ids")
    return {
        "solution": "\n".join(solution) + "\n",
        "skeleton": {lang: "\n".join(skeletons[lang]) + "\n" for lang in LANGS},
        "blanks": blanks,
    }

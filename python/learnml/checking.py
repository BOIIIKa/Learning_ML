"""Run an exercise's checks against student code and report bilingual results.

The same entry point runs in CPython (pytest) and in the browser (Pyodide):

    result = run(student_source, check_source)
    # {"tests": [{"name": {...}, "passed": bool, "message": str}], "traces": [...]}
"""

import traceback
import types
import warnings


class Checker:
    def __init__(self):
        self.tests = []
        self.traces = []

    def test(self, zh, en, fn):
        """Run `fn()`; an AssertionError or any exception marks the test failed."""
        try:
            fn()
        except NotImplementedError as e:
            self._add(zh, en, False, f"TODO({e.args[0] if e.args else '?'})")
        except AssertionError as e:
            self._add(zh, en, False, str(e) or "assertion failed")
        except Exception as e:  # student code may raise anything
            self._add(zh, en, False, f"{type(e).__name__}: {e}")
        else:
            self._add(zh, en, True, "")

    def trace(self, fn):
        """Collect a trace for the player; failures here are reported as a test, not raised."""
        try:
            self.traces.append(fn().to_dict())
        except NotImplementedError:
            pass  # unfinished blanks are already reported by the tests
        except Exception as e:
            self._add("生成训练轨迹", "Recording the training trace", False, f"{type(e).__name__}: {e}")

    def _add(self, zh, en, passed, message):
        self.tests.append({"name": {"zh": zh, "en": en}, "passed": passed, "message": message})

    def result(self):
        return {"tests": self.tests, "traces": self.traces}


def load_module(name, source):
    module = types.ModuleType(name)
    exec(compile(source, f"{name}.py", "exec"), module.__dict__)
    return module


def run(student_source, check_source):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # library warnings are noise for students
        return _run(student_source, check_source)


def _run(student_source, check_source):
    checker = Checker()
    try:
        student = load_module("student", student_source)
    except Exception:
        tb = traceback.format_exc(limit=1)
        checker._add("代码可以运行", "Code runs", False, tb.strip().splitlines()[-1])
        return checker.result()
    load_module("check", check_source).check(student, checker)
    return checker.result()

"""Import-graph isolation of the evaluator package (ADR-002; WP-16 test 5b).

Born 2026-10-21 with GOV-009 WP-16. Checkpoint detectors read the RED channel
(engine counters and events). If a learning module could import them, engine
state could reach a reward or a target and the program's whole claim
(learned, not declared) would be void. The rule is therefore mechanical:

    No module under developmental_ai/ outside developmental_ai/evaluator/
    imports developmental_ai.evaluator (by import statement, relative import,
    or importlib/__import__ with a literal string).

Contracts:
    A. THE SCANNER FLAGS EVERY FORM (falsification first). Planted sources
       using each import form must all be reported; a source that does not
       import it must not be. A scanner that cannot see a form proves nothing.
    B. NO LEARNING MODULE IMPORTS THE EVALUATOR. Every .py under
       developmental_ai/ except the evaluator package itself.
    C. THE ADAPTER SIDE DOES NOT EITHER. environments/ and game_template/ are
       inside developmental_ai/ so B covers them; reported separately because a
       game interface importing its own detector is the likeliest mistake.
    D. THE EVALUATOR DOES NOT IMPORT THE LEARNER. Detectors depend only on
       dicts: no developmental_ai module outside the evaluator package.

Run:  PYTHONPATH=. python tests/_evaluator_isolation_smoke.py   (no engine, no torch)
NOT RUN at build time (GOV-011 entry WP-16).
"""
from __future__ import annotations

import ast
import os
import sys

NAME = "evaluator-isolation"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "developmental_ai")
EVAL = os.path.join(PKG, "evaluator")
TARGET = "developmental_ai.evaluator"


def _fail(msg):
    print("[%s] FAIL: %s" % (NAME, msg))
    sys.exit(1)


def _module_name(path):
    rel = os.path.relpath(path, REPO)[:-3].replace(os.sep, ".")
    return rel[:-9] if rel.endswith(".__init__") else rel


def imports_of(source, module_name, is_pkg=False):
    """Absolute dotted names a source imports (statements + literal importlib calls)."""
    out = []
    tree = ast.parse(source)
    base = module_name.split(".")
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            if n.level:
                keep = len(base) - n.level + (1 if is_pkg else 0)
                head = ".".join(base[:max(keep, 0)])
                mod = (head + "." + n.module) if n.module else head
            else:
                mod = n.module or ""
            out.append(mod)
            out += [mod + "." + a.name for a in n.names]
        elif isinstance(n, ast.Call):
            f = n.func
            fname = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if fname in ("import_module", "__import__") and n.args:
                a0 = n.args[0]
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    out.append(a0.value)
    return out


def hits(names, target=TARGET):
    return [x for x in names if x == target or x.startswith(target + ".")]


def py_files(root):
    for d, _, fs in os.walk(root):
        if "__pycache__" in d:
            continue
        for f in fs:
            if f.endswith(".py"):
                yield os.path.join(d, f)


def main():
    # A. falsification of the scanner
    planted = {
        "import developmental_ai.evaluator.registry": True,
        "from developmental_ai.evaluator import detector": True,
        "from developmental_ai.evaluator.registry import get_detector": True,
        "import importlib\nimportlib.import_module('developmental_ai.evaluator.detectors.x')": True,
        "__import__('developmental_ai.evaluator')": True,
        "import os\nfrom developmental_ai.core import loop": False,
    }
    for src, want in planted.items():
        got = bool(hits(imports_of(src, "developmental_ai.core.x")))
        if got != want:
            _fail("A: scanner wrong on %r (got %s, want %s)" % (src, got, want))
    rel = imports_of("from ..evaluator import detector", "developmental_ai.core.x")
    if not hits(rel):
        _fail("A: relative import `from ..evaluator import ...` not flagged: %r" % (rel,))
    rel2 = imports_of("from .. import evaluator", "developmental_ai.core.x")
    if not hits(rel2):
        _fail("A: relative import `from .. import evaluator` not flagged: %r" % (rel2,))
    print("[%s] A. scanner flags every import form (7 planted sources)" % NAME)

    # B / C. no learner module imports the evaluator
    bad_b, bad_c, n = [], [], 0
    for p in py_files(PKG):
        if p.startswith(EVAL + os.sep):
            continue
        n += 1
        src = open(p, encoding="utf-8").read()
        h = hits(imports_of(src, _module_name(p), p.endswith("__init__.py")))
        if h:
            bad_b.append((os.path.relpath(p, REPO), h[0]))
            if os.sep + "environments" + os.sep in p or os.sep + "game_template" + os.sep in p:
                bad_c.append(p)
    if bad_b:
        _fail("B: modules import the evaluator: %s" % bad_b)
    print("[%s] B. none of %d modules outside the evaluator imports it" % (NAME, n))
    if bad_c:
        _fail("C: adapter-side modules import the evaluator: %s" % bad_c)
    print("[%s] C. environments/ and game_template/ do not import it" % (NAME))

    # D. the evaluator does not import the learner
    bad_d = []
    for p in py_files(EVAL):
        src = open(p, encoding="utf-8").read()
        for x in imports_of(src, _module_name(p), p.endswith("__init__.py")):
            if x.startswith("developmental_ai") and not (x == TARGET or x.startswith(TARGET + ".")
                                                           or x == "developmental_ai"):
                bad_d.append((os.path.relpath(p, REPO), x))
    if bad_d:
        _fail("D: the evaluator imports learner modules: %s" % bad_d)
    print("[%s] D. the evaluator imports nothing from the learner" % NAME)
    print("[%s] ALL PASS" % NAME)


if __name__ == "__main__":
    main()

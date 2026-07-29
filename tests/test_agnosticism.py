"""The embodiment-agnosticism gate.

Remoroo enrols cells it does not choose -- brands, DOF counts, limb counts.  A
hardcoded embodiment assumption is therefore a bug in the business model, not
just in the code, and it is the kind of bug that passes every functional test on
the cell it was written against.  So it is checked structurally, by reading the
source, in addition to being checked behaviourally by running the whole suite
over the cell matrix.

The scan tokenises rather than regexing text, so prose in docstrings and comments
is free to say "arm" or "gripper" while code is not.
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

import pytest

from tests.conftest import CELL_IDS, ROOT

CORE = ROOT / "src" / "remoroo_lc"
#: Vendor I/O lives behind the adapter protocol and is the ONLY place a brand may
#: appear.  `rig/` is offline tooling -- recordings, replay, metrics -- that never
#: runs at command rate; it is held to the naming rule but not scanned for the
#: numeric literals a data format legitimately contains.
ADAPTERS = CORE / "adapters"
RIG = CORE / "rig"

#: Identifier fragments that name a kind of machine or a kind of end effector.
#: Split on underscores and compared per fragment, so `warm_start` and `params`
#: are fine while `arm_count` is not.
FORBIDDEN_WORDS = {
    "arm", "arms", "forearm", "upperarm",
    "gripper", "grippers", "grip", "grasp", "grasps", "grasping",
    "hand", "hands", "finger", "fingers", "jaw", "jaws", "wrist", "elbow",
    "shoulder", "leg", "legs", "foot", "feet",
    "xarm", "ufactory", "franka", "panda", "kuka", "universal", "kinova",
    "fanuc", "abb", "doosan", "techman", "realman",
    "bimanual", "humanoid", "quadruped",
}

#: Numeric literals that would encode a DOF count or a chain count.  TASK_DIM is
#: the only fixed dimension in the package and it lives in `constants`.
FORBIDDEN_NUMBERS = {6, 7}


def _core_files() -> list[Path]:
    return sorted(
        p
        for p in CORE.rglob("*.py")
        if ADAPTERS not in p.parents and p.name != "__pycache__"
    )


def test_there_are_core_files_to_check():
    files = _core_files()
    assert len(files) >= 8, files
    assert not any("adapters" in p.parts for p in files)


@pytest.mark.parametrize("path", _core_files(), ids=lambda p: p.name)
def test_core_never_imports_an_adapter(path):
    """Vendor code lives behind the adapter protocol; the core cannot see it."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "adapters" not in alias.name, f"{path.name} imports {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert "adapters" not in mod, f"{path.name} imports from {mod}"


@pytest.mark.parametrize("path", _core_files(), ids=lambda p: p.name)
def test_core_names_no_kind_of_machine(path):
    """No identifier in core code may name an arm, an effector or a brand."""
    offenders = []
    for tok in tokenize.generate_tokens(io.StringIO(path.read_text()).readline):
        if tok.type != tokenize.NAME:
            continue
        for part in tok.string.lower().split("_"):
            if part in FORBIDDEN_WORDS:
                offenders.append((tok.start[0], tok.string))
    assert not offenders, f"{path.name} names an embodiment: {offenders}"


@pytest.mark.parametrize("path", _core_files(), ids=lambda p: p.name)
def test_core_hardcodes_no_dof_count(path):
    """No bare 6 or 7 in core code -- task-space dimension goes through TASK_DIM.

    `constants.py` is the one exemption, because it is where TASK_DIM = 6 is
    written down.  Keeping the exemption to a single file is the point: the gate
    then proves that every other use in the package goes through the name.
    """
    if RIG in path.parents:
        pytest.skip("rig/ is offline tooling; byte offsets are not DOF counts")
    if path.name == "constants.py":
        text = path.read_text()
        assert "TASK_DIM = 6" in text, "the one exempt file must still define TASK_DIM"
        pytest.skip("constants.py defines TASK_DIM; every other file must use it")
    offenders = []
    for tok in tokenize.generate_tokens(io.StringIO(path.read_text()).readline):
        if tok.type != tokenize.NUMBER:
            continue
        try:
            value = ast.literal_eval(tok.string)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, int) and value in FORBIDDEN_NUMBERS:
            offenders.append((tok.start[0], tok.string))
    assert not offenders, (
        f"{path.name} has a hardcoded DOF-shaped literal at lines {offenders}; "
        "use TASK_DIM, POINT_DIM, or a value derived from the cell"
    )


def test_the_gate_would_actually_catch_something(tmp_path):
    """A gate nobody has seen fail is not evidence of anything."""
    bad = tmp_path / "bad.py"
    bad.write_text("from remoroo_lc.adapters import base\nARM_DOF = 6\n")

    tree = ast.parse(bad.read_text())
    assert any(
        isinstance(n, ast.ImportFrom) and "adapters" in (n.module or "")
        for n in ast.walk(tree)
    )
    names, numbers = [], []
    for tok in tokenize.generate_tokens(io.StringIO(bad.read_text()).readline):
        if tok.type == tokenize.NAME and any(
            p in FORBIDDEN_WORDS for p in tok.string.lower().split("_")
        ):
            names.append(tok.string)
        if tok.type == tokenize.NUMBER and ast.literal_eval(tok.string) in FORBIDDEN_NUMBERS:
            numbers.append(tok.string)
    assert names == ["ARM_DOF"]
    assert numbers == ["6"]


def test_prose_is_exempt(tmp_path):
    """Docstrings and comments may use whatever words explain the code best."""
    ok = tmp_path / "ok.py"
    ok.write_text('"""An arm is a chain with a gripper."""\n# 6 DOF example\nX = 5\n')
    offenders = []
    for tok in tokenize.generate_tokens(io.StringIO(ok.read_text()).readline):
        if tok.type == tokenize.NAME and any(
            p in FORBIDDEN_WORDS for p in tok.string.lower().split("_")
        ):
            offenders.append(tok.string)
    assert offenders == []


def test_the_cell_matrix_is_actually_diverse():
    """The suite is only a gate if the matrix spans the cases that differ."""
    from remoroo_lc.schema import load_cell
    from tests.conftest import CELL_FILES

    cells = [load_cell(p) for p in CELL_FILES]
    joint_counts = {c.n_joints for c in cells}
    widths = {c.effector_widths for c in cells}
    assert len(cells) >= 5, CELL_IDS
    assert len(joint_counts) >= 4, f"cells are too alike in DOF: {joint_counts}"
    assert any(0 in w for w in widths), "no effectorless cell in the matrix"
    assert any(len(set(w)) > 1 for w in widths), "no cell with mixed effector widths"
    assert any(c.n_tcps == 1 for c in cells), "no single-TCP cell"
    assert any(c.n_tcps >= 2 for c in cells), "no multi-TCP cell"
    # A cell whose two TCPs share joints, and one where they do not.
    shared = [c for c in cells if len({t.model for t in c.tcps}) < c.n_tcps]
    assert shared, "no cell with chains sharing a model (and therefore joints)"
    assert any(c.n_joints < c.task_dim for c in cells), "no over-constrained cell"
    assert any(c.n_joints > c.task_dim for c in cells), "no redundant cell"

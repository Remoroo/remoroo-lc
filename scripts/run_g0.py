#!/usr/bin/env python3
"""Run the G0 scoreboard: cell + tapes -> controller -> plant -> reports/.

    python scripts/run_g0.py                          # every shipped cell
    python scripts/run_g0.py configs/cells/mixed.yaml # one cell
    python scripts/run_g0.py --write-tapes            # also dump the tapes used

Exit status is non-zero if any cell misses a bar, so this is usable as a gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from remoroo_lc.schema import load_cell  # noqa: E402
from remoroo_lc.scoreboard import score_cell, write_reports  # noqa: E402
from remoroo_lc.tapes import standard_tapes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="*", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "reports")
    ap.add_argument("--write-tapes", action="store_true")
    a = ap.parse_args()

    cell_paths = a.cells or sorted((ROOT / "configs" / "cells").glob("*.yaml"))
    all_pass = True
    for path in cell_paths:
        cell = load_cell(path)
        tapes = standard_tapes(cell)
        if a.write_tapes:
            for t in tapes:
                t.write(a.out / "tapes" / cell.name / f"{t.name}.jsonl")
        print(f"[{cell.name}] {len(tapes)} tapes, n={cell.n_joints}, T={cell.n_tcps} ...")
        report = score_cell(cell, tapes)
        md, js = write_reports(report, a.out)
        verdict = "PASS" if report.passed() else "FAIL"
        print(f"[{cell.name}] {verdict}  ->  {md.relative_to(ROOT)}")
        for name, b in report.bars.items():
            if not b["pass"]:
                print(f"    MISS {name}: {b['value']} (bar {b['bar']})")
        all_pass &= report.passed()
        _ = js
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

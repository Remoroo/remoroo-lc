"""The G0 scoreboard: run tapes through the controller and the plant, and report.

What is measured, and against what:

  tracking    TCP pose error between where the controller COMMANDED the TCP
              (Layer 1's output) and where FK says it actually is.  Measured on
              clean tapes only, and only on ticks where no wall was binding --
              a filter that is correctly refusing to drive into a table is not a
              tracking failure and must not be averaged in as one.
  safety      any pair distance below zero, or any joint outside its limits.
              Measured on everything.  The bar is zero, on every cell.
  smoothness  joint jerk percentiles, and a chatter count: sign changes of joint
              acceleration per second while a wall is binding, which is what
              filter-induced buzzing looks like from the outside.
  filter      fraction of ticks with a binding wall, by tape category.  Clean
              tapes should be near zero; adversarial tapes should not be.
  cost        wall-clock per step.

Every number is written to both a markdown report for a human and a JSON file for
CI.  Nothing is rounded away in the JSON.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from remoroo_lc.constants import DTYPE
from remoroo_lc.plant import Plant
from remoroo_lc.reference.controller import Controller
from remoroo_lc.schema import CellSpec
from remoroo_lc.spatial import log_so3
from remoroo_lc.tapes import Tape


@dataclass
class TapeResult:
    """Everything measured on one tape."""

    name: str
    category: str
    ticks: int
    duration_s: float
    rms_pos_mm: float
    max_pos_mm: float
    rms_rot_deg: float
    max_rot_deg: float
    tracked_ticks: int
    min_pair_distance_m: float
    hard_collisions: int
    joint_limit_violations: int
    max_joint_excursion_rad: float
    jerk_p50: float
    jerk_p99: float
    chatter_per_s: float
    wall_active_fraction: float
    max_violation: float
    max_slack_norm: float
    box_clamped_ticks: int
    step_ms_p50: float
    step_ms_p99: float
    peak_tcp_speed_mps: float


@dataclass
class CellReport:
    cell: str
    n_joints: int
    n_tcps: int
    action_dim: int
    effector_widths: list
    n_constraint_rows: int
    n_collision_pairs: int
    backend: str
    results: list = field(default_factory=list)
    bars: dict = field(default_factory=dict)

    def passed(self) -> bool:
        return all(v["pass"] for v in self.bars.values())


#: Suggested pass bars.  Tuned deviations are recorded in SUMMARY.md rather than
#: silently applied here.
DEFAULT_BARS = {
    "clean_rms_pos_mm": 3.0,
    "clean_rms_rot_deg": 1.5,
    "hard_collisions": 0,
    "joint_limit_violations": 0,
    "clean_wall_active_fraction": 0.05,
}


def _percentile(x: np.ndarray, p: float) -> float:
    return float(np.percentile(x, p)) if x.size else 0.0


def run_tape(
    cell: CellSpec,
    tape: Tape,
    controller: Controller | None = None,
    plant: Plant | None = None,
    settle_ticks: int = 25,
) -> TapeResult:
    """Drive one tape through controller + plant and measure everything."""
    tape.check_against(cell)
    controller = controller or Controller(cell)
    plant = plant or Plant(cell)

    ticks_per_action = int(
        round(
            float(cell.limits["rates"]["command_hz"]) / float(cell.limits["rates"]["policy_hz"])
        )
    )
    chunk_len = 8
    dt_c = 1.0 / float(cell.limits["rates"]["command_hz"])

    q = cell.rest_posture()
    plant.reset(q)
    controller.reset(q)

    ep, er, active, jerk, viol, slack, step_ms = [], [], [], [], [], [], []
    min_pair = np.inf
    hard_collisions = 0
    limit_violations = 0
    max_excursion = 0.0
    box_ticks = 0
    sign_flips = 0
    peak_speed = 0.0
    prev_qd = None
    prev_acc_sign = None
    prev_p = None
    lo, hi = cell.joint_limits()

    n_actions = tape.actions.shape[0]
    tick = 0
    for start in range(0, n_actions, chunk_len):
        chunk = tape.actions[start : start + chunk_len]
        controller.set_chunk(chunk, q)
        for _ in range(chunk.shape[0] * ticks_per_action):
            t0 = time.perf_counter()
            out = controller.step(q)
            step_ms.append((time.perf_counter() - t0) * 1e3)
            q, _ = plant.step(out.q_target)
            tick += 1

            d = out.diag
            wall = d["n_active"] > 0
            active.append(wall)
            viol.append(d["max_violation"])
            slack.append(d["slack_norm"])
            box_ticks += int(d["box_clamped"] > 0)
            min_pair = min(min_pair, d["min_pair_distance"])
            if d["min_pair_distance"] < 0.0:
                hard_collisions += 1
            below = np.maximum(lo - q, 0.0)
            above = np.maximum(q - hi, 0.0)
            exc = float(np.max(np.maximum(below, above)))
            if exc > 1e-6:
                limit_violations += 1
            max_excursion = max(max_excursion, exc)

            if tick > settle_ticks:
                e_p, e_r = controller.task_error(out)
                if not wall:
                    ep.append(float(np.max(e_p)))
                    er.append(float(np.max(e_r)))
            if prev_qd is not None:
                acc = (out.qd - prev_qd) / DTYPE(dt_c)
                jerk.append(float(np.max(np.abs(acc))))
                s = np.sign(acc)
                if prev_acc_sign is not None and wall:
                    sign_flips += int(np.count_nonzero((s != prev_acc_sign) & (s != 0)))
                prev_acc_sign = s
            prev_qd = out.qd.copy()
            if prev_p is not None:
                peak_speed = max(
                    peak_speed, float(np.max(np.linalg.norm(out.p_meas - prev_p, axis=1)) / dt_c)
                )
            prev_p = out.p_meas.copy()

    duration = tick * dt_c
    ep_a = np.asarray(ep)
    er_a = np.asarray(er)
    jerk_a = np.asarray(jerk)
    return TapeResult(
        name=tape.name,
        category=tape.category,
        ticks=tick,
        duration_s=duration,
        rms_pos_mm=float(np.sqrt(np.mean(ep_a**2)) * 1e3) if ep_a.size else 0.0,
        max_pos_mm=float(np.max(ep_a) * 1e3) if ep_a.size else 0.0,
        rms_rot_deg=float(np.degrees(np.sqrt(np.mean(er_a**2)))) if er_a.size else 0.0,
        max_rot_deg=float(np.degrees(np.max(er_a))) if er_a.size else 0.0,
        tracked_ticks=int(ep_a.size),
        min_pair_distance_m=float(min_pair),
        hard_collisions=hard_collisions,
        joint_limit_violations=limit_violations,
        max_joint_excursion_rad=max_excursion,
        jerk_p50=_percentile(jerk_a, 50),
        jerk_p99=_percentile(jerk_a, 99),
        chatter_per_s=float(sign_flips / max(duration, 1e-9)),
        wall_active_fraction=float(np.mean(np.asarray(active))) if active else 0.0,
        max_violation=float(np.max(np.asarray(viol))) if viol else 0.0,
        max_slack_norm=float(np.max(np.asarray(slack))) if slack else 0.0,
        box_clamped_ticks=box_ticks,
        step_ms_p50=_percentile(np.asarray(step_ms), 50),
        step_ms_p99=_percentile(np.asarray(step_ms), 99),
        peak_tcp_speed_mps=peak_speed,
    )


def score_cell(
    cell: CellSpec, tapes: list[Tape], bars: dict | None = None, backend: str = "reference"
) -> CellReport:
    controller = Controller(cell)
    plant = Plant(cell)
    results = [run_tape(cell, t, controller, plant) for t in tapes]

    bars = dict(bars or DEFAULT_BARS)
    clean = [r for r in results if r.category == "clean"]
    worst_clean_pos = max((r.rms_pos_mm for r in clean), default=0.0)
    worst_clean_rot = max((r.rms_rot_deg for r in clean), default=0.0)
    worst_clean_wall = max((r.wall_active_fraction for r in clean), default=0.0)
    collisions = sum(r.hard_collisions for r in results)
    limit_hits = sum(r.joint_limit_violations for r in results)

    report = CellReport(
        cell=cell.name,
        n_joints=cell.n_joints,
        n_tcps=cell.n_tcps,
        action_dim=cell.action_dim,
        effector_widths=list(cell.effector_widths),
        n_constraint_rows=controller.n_rows,
        n_collision_pairs=len(controller.walls.pairs),
        backend=backend,
        results=[asdict(r) for r in results],
        bars={
            "clean_rms_pos_mm": {
                "value": worst_clean_pos,
                "bar": bars["clean_rms_pos_mm"],
                "pass": worst_clean_pos <= bars["clean_rms_pos_mm"],
            },
            "clean_rms_rot_deg": {
                "value": worst_clean_rot,
                "bar": bars["clean_rms_rot_deg"],
                "pass": worst_clean_rot <= bars["clean_rms_rot_deg"],
            },
            "hard_collisions": {
                "value": collisions,
                "bar": bars["hard_collisions"],
                "pass": collisions <= bars["hard_collisions"],
            },
            "joint_limit_violations": {
                "value": limit_hits,
                "bar": bars["joint_limit_violations"],
                "pass": limit_hits <= bars["joint_limit_violations"],
            },
            "clean_wall_active_fraction": {
                "value": worst_clean_wall,
                "bar": bars["clean_wall_active_fraction"],
                "pass": worst_clean_wall <= bars["clean_wall_active_fraction"],
            },
        },
    )
    return report


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def write_reports(report: CellReport, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"g0_{report.cell}_report.json"
    md_path = out_dir / f"g0_{report.cell}_report.md"
    json_path.write_text(json.dumps(asdict(report), indent=2) + "\n")
    md_path.write_text(_markdown(report))
    return md_path, json_path


def _markdown(report: CellReport) -> str:
    verdict = "PASS" if report.passed() else "FAIL"
    lines = [
        f"# G0 scoreboard: `{report.cell}` -- **{verdict}**",
        "",
        f"- joints: {report.n_joints}  |  TCPs: {report.n_tcps}  |  "
        f"action dim: {report.action_dim}  |  effector widths: {report.effector_widths}",
        f"- constraint rows: {report.n_constraint_rows} "
        f"({report.n_collision_pairs} collision pairs)",
        f"- backend: {report.backend}",
        "",
        "## Bars",
        "",
        "| metric | value | bar | |",
        "| --- | ---: | ---: | :-: |",
    ]
    for name, b in report.bars.items():
        mark = "ok" if b["pass"] else "**FAIL**"
        val = b["value"]
        val_s = f"{val:.4g}" if isinstance(val, float) else str(val)
        bar_s = f"{b['bar']:.4g}" if isinstance(b["bar"], float) else str(b["bar"])
        lines.append(f"| {name} | {val_s} | {bar_s} | {mark} |")

    lines += [
        "",
        "## Tracking (clean tapes, wall-active ticks excluded)",
        "",
        "| tape | peak speed | RMS pos | max pos | RMS rot | max rot | ticks |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in report.results:
        if r["category"] != "clean":
            continue
        lines.append(
            f"| {r['name']} | {r['peak_tcp_speed_mps']:.3f} m/s | "
            f"{r['rms_pos_mm']:.2f} mm | {r['max_pos_mm']:.2f} mm | "
            f"{r['rms_rot_deg']:.3f} deg | {r['max_rot_deg']:.3f} deg | {r['tracked_ticks']} |"
        )

    lines += [
        "",
        "## Safety and filter behaviour (all tapes)",
        "",
        "| tape | category | min pair | hard coll | limit viol | wall active | max viol | box ticks |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in report.results:
        lines.append(
            f"| {r['name']} | {r['category']} | {r['min_pair_distance_m'] * 1e3:.1f} mm | "
            f"{r['hard_collisions']} | {r['joint_limit_violations']} | "
            f"{r['wall_active_fraction'] * 100:.1f}% | {r['max_violation']:.3g} | "
            f"{r['box_clamped_ticks']} |"
        )

    lines += [
        "",
        "## Smoothness and cost",
        "",
        "| tape | jerk p50 | jerk p99 | chatter | step p50 | step p99 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in report.results:
        lines.append(
            f"| {r['name']} | {r['jerk_p50']:.1f} | {r['jerk_p99']:.1f} | "
            f"{r['chatter_per_s']:.1f} /s | {r['step_ms_p50']:.3f} ms | "
            f"{r['step_ms_p99']:.3f} ms |"
        )
    lines += [
        "",
        "Jerk is rad/s^3 at the joint, from the commanded q_dot.  Chatter counts "
        "joint-acceleration sign changes per second while a wall is binding.  "
        "Step times are the pure-NumPy reference, which is a specification and not "
        "the runtime; see bench.py for the kernels.",
        "",
    ]
    return "\n".join(lines)

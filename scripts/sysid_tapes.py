#!/usr/bin/env python3
"""System identification: measure a cell's actuator gains and command delay.

Generates per-joint step and chirp excitations, streams them through a
CellAdapter, records the response, and fits the plant model

    M q_ddot = kp (q_target - q) - kd q_dot        with an N-tick command delay

writing a gains yaml of the same shape as configs/gains.default.yaml.

These are JOINT-space tapes, not action tapes: they drive the servos directly and
never go through the controller, because what is being measured is the thing the
controller sits on top of.

    python scripts/sysid_tapes.py configs/cells/dual_xarm6.yaml            # self-check
    python scripts/sysid_tapes.py configs/cells/dual_xarm6.yaml --out g.yaml

Self-check is the default and is what the test runs: excite the reference plant
with known gains and confirm the fit recovers them.  A fitter that has never been
shown to recover a known answer is not evidence about an unknown one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from remoroo_lc.constants import DTYPE  # noqa: E402
from remoroo_lc.plant import Plant  # noqa: E402
from remoroo_lc.reference.kinematics import KIND_PRISMATIC, KIND_REVOLUTE, KinematicTree  # noqa: E402
from remoroo_lc.schema import load_cell  # noqa: E402

MAX_DELAY_TICKS = 8


def step_tape(cell, joint: int, amplitude: float = 0.15, hold_s: float = 1.2) -> np.ndarray:
    """A single joint steps by `amplitude` and holds; everything else holds still."""
    hz = float(cell.limits["rates"]["command_hz"])
    n = int(round(hold_s * hz))
    q0 = cell.rest_posture()
    tape = np.repeat(q0[None], 2 * n, axis=0).astype(DTYPE)
    lo, hi = cell.joint_limits()
    amp = float(np.clip(amplitude, 0.0, 0.4 * (hi[joint] - lo[joint])))
    tape[n:, joint] = np.clip(q0[joint] + amp, lo[joint] + 1e-3, hi[joint] - 1e-3)
    return tape


def chirp_tape(
    cell, joint: int, amplitude: float = 0.08, f0: float = 0.2, f1: float = 25.0,
    duration_s: float = 6.0,
) -> np.ndarray:
    """A linear-frequency chirp on one joint.

    The chirp is what pins kd down: a step mostly reports kp and the delay,
    because the damping only shows up in how the approach is shaped, whereas a
    sweep through the resonance reports both.
    """
    hz = float(cell.limits["rates"]["command_hz"])
    n = int(round(duration_s * hz))
    t = np.arange(n) / hz
    q0 = cell.rest_posture()
    lo, hi = cell.joint_limits()
    amp = float(np.clip(amplitude, 0.0, 0.2 * (hi[joint] - lo[joint])))
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) / duration_s * t**2)
    tape = np.repeat(q0[None], n, axis=0).astype(DTYPE)
    tape[:, joint] = np.clip(
        q0[joint] + amp * np.sin(phase), lo[joint] + 1e-3, hi[joint] - 1e-3
    )
    return tape


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #


def fit_joint(
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    dt: float,
    inertia: float,
    substeps: int = 1,
) -> tuple[float, float, int, float]:
    """Fit (kp, kd, delay) for one joint from a recorded response.

    Fitted against the EXACT discrete update rather than a continuous
    approximation of it:

        qd[k+1] = qd[k] + (dt/M) (kp (applied[k] - q[k]) - kd qd[k])

    A centred derivative of qd instead sits half a sample away from the
    acceleration that semi-implicit Euler actually applied, and that half sample
    biases kp low by a factor of four even when the feedback rate is ample -- the
    fit looks converged and is simply wrong.  Against real hardware the discrete
    model is an approximation either way; against this plant it is exact, which
    is what makes the self-check meaningful.

    Excitations are fitted as SEPARATE segments and their regression rows
    stacked.  Concatenating the recordings first would put one row across the
    join between two runs -- where the state was reset and nothing continuous
    happened -- and that single row is enough to move the answer by a factor of
    four, because it is a huge apparent acceleration with no cause.

    The model is linear in (kp, kd) once the delay is fixed, so the delay is
    searched over its small integer range and the gains come from a least-squares
    solve at each candidate.  Searching rather than optimising keeps it exact:
    the delay is an integer number of samples and there is nothing to descend.
    """
    best = None
    for d in range(MAX_DELAY_TICKS * substeps + 1):
        rows_A, rows_b = [], []
        for q_target, q, qd in segments:
            # Re-based by one sample so that d = 0 means "no delay".  The state
            # recorded at sample k is the state AFTER that sample's update, so the
            # update from k to k+1 used the target in force at k+1; without the
            # rebase a genuinely undelayed controller has no representable shift
            # and the fit runs away to a negative stiffness.
            base = np.concatenate([q_target[1:], q_target[-1:]])
            applied = np.concatenate([np.full(d, base[0]), base[:-d]]) if d else base
            rows_A.append(np.stack([applied[:-1] - q[:-1], -qd[:-1]], axis=1))
            rows_b.append(inertia * (qd[1:] - qd[:-1]) / dt)
        A = np.concatenate(rows_A)
        b = np.concatenate(rows_b)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        resid = float(np.mean((A @ sol - b) ** 2))
        if best is None or resid < best[3]:
            best = (float(sol[0]), float(sol[1]), d, resid)
    kp, kd, d, resid = best
    # The delay is searched in FEEDBACK samples and reported in command ticks,
    # which is the unit the plant and the vendor both speak.
    return kp, kd, max(0, int(round(d / substeps))), resid


def identify(cell, adapter, inertia: np.ndarray, verbose: bool = True) -> dict:
    """Run every excitation and fit every joint.

    State is read at the controller's feedback rate rather than the command rate
    where the adapter offers one.  A 48 Hz servo sampled at the 250 Hz command
    rate spans its own rise in about three samples, and a numerical derivative of
    that recovers kp an order of magnitude low -- which is the sort of quiet
    wrongness that would then propagate into every scoreboard number without
    ever announcing itself.
    """
    sub = int(getattr(adapter, "substeps_per_tick", 1))
    dt = 1.0 / (float(cell.limits["rates"]["command_hz"]) * sub)
    if verbose and sub == 1:
        print(
            "  NOTE: no high-rate feedback available; fitting at the command rate, "
            "which is only trustworthy for servos well below command_hz / 10"
        )
    labels = cell.joint_labels()
    out: dict[str, dict] = {}
    for j in range(cell.n_joints):
        segments = []
        for tape in (step_tape(cell, j), chirp_tape(cell, j)):
            rec_t, rec_q, rec_qd = [], [], []
            adapter.reset(cell.rest_posture())
            for row in tape:
                adapter.stream_targets(row)
                q, qd = adapter.tick(record=sub > 1) if sub > 1 else adapter.tick()
                if sub > 1:
                    sq, sqd = adapter.last_substeps
                    rec_t.extend([row[j]] * sub)
                    rec_q.extend(sq[:, j].tolist())
                    rec_qd.extend(sqd[:, j].tolist())
                else:
                    rec_t.append(row[j])
                    rec_q.append(q[j])
                    rec_qd.append(qd[j])
            segments.append(
                (
                    np.asarray(rec_t, float),
                    np.asarray(rec_q, float),
                    np.asarray(rec_qd, float),
                )
            )
        kp, kd, delay, resid = fit_joint(segments, dt, float(inertia[j]), substeps=sub)
        out[labels[j]] = {
            "inertia": float(inertia[j]),
            "kp": round(kp, 4),
            "kd": round(kd, 4),
            "delay_ticks": int(delay),
            "residual": resid,
        }
        if verbose:
            print(
                f"  {labels[j]:<24} kp={kp:10.2f}  kd={kd:8.2f}  "
                f"delay={delay} ticks  resid={resid:.3e}"
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--hardware", action="store_true",
        help="drive real hardware instead of the reference plant (needs adapters)",
    )
    a = ap.parse_args()

    cell = load_cell(a.cell)
    if a.hardware:
        raise SystemExit(
            "wire your CellAdapter here; the shape is identical to the mock path "
            "below, and doing it blind from a script flag is how a robot gets a "
            "step command it was not ready for"
        )

    from remoroo_lc.adapters.mock import MockCellAdapter

    adapter = MockCellAdapter(cell)
    adapter.connect()
    tree = KinematicTree(cell)
    by_kind = cell.gains.get("by_kind", {})
    inertia = np.asarray(
        [
            by_kind[{KIND_REVOLUTE: "revolute", KIND_PRISMATIC: "prismatic"}[int(k)]]["inertia"]
            for k in tree.joint_kind
        ],
        dtype=float,
    )

    print(f"{cell.name}: identifying {cell.n_joints} joints")
    fits = identify(cell, adapter, inertia)
    adapter.disconnect()

    delays = {v["delay_ticks"] for v in fits.values()}
    doc = {
        "plant_rate_hz": cell.gains.get("plant_rate_hz", 1000),
        "command_delay_ticks": int(max(delays)),
        "by_kind": by_kind,
        "per_joint": {
            k: {"inertia": v["inertia"], "kp": v["kp"], "kd": v["kd"]} for k, v in fits.items()
        },
    }
    if len(delays) > 1:
        print(
            f"  NOTE: joints disagree about the delay {sorted(delays)}; the plant "
            "model carries one delay for the whole cell, so the largest is used"
        )
    if a.out:
        a.out.write_text(yaml.safe_dump(doc, sort_keys=False))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

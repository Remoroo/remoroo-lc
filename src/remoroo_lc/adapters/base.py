"""The adapter contract.

A `RobotAdapter` speaks to ONE hardware unit -- one controller box, one set of
joints, one effector.  A `CellAdapter` composes however many of those a cell has
onto a shared clock and presents the cell's single global joint vector, in the
order `cell.joint_labels()` fixes.

Everything vendor-specific lives behind this: SDK calls, unit conversions
(degrees vs radians, mm vs m), effector command mappings, whatever rate the
controller insists on, and whatever the vendor calls an emergency stop.  The
controller above it sees joint positions in radians, in config order, and
nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from remoroo_lc.constants import DTYPE
from remoroo_lc.schema import CellSpec


@dataclass(frozen=True)
class UnitLimits:
    """What one hardware unit says about itself, in this package's units."""

    n_joints: int
    q_lo: np.ndarray  # radians (or metres, for a prismatic joint)
    q_hi: np.ndarray
    qd_max: np.ndarray
    effector_width: int
    command_hz: float


@runtime_checkable
class RobotAdapter(Protocol):
    """One hardware unit."""

    name: str

    def connect(self) -> None:
        """Open the connection and leave the unit ready to stream.  Idempotent."""

    def disconnect(self) -> None:
        """Close cleanly.  Must be safe to call twice, and after an estop."""

    def read_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Current (q, q_dot) in radians and radians/second, joint order as configured."""

    def stream_targets(self, q_target: np.ndarray) -> None:
        """Send one command-rate joint position target.  Non-blocking."""

    def set_effector(self, u: np.ndarray) -> None:
        """Set this unit's effector from `effector_width` normalised floats in [0, 1]."""

    def limits(self) -> UnitLimits:
        """What the unit reports about itself, for cross-checking against the cell."""

    def estop(self) -> None:
        """Stop as hard as this unit can.  Must be safe to call from any thread,
        at any time, including before connect() and after disconnect()."""


class CellAdapter:
    """Composes per-unit adapters into one cell-shaped interface.

    The unit-to-joint mapping comes from the cell: each unit is named after a
    model, and that model's joints occupy a contiguous slice of the global vector.
    A cell whose model spans two controllers, or whose controller drives two
    models, needs a different mapping and should say so in config rather than
    being special-cased here.
    """

    def __init__(self, cell: CellSpec, adapters: dict[str, RobotAdapter]) -> None:
        self.cell = cell
        missing = {m.name for m in cell.models} - set(adapters)
        extra = set(adapters) - {m.name for m in cell.models}
        if missing or extra:
            raise ValueError(
                f"adapter set does not match the cell: missing {sorted(missing)}, "
                f"unexpected {sorted(extra)}"
            )
        self.adapters = adapters
        self.slices: dict[str, slice] = {}
        k = 0
        for m in cell.models:
            self.slices[m.name] = slice(k, k + len(m.joint_names))
            k += len(m.joint_names)

        # Effector channels are indexed, not sliced.  One model can carry several
        # TCPs -- a torso with two limbs is one model with two TCPs -- and their
        # channels need not be adjacent in the action vector, so a per-model
        # slice would silently drop all but the last.
        idx: dict[str, list[int]] = {}
        e = 0
        for t in cell.tcps:
            g = t.effector.width
            if g:
                idx.setdefault(t.model, []).extend(range(e, e + g))
            e += g
        self.eff_index: dict[str, np.ndarray] = {
            name: np.asarray(v, dtype=int) for name, v in idx.items()
        }
        self.n_joints = k
        self.eff_dim = e

    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        for a in self.adapters.values():
            a.connect()
        self.check_limits()

    def disconnect(self) -> None:
        for a in self.adapters.values():
            a.disconnect()

    def check_limits(self) -> None:
        """Fail loudly if the hardware disagrees with the cell file.

        A cell file that claims a joint range the machine does not have is how a
        controller ends up confidently commanding something the servo will refuse,
        and the failure shows up as an unexplained fault mid-episode rather than
        at startup.
        """
        lo, hi = self.cell.joint_limits()
        qd_max = self.cell.joint_velocity_limits()
        for name, a in self.adapters.items():
            sl = self.slices[name]
            lim = a.limits()
            if lim.n_joints != sl.stop - sl.start:
                raise ValueError(
                    f"{name}: hardware reports {lim.n_joints} joints, cell says "
                    f"{sl.stop - sl.start}"
                )
            if np.any(lo[sl] < lim.q_lo - 1e-6) or np.any(hi[sl] > lim.q_hi + 1e-6):
                raise ValueError(
                    f"{name}: the cell's joint limits are wider than the hardware's"
                )
            if np.any(qd_max[sl] > lim.qd_max + 1e-6):
                raise ValueError(
                    f"{name}: the cell's velocity limits exceed the hardware's"
                )

    def read_state(self) -> tuple[np.ndarray, np.ndarray]:
        q = np.zeros(self.n_joints, dtype=DTYPE)
        qd = np.zeros(self.n_joints, dtype=DTYPE)
        for name, a in self.adapters.items():
            sl = self.slices[name]
            qa, qda = a.read_state()
            q[sl] = qa
            qd[sl] = qda
        return q, qd

    def stream_targets(self, q_target: np.ndarray) -> None:
        q_target = np.asarray(q_target, dtype=DTYPE).reshape(-1)
        for name, a in self.adapters.items():
            a.stream_targets(q_target[self.slices[name]])

    def set_effector(self, u: np.ndarray) -> None:
        u = np.asarray(u, dtype=DTYPE).reshape(-1)
        for name, ix in self.eff_index.items():
            self.adapters[name].set_effector(u[ix])

    def estop(self) -> None:
        """Stop every unit.  One unit failing to stop must not stop the others
        from being told to."""
        errors = []
        for name, a in self.adapters.items():
            try:
                a.estop()
            except Exception as exc:  # noqa: BLE001 -- every unit must be tried
                errors.append((name, exc))
        if errors:
            raise RuntimeError(f"estop failed on {[n for n, _ in errors]}: {errors}")

    def __enter__(self) -> CellAdapter:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.estop()
        self.disconnect()

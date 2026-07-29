"""Plant-backed adapter.

Wraps `remoroo_lc.plant.Plant` behind the same protocol the real hardware uses,
so integration tests exercise the production code path -- CellAdapter, unit
splitting, limit cross-checks, effector mapping -- instead of a shortcut around
it.  A bug in the composition layer that only shows up on hardware is a bug that
was never tested.
"""

from __future__ import annotations

import numpy as np

from remoroo_lc.adapters.base import CellAdapter, RobotAdapter, UnitLimits
from remoroo_lc.constants import DTYPE
from remoroo_lc.plant import Plant
from remoroo_lc.schema import CellSpec


class MockUnit(RobotAdapter):
    """One model's worth of joints, driven by a shared plant."""

    def __init__(self, name: str, plant: Plant, sl: slice, effector_width: int,
                 command_hz: float, q_lo, q_hi, qd_max) -> None:
        self.name = name
        self._plant = plant
        self._slice = sl
        self._effector_width = effector_width
        self._command_hz = command_hz
        self._q_lo = np.asarray(q_lo, dtype=DTYPE)
        self._q_hi = np.asarray(q_hi, dtype=DTYPE)
        self._qd_max = np.asarray(qd_max, dtype=DTYPE)
        self.effector = np.zeros(effector_width, dtype=DTYPE)
        self.connected = False
        self.estopped = False
        self.pending: np.ndarray | None = None

    def connect(self) -> None:
        self.connected = True
        self.estopped = False

    def disconnect(self) -> None:
        self.connected = False

    def read_state(self):
        q, qd = self._plant.state
        return q[self._slice].copy(), qd[self._slice].copy()

    def stream_targets(self, q_target: np.ndarray) -> None:
        if not self.connected:
            raise RuntimeError(f"{self.name}: stream_targets before connect")
        if self.estopped:
            return
        self.pending = np.asarray(q_target, dtype=DTYPE).copy()

    def set_effector(self, u: np.ndarray) -> None:
        u = np.asarray(u, dtype=DTYPE).reshape(-1)
        if u.shape[0] != self._effector_width:
            raise ValueError(
                f"{self.name}: effector command has {u.shape[0]} channels, "
                f"expected {self._effector_width}"
            )
        self.effector = np.clip(u, DTYPE(0.0), DTYPE(1.0))

    def limits(self) -> UnitLimits:
        return UnitLimits(
            n_joints=self._slice.stop - self._slice.start,
            q_lo=self._q_lo,
            q_hi=self._q_hi,
            qd_max=self._qd_max,
            effector_width=self._effector_width,
            command_hz=self._command_hz,
        )

    def estop(self) -> None:
        self.estopped = True
        self.pending = None


class MockCellAdapter(CellAdapter):
    """A whole cell backed by one plant, on a shared clock.

    `stream_targets` only queues; `tick()` is what advances the plant, so the
    units stay synchronised the way a real cell's shared clock keeps them.
    """

    def __init__(self, cell: CellSpec, plant: Plant | None = None) -> None:
        plant = plant or Plant(cell)
        lo, hi = cell.joint_limits()
        qd_max = cell.joint_velocity_limits()
        command_hz = float(cell.limits["rates"]["command_hz"])

        # Effector width per MODEL, summed over that model's TCPs.
        width: dict[str, int] = {m.name: 0 for m in cell.models}
        for t in cell.tcps:
            width[t.model] += t.effector.width

        units: dict[str, RobotAdapter] = {}
        k = 0
        for m in cell.models:
            sl = slice(k, k + len(m.joint_names))
            k += len(m.joint_names)
            units[m.name] = MockUnit(
                name=m.name,
                plant=plant,
                sl=sl,
                effector_width=width[m.name],
                command_hz=command_hz,
                # A real unit reports slightly wider limits than the cell claims,
                # which is what check_limits is there to verify.
                q_lo=lo[sl] - DTYPE(0.01),
                q_hi=hi[sl] + DTYPE(0.01),
                qd_max=qd_max[sl] + DTYPE(0.01),
            )
        super().__init__(cell, units)
        self.plant = plant

    def reset(self, q: np.ndarray) -> None:
        self.plant.reset(np.asarray(q, dtype=DTYPE))

    def tick(self, record: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Advance the shared clock by one command tick.

        `record` exposes the plant's intra-tick states through `last_substeps`,
        standing in for the high-rate feedback stream a real controller provides
        and which system identification needs.
        """
        target = np.zeros(self.n_joints, dtype=DTYPE)
        q_now, _ = self.plant.state
        for name, a in self.adapters.items():
            sl = self.slices[name]
            pending = getattr(a, "pending", None)
            # An estopped or silent unit holds position rather than inheriting
            # whatever was last in the buffer.
            target[sl] = q_now[sl] if pending is None else pending
        return self.plant.step(target, record=record)

    @property
    def last_substeps(self):
        return self.plant.last_substeps

    @property
    def substeps_per_tick(self) -> int:
        return self.plant.substeps

    @property
    def effector_state(self) -> np.ndarray:
        out = np.zeros(self.eff_dim, dtype=DTYPE)
        for name, ix in self.eff_index.items():
            out[ix] = self.adapters[name].effector
        return out

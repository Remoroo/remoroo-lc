"""Per-joint second-order plant for headless testing.

    M_j qdd_j = kp_j (q_target_j - q_j) - kd_j qd_j

integrated with semi-implicit Euler at plant_rate_hz, behind a pure command delay
measured in command ticks.  It stands in for two different things -- the Isaac Lab
actuator model in simulation and the vendor's servo loop on hardware -- which is
exactly why it is deliberately crude: it has no dynamics the real ones do not
have, so a controller that behaves well against it has not been tuned against
anything the real system lacks.

Everything is sized from cell config; gains come from a gains yaml, whose real
values come from a sysid run.  There is no default that assumes a joint count.
"""

from __future__ import annotations

import numpy as np

from remoroo_lc.constants import DTYPE
from remoroo_lc.reference.kinematics import KIND_PRISMATIC, KIND_REVOLUTE, KinematicTree
from remoroo_lc.schema import CellSpec

_KIND_KEY = {KIND_REVOLUTE: "revolute", KIND_PRISMATIC: "prismatic"}


class Plant:
    """A joint-space plant that a controller can be closed around."""

    def __init__(self, cell: CellSpec, tree: KinematicTree | None = None) -> None:
        self.cell = cell
        tree = tree or KinematicTree(cell)
        gains = cell.gains or {}
        by_kind = gains.get("by_kind") or {}
        per_joint = gains.get("per_joint") or {}

        self.n = cell.n_joints
        self.plant_hz = float(gains.get("plant_rate_hz", 1000.0))
        self.command_hz = float(cell.limits["rates"]["command_hz"])
        substeps = self.plant_hz / self.command_hz
        if abs(substeps - round(substeps)) > 1.0e-9:
            raise ValueError(
                f"plant_rate_hz ({self.plant_hz}) must be an integer multiple of "
                f"command_hz ({self.command_hz}); a fractional ratio would make the "
                "number of substeps depend on accumulated float error"
            )
        self.substeps = int(round(substeps))
        self.dt = DTYPE(1.0 / self.plant_hz)
        self.delay_ticks = int(gains.get("command_delay_ticks", 0))

        labels = cell.joint_labels()
        self.inertia = np.zeros(self.n, dtype=DTYPE)
        self.kp = np.zeros(self.n, dtype=DTYPE)
        self.kd = np.zeros(self.n, dtype=DTYPE)
        for j in range(self.n):
            key = _KIND_KEY.get(int(tree.joint_kind[j]))
            base = dict(by_kind.get(key, {})) if key else {}
            base.update(per_joint.get(labels[j], {}))
            missing = {"inertia", "kp", "kd"} - set(base)
            if missing:
                raise ValueError(
                    f"gains file has no {sorted(missing)} for joint {labels[j]} "
                    f"(kind {key}); add a by_kind.{key} block or a per_joint entry"
                )
            self.inertia[j] = DTYPE(base["inertia"])
            self.kp[j] = DTYPE(base["kp"])
            self.kd[j] = DTYPE(base["kd"])

        self.q = np.zeros(self.n, dtype=DTYPE)
        self.qd = np.zeros(self.n, dtype=DTYPE)
        self._queue: list[np.ndarray] = []
        self.last_substeps: tuple[np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------ #
    def reset(self, q: np.ndarray, qd: np.ndarray | None = None) -> None:
        self.q = np.asarray(q, dtype=DTYPE).reshape(-1).copy()
        self.qd = (
            np.zeros(self.n, dtype=DTYPE)
            if qd is None
            else np.asarray(qd, dtype=DTYPE).reshape(-1).copy()
        )
        self._queue = [self.q.copy() for _ in range(self.delay_ticks)]

    def step(
        self, q_target: np.ndarray, record: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance one COMMAND tick (substeps plant ticks).  Returns (q, qd).

        With `record`, also stashes the state after every SUBSTEP in
        `last_substeps`.  System identification needs that: a joint servo at
        48 Hz sampled at the 250 Hz command rate is barely three samples across
        its own rise, and differentiating that gives a fit off by an order of
        magnitude.  Real controllers publish a high-rate feedback stream for
        exactly this reason; this is its stand-in.
        """
        target = np.asarray(q_target, dtype=DTYPE).reshape(-1).copy()
        if self.delay_ticks:
            self._queue.append(target)
            applied = self._queue.pop(0)
        else:
            applied = target
        trace_q = [] if record else None
        trace_qd = [] if record else None
        for _ in range(self.substeps):
            acc = (self.kp * (applied - self.q) - self.kd * self.qd) / self.inertia
            self.qd = (self.qd + acc * self.dt).astype(DTYPE)
            self.q = (self.q + self.qd * self.dt).astype(DTYPE)
            if record:
                trace_q.append(self.q.copy())
                trace_qd.append(self.qd.copy())
        if record:
            self.last_substeps = (np.stack(trace_q), np.stack(trace_qd))
        return self.q.copy(), self.qd.copy()

    # ------------------------------------------------------------------ #
    @property
    def state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.q.copy(), self.qd.copy()

    def natural_frequency(self) -> np.ndarray:
        """sqrt(kp/M) per joint, rad/s -- useful for sanity-checking a sysid fit."""
        return np.sqrt(self.kp / self.inertia).astype(DTYPE)

    def damping_ratio(self) -> np.ndarray:
        """kd / (2 sqrt(kp M)) per joint."""
        return (self.kd / (DTYPE(2.0) * np.sqrt(self.kp * self.inertia))).astype(DTYPE)

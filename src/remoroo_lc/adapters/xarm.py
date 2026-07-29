"""UFACTORY xArm adapter.

The first implementation of RobotAdapter, and the template for every other
brand.  Everything vendor-shaped is here and nowhere else:

  * the SDK import, which is optional and deferred so this file can be read,
    type-checked and imported without the SDK present
  * degrees vs radians (the SDK's servo API is radians; its status API is not)
  * the servo-mode handshake, and the fact that servo_j must be fed at a steady
    rate or the controller faults
  * the gripper's 0..850 integer range mapping onto one normalised float
  * what "stop" means on this controller

Nothing above this file knows any of it.  See README for the recipe.

HARDWARE ONLY.  Nothing here is exercised in CI; the smoke test is marked `hw`.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from remoroo_lc.adapters.base import RobotAdapter, UnitLimits
from remoroo_lc.constants import DTYPE

#: The SDK's gripper position range, in its own integer units.
_GRIPPER_OPEN = 850
_GRIPPER_CLOSED = 0

#: servo_j wants targets at a steady cadence; falling behind faults the
#: controller rather than merely lagging.
_SERVO_MODE = 1
_SERVO_STATE = 0


class XArmUnit(RobotAdapter):
    """One xArm controller box, plus whatever effector is on it."""

    def __init__(
        self,
        name: str,
        host: str,
        n_joints: int,
        command_hz: float,
        effector_width: int = 1,
        q_lo: np.ndarray | None = None,
        q_hi: np.ndarray | None = None,
        qd_max: np.ndarray | None = None,
        has_effector: bool = True,
    ) -> None:
        self.name = name
        self.host = host
        self.n_joints = int(n_joints)
        self.command_hz = float(command_hz)
        self.effector_width = int(effector_width)
        self.has_effector = has_effector
        self._arm = None
        self._lock = threading.Lock()
        self._estopped = False
        self._last_stream = 0.0
        self._q_lo = q_lo
        self._q_hi = q_hi
        self._qd_max = qd_max

    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        try:
            from xarm.wrapper import XArmAPI  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - hardware only
            raise RuntimeError(
                "the UFACTORY SDK is not installed; `pip install xarm-python-sdk`. "
                "It is deliberately not a dependency of remoroo-lc: no vendor SDK is."
            ) from exc
        with self._lock:
            if self._arm is not None:
                return
            arm = XArmAPI(self.host, is_radian=True)
            arm.connect()
            arm.clean_error()
            arm.clean_warn()
            arm.motion_enable(enable=True)
            arm.set_mode(_SERVO_MODE)
            arm.set_state(_SERVO_STATE)
            if self.has_effector:
                arm.set_gripper_enable(True)
                arm.set_gripper_mode(0)
            self._arm = arm
            self._estopped = False

    def disconnect(self) -> None:
        with self._lock:
            if self._arm is None:
                return
            try:
                self._arm.set_state(4)  # stop
                self._arm.motion_enable(enable=False)
                self._arm.disconnect()
            finally:
                self._arm = None

    # ------------------------------------------------------------------ #
    def read_state(self) -> tuple[np.ndarray, np.ndarray]:
        arm = self._require()
        code, (q, qd, _tau) = arm.get_joint_states(is_radian=True)
        if code != 0:
            raise RuntimeError(f"{self.name}: get_joint_states returned {code}")
        return (
            np.asarray(q[: self.n_joints], dtype=DTYPE),
            np.asarray(qd[: self.n_joints], dtype=DTYPE),
        )

    def stream_targets(self, q_target: np.ndarray) -> None:
        if self._estopped:
            return
        arm = self._require()
        q = np.asarray(q_target, dtype=np.float64).reshape(-1)
        if q.shape[0] != self.n_joints:
            raise ValueError(
                f"{self.name}: got {q.shape[0]} targets, unit has {self.n_joints} joints"
            )
        code = arm.set_servo_angle_j(angles=q.tolist(), is_radian=True)
        if code != 0:
            self.estop()
            raise RuntimeError(f"{self.name}: set_servo_angle_j returned {code}; stopped")
        self._last_stream = time.perf_counter()

    def set_effector(self, u: np.ndarray) -> None:
        if not self.has_effector or self._estopped:
            return
        arm = self._require()
        frac = float(np.clip(np.asarray(u, dtype=np.float64).reshape(-1)[0], 0.0, 1.0))
        # 0 = open, 1 = closed in this package; the SDK is the other way round.
        pos = int(round(_GRIPPER_OPEN + frac * (_GRIPPER_CLOSED - _GRIPPER_OPEN)))
        arm.set_gripper_position(pos, wait=False)

    def limits(self) -> UnitLimits:
        arm = self._require()
        if self._q_lo is not None and self._q_hi is not None:
            lo, hi = np.asarray(self._q_lo, DTYPE), np.asarray(self._q_hi, DTYPE)
        else:
            # The SDK's reported range is per joint, in radians when is_radian.
            lim = np.asarray(arm.joint_limits or [], dtype=DTYPE)
            if lim.shape != (self.n_joints, 2):
                raise RuntimeError(
                    f"{self.name}: the SDK reported joint limits of shape {lim.shape}; "
                    "pass q_lo/q_hi explicitly from the cell's URDF instead"
                )
            lo, hi = lim[:, 0], lim[:, 1]
        qd_max = (
            np.asarray(self._qd_max, DTYPE)
            if self._qd_max is not None
            else np.full(self.n_joints, DTYPE(3.14))
        )
        return UnitLimits(
            n_joints=self.n_joints,
            q_lo=lo,
            q_hi=hi,
            qd_max=qd_max,
            effector_width=self.effector_width if self.has_effector else 0,
            command_hz=self.command_hz,
        )

    def estop(self) -> None:
        self._estopped = True
        arm = self._arm
        if arm is None:
            return
        try:
            arm.set_state(4)
            arm.emergency_stop()
        except Exception:  # noqa: BLE001 -- estop must never raise on the way out
            pass

    # ------------------------------------------------------------------ #
    def _require(self):
        arm = self._arm
        if arm is None:
            raise RuntimeError(f"{self.name}: not connected")
        return arm

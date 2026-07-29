"""Isaac Lab adapter: the batched training seam.

Isaac Lab is optional and is NOT imported here.  This module takes whatever
object exposes the two things the controller needs -- a way to read joint state
for every environment and a way to write joint position targets for every
environment -- and adapts the batched kernel controller onto it.  That keeps the
seam testable without a simulator: `tests/test_adapters.py` drives it with a fake
environment that implements the same two methods.

Tensor interop.  Warp arrays and torch tensors share the CUDA array interface, so
`wp.to_torch` / `wp.from_torch` are zero copy.  A batched training loop should
therefore stay on device: call `step_device()` and hand Isaac Lab the Warp array
directly rather than round-tripping through NumPy, which is what `step()` does
and which is only appropriate for tests and for single-instance use.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from remoroo_lc.constants import DTYPE
from remoroo_lc.schema import CellSpec


class BatchedEnv(Protocol):
    """The two methods this adapter needs from a simulation backend."""

    num_envs: int

    def read_joint_state(self) -> tuple[Any, Any]:
        """(q, q_dot), each shaped (num_envs, n_joints), in the cell's joint order."""

    def write_joint_targets(self, q_target: Any) -> None:
        """Apply (num_envs, n_joints) position targets to the actuators."""


class IsaacLabAdapter:
    """Runs the batched controller against a batched environment.

    The joint ORDER is the contract.  Isaac Lab orders joints by its own articulation
    parse; this package orders them by the cell file.  `joint_order` is the
    permutation between them and must be supplied by whoever builds the scene --
    guessing it would produce a controller that works and a robot that moves the
    wrong joints, which is the worst possible failure mode because it looks fine
    in aggregate metrics.
    """

    def __init__(
        self,
        cell: CellSpec,
        env: BatchedEnv,
        device: str = "cuda",
        joint_order: list[int] | None = None,
        delta_mode: str = "cumulative",
    ) -> None:
        from remoroo_lc.kernels.warp_backend import BatchedController

        self.cell = cell
        self.env = env
        self.controller = BatchedController(
            cell, num_envs=int(env.num_envs), device=device, delta_mode=delta_mode
        )
        if joint_order is None:
            self.perm = np.arange(cell.n_joints)
        else:
            self.perm = np.asarray(joint_order, dtype=int)
            if sorted(self.perm.tolist()) != list(range(cell.n_joints)):
                raise ValueError(
                    "joint_order must be a permutation of range(n_joints); it maps "
                    "the environment's joint index to this cell's"
                )
        self.inv_perm = np.argsort(self.perm)

    # ------------------------------------------------------------------ #
    def _to_cell(self, arr) -> np.ndarray:
        a = np.asarray(_as_numpy(arr), dtype=DTYPE).reshape(self.env.num_envs, -1)
        return a[:, self.perm]

    def _to_env(self, arr: np.ndarray) -> np.ndarray:
        return arr[:, self.inv_perm]

    def reset(self) -> None:
        q, _ = self.env.read_joint_state()
        self.controller.reset(self._to_cell(q))

    def set_chunk(self, actions) -> None:
        """actions: (num_envs, K, action_dim) or (K, action_dim) broadcast."""
        q, _ = self.env.read_joint_state()
        self.controller.set_chunk(np.asarray(_as_numpy(actions), dtype=DTYPE),
                                  self._to_cell(q))

    def step(self) -> dict:
        """One command tick, host round-tripped.  Fine for tests, wasteful in training."""
        q, _ = self.env.read_joint_state()
        out = self.controller.step(self._to_cell(q))
        self.env.write_joint_targets(self._to_env(out["q_target"]))
        return out

    def step_device(self):
        """One command tick, leaving the result on device.

        Returns the Warp array of joint targets in CELL order.  Use
        `warp.to_torch(...)` on it and index with `inv_perm` on device; the
        conversion is a view, not a copy.
        """
        q, _ = self.env.read_joint_state()
        self.controller.step(self._to_cell(q))
        return self.controller.q_target


def _as_numpy(x):
    """NumPy view of a NumPy array, a torch tensor, or a Warp array."""
    if isinstance(x, np.ndarray):
        return x
    for attr in ("numpy",):
        fn = getattr(x, attr, None)
        if callable(fn):
            try:
                return fn()
            except TypeError:  # torch CUDA tensors need .cpu() first
                return x.detach().cpu().numpy()
    return np.asarray(x)

"""Remoroo Local Controller (G0).

An embodiment-agnostic reactive controller that sits between a learned policy and
the joint servos of any fixed-base robot cell.  The same code runs batched across
thousands of simulation environments and single-instance at command rate on real
hardware.

The unit of control is a TCP -- a task frame at the end of a kinematic chain in a
fixed-base kinematic tree.  Not an arm.  A cell may contain any number of models,
any number of TCPs, shared joints between chains, and TCPs with no effector at
all (a foot).  Nothing in this package may assume otherwise.

Public entry points:
    load_cell            -- parse and validate a cell.yaml into a CellSpec
    KinematicTree        -- generic FK + task/point Jacobians over the whole cell
    Controller           -- the three-layer reference controller (pure NumPy)
    Plant                -- second-order per-joint plant for headless testing
"""

from remoroo_lc.constants import DTYPE, TASK_DIM
from remoroo_lc.schema import (
    CellSpec,
    EffectorSpec,
    EnvPrimitive,
    ModelSpec,
    TcpSpec,
    load_cell,
    load_limits,
    validate_cell,
)

__all__ = [
    "CellSpec",
    "Controller",
    "DTYPE",
    "EffectorSpec",
    "EnvPrimitive",
    "KinematicTree",
    "ModelSpec",
    "Plant",
    "TASK_DIM",
    "TcpSpec",
    "load_cell",
    "load_limits",
    "validate_cell",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    # Lazy so that `import remoroo_lc` stays cheap for config-only consumers.
    if name == "KinematicTree":
        from remoroo_lc.reference.kinematics import KinematicTree

        return KinematicTree
    if name == "Controller":
        from remoroo_lc.reference.controller import Controller

        return Controller
    if name == "Plant":
        from remoroo_lc.plant import Plant

        return Plant
    raise AttributeError(name)

"""Vendor I/O.

This is the only place in the package that is allowed to know a brand exists.
A static test asserts that nothing in the core imports from here, and the core
contains no vendor name at all -- see tests/test_agnosticism.py.

Adding a robot brand is: one file here, one cell yaml, sphere files, and a sysid
run.  Nothing else changes.  If a brand ever needs a change outside this package,
that is a bug in the abstraction and not a fact about the robot.
"""

from remoroo_lc.adapters.base import CellAdapter, RobotAdapter, UnitLimits

__all__ = ["CellAdapter", "RobotAdapter", "UnitLimits"]

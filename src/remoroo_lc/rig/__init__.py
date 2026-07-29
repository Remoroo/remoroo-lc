"""Rig-side tooling: recordings, replay, and the motion-validation metrics.

Separate from `remoroo_lc.reference` and `remoroo_lc.kernels` because none of this
is in the control loop.  It reads recordings, drives replays, and computes
metrics; it never runs at command rate on a robot.
"""

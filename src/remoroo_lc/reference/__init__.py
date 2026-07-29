"""Pure-NumPy reference implementation.

This package is the canonical, readable specification of what the controller
computes.  The Warp kernels in `remoroo_lc.kernels` mirror it operation for
operation, and the equivalence test between the two is what keeps that true.

Everything here runs in float32 for exactly that reason -- see
`remoroo_lc.constants`.
"""

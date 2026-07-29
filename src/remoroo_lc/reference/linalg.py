"""Dense linear algebra written to be mirrored exactly by the kernels.

NumPy's ``linalg.solve`` would be faster and better conditioned, but it dispatches
to LAPACK, whose blocking and pivoting differ from anything expressible in a
per-thread Warp kernel.  Reference and kernel would then disagree at a level that
grows with the problem, and there would be no way to tell a real divergence from
a library difference.  So the reference does its own unblocked Cholesky in the
same loop order the kernel uses, and the two agree to the last bit that float32
associativity allows.
"""

from __future__ import annotations

import numpy as np

from remoroo_lc.constants import DTYPE


def cholesky_spd(A: np.ndarray, floor: float = 1.0e-12) -> tuple[np.ndarray, bool]:
    """Unblocked lower Cholesky of a symmetric matrix.

    Returns (L, ok).  `ok` is False if a pivot fell to or below `floor`, in which
    case L is only filled up to that column; callers decide what that means.  No
    pivoting, no early return that would change the arithmetic of the rows that
    did succeed.
    """
    m = A.shape[0]
    L = np.zeros((m, m), dtype=DTYPE)
    ok = True
    for r in range(m):
        s = A[r, r] - np.dot(L[r, :r], L[r, :r])
        if s <= floor:
            ok = False
            break
        d = DTYPE(np.sqrt(s))
        L[r, r] = d
        for c in range(r + 1, m):
            L[c, r] = (A[c, r] - np.dot(L[c, :r], L[r, :r])) / d
    return L, ok


def chol_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve L L^T x = b by forward then back substitution."""
    m = L.shape[0]
    y = np.zeros(m, dtype=DTYPE)
    for r in range(m):
        y[r] = (b[r] - np.dot(L[r, :r], y[:r])) / L[r, r]
    x = np.zeros(m, dtype=DTYPE)
    for r in range(m - 1, -1, -1):
        x[r] = (y[r] - np.dot(L[r + 1 :, r], x[r + 1 :])) / L[r, r]
    return x


def spd_solve(A: np.ndarray, b: np.ndarray, ridge: float = 1.0e-9) -> np.ndarray:
    """Solve A x = b for symmetric positive definite A.

    On a failed factorisation the ridge is added once and the factorisation
    retried, a fixed two-attempt sequence rather than an adaptive loop.
    """
    L, ok = cholesky_spd(A)
    if not ok:
        L, ok = cholesky_spd(A + DTYPE(ridge) * np.eye(A.shape[0], dtype=DTYPE))
        if not ok:
            return np.zeros(A.shape[0], dtype=DTYPE)
    return chol_solve(L, b.astype(DTYPE))


def chol_det(L: np.ndarray) -> np.floating:
    """Product of the Cholesky diagonal, i.e. sqrt(det(A))."""
    out = DTYPE(1.0)
    for r in range(L.shape[0]):
        out = out * L[r, r]
    return out

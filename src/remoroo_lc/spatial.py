"""SO(3) / SE(3) helpers.

Deliberately small, allocation-light and written so that every operation has an
exact one-to-one counterpart in the Warp kernels.  Rotations are 3x3 matrices;
orientation *deltas* are rotation vectors (axis * angle).

All functions are float32 in / float32 out.  Series expansions near theta = 0 use
fixed-order polynomials rather than branching on magnitude wherever that is
possible without losing accuracy, because a data-dependent branch that changes
the arithmetic performed is a determinism hazard when the same code runs on two
devices.  Where a branch is unavoidable (log of a rotation near pi) the branch
condition is on the input only, never on iteration state.
"""

from __future__ import annotations

import numpy as np

from remoroo_lc.constants import DTYPE, POINT_DIM

_TAYLOR_CUTOFF = 1.0e-4  # |theta| below which the series form is used


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix of a 3-vector."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=DTYPE,
    )


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Rodrigues: rotation vector -> rotation matrix."""
    w = np.asarray(w, dtype=DTYPE)
    theta = float(np.linalg.norm(w))
    K = skew(w)
    if theta < _TAYLOR_CUTOFF:
        # sin(t)/t and (1-cos t)/t^2 to second order; exact to float32 here.
        a = 1.0 - theta * theta / 6.0
        b = 0.5 - theta * theta / 24.0
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / (theta * theta)
    return (np.eye(POINT_DIM, dtype=DTYPE) + DTYPE(a) * K + DTYPE(b) * (K @ K)).astype(DTYPE)


def log_so3(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector, with |result| in [0, pi].

    The angle comes from atan2(|sin|, cos) rather than arccos(cos).  arccos is
    stationary at theta = pi -- the trace stops responding to the angle there --
    so in float32 it recovers pi-1e-4 with about 4e-4 of error, which is visible
    as orientation jitter on any TCP that passes through a half turn.  atan2 of
    the antisymmetric and symmetric parts stays first-order sensitive at both
    ends of the range and recovers the same angle to ~1e-7.
    """
    R = np.asarray(R, dtype=DTYPE)
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    anti = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=DTYPE
    )
    sin_t = 0.5 * float(np.linalg.norm(anti))  # |sin theta| >= 0
    cos_t = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
    theta = float(np.arctan2(sin_t, cos_t))  # in [0, pi]
    if theta < _TAYLOR_CUTOFF:
        # Near identity: vee(R - R^T) / 2 with the 1 + t^2/6 correction.
        s = 0.5 * (1.0 + theta * theta / 6.0)
        return np.array(
            [
                s * (R[2, 1] - R[1, 2]),
                s * (R[0, 2] - R[2, 0]),
                s * (R[1, 0] - R[0, 1]),
            ],
            dtype=DTYPE,
        )
    if theta > np.pi - 1.0e-3:
        # Near pi the antisymmetric part of R is O(sin theta) and useless for the
        # axis MAGNITUDE, so that comes from the symmetric part, which is
        # well conditioned exactly there.  Its SIGN is still perfectly good
        # though -- vee(R - R^T) = 2 sin(theta) * axis and sin(theta) > 0 on
        # (0, pi) -- so the two parts are combined rather than one being trusted
        # for both.  Taking the sign from the symmetric part alone (say, "the
        # largest component is positive") silently returns the antipodal
        # rotation vector, which is only an equivalent answer at exactly pi.
        A = 0.5 * (R + R.T) - cos_t * np.eye(POINT_DIM, dtype=DTYPE)
        denom = max(1.0 - cos_t, 1.0e-12)
        axis = np.sqrt(np.clip(np.diag(A) / denom, 0.0, None)).astype(DTYPE)
        k = int(np.argmax(axis))
        if axis[k] > 0.0:
            # Relative signs from row k of A = (1 - cos) axis axis^T.
            for j in range(POINT_DIM):
                if j != k:
                    axis[j] = A[k, j] / (denom * axis[k])
        # Global sign.  At exactly pi `anti` is zero and both signs are the same
        # rotation, so the fallback is arbitrary but fixed.
        if anti[k] < 0.0:
            axis = -axis
        n = float(np.linalg.norm(axis))
        axis = axis / DTYPE(max(n, 1.0e-12))
        return (axis * DTYPE(theta)).astype(DTYPE)
    s = 0.5 * theta / np.sin(theta)
    return np.array(
        [
            s * (R[2, 1] - R[1, 2]),
            s * (R[0, 2] - R[2, 0]),
            s * (R[1, 0] - R[0, 1]),
        ],
        dtype=DTYPE,
    )


def rpy_to_mat(rpy) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw (X then Y then Z) -> rotation matrix."""
    r, p, y = (float(v) for v in rpy)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=DTYPE)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=DTYPE)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=DTYPE)
    return (Rz @ Ry @ Rx).astype(DTYPE)


def quat_to_mat(quat) -> np.ndarray:
    """Quaternion (w, x, y, z) -> rotation matrix."""
    w, x, y, z = (float(v) for v in quat)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n <= 0.0:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=DTYPE,
    )


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (w, x, y, z), w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array(
            [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        )
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array(
            [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        )
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array(
            [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
        )
    if q[0] < 0.0:
        q = -q
    return (q / np.linalg.norm(q)).astype(DTYPE)


def make_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 homogeneous transform."""
    T = np.eye(4, dtype=DTYPE)
    T[:POINT_DIM, :POINT_DIM] = R
    T[:POINT_DIM, POINT_DIM] = p
    return T


def transform_inv(T: np.ndarray) -> np.ndarray:
    """Inverse of a homogeneous transform."""
    R = T[:POINT_DIM, :POINT_DIM]
    p = T[:POINT_DIM, POINT_DIM]
    Ti = np.eye(4, dtype=DTYPE)
    Ti[:POINT_DIM, :POINT_DIM] = R.T
    Ti[:POINT_DIM, POINT_DIM] = -(R.T @ p)
    return Ti


def joint_transform(axis: np.ndarray, jtype: str, value: float) -> np.ndarray:
    """Motion transform of a single joint at the given value."""
    T = np.eye(4, dtype=DTYPE)
    if jtype in ("revolute", "continuous"):
        T[:POINT_DIM, :POINT_DIM] = exp_so3(np.asarray(axis, dtype=DTYPE) * DTYPE(value))
    elif jtype == "prismatic":
        T[:POINT_DIM, POINT_DIM] = np.asarray(axis, dtype=DTYPE) * DTYPE(value)
    elif jtype == "fixed":
        pass
    else:
        raise ValueError(f"unsupported joint type: {jtype}")
    return T


def pose_delta(
    p_from: np.ndarray, R_from: np.ndarray, p_to: np.ndarray, R_to: np.ndarray
) -> np.ndarray:
    """Spatial error taking (p_from, R_from) to (p_to, R_to), stacked [dp; dw]."""
    dp = np.asarray(p_to, dtype=DTYPE) - np.asarray(p_from, dtype=DTYPE)
    dw = log_so3(np.asarray(R_to, dtype=DTYPE) @ np.asarray(R_from, dtype=DTYPE).T)
    return np.concatenate([dp, dw]).astype(DTYPE)

"""Dimensional constants and the pipeline dtype.

TASK_DIM is the dimension of se(3) -- a property of space, not of any robot.  It
is the ONLY fixed dimension in this package.  Everything else (joint count, chain
count, effector width, constraint count) is derived from cell config at load time.

The whole pipeline runs in float32, reference and kernels alike, so that the
NumPy reference is a bit-level specification of what the kernels compute rather
than a higher-precision approximation of it.  Sim/real parity is the product
requirement; matching precision between the two implementations is part of it.
"""

import numpy as np

#: Dimension of a spatial velocity / pose error (se(3)).
TASK_DIM = 6

#: Dimension of a position vector.
POINT_DIM = 3

#: Dimension of a quaternion.
QUAT_DIM = 4

#: Dimension of a pose serialised as position + quaternion.
POSE_DIM = POINT_DIM + QUAT_DIM

#: Pipeline dtype.  float32 everywhere -- see module docstring.
DTYPE = np.float32

#: Sentinel used for the right-hand side of an inactive constraint row.  Rows keep
#: their slot when inactive so that constraint count and ordering are fixed per
#: cell, which is what makes the solver deterministic across ticks and devices.
BIG = np.float32(1.0e6)

#: Numerical floor used where a division by a norm could otherwise blow up.
EPS = np.float32(1.0e-9)

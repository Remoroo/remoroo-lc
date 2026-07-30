"""Warp kernels: the same three layers, batched, on CPU and CUDA.

Every kernel here launches one thread per environment and does that
environment's whole job inside the thread.  Nothing is reduced across threads,
so nothing needs atomics and nothing depends on scheduling; two runs on the same
device produce identical bits, and CPU and CUDA agree to float32 rounding.

The code deliberately mirrors `remoroo_lc.reference` statement for statement --
same series cutoffs, same branch conditions, same Cholesky loop order, same
Gauss-Seidel sweep order -- because the equivalence test between them is what
makes the readable NumPy version a specification of this one rather than a
separate program that happens to agree.

Warp is Apache-2.0 (verified by scripts/license_check.py, which is what would
have sent us down the PyTorch fallback path had it not been).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from remoroo_lc.constants import DTYPE, POINT_DIM, TASK_DIM
from remoroo_lc.kernels.structure import CellStructure, build_structure
from remoroo_lc.reference.kinematics import KIND_PRISMATIC, KIND_REVOLUTE
from remoroo_lc.schema import CellSpec

wp.config.quiet = True
wp.init()

# Compile-time constants.  TASK_DIM is the dimension of se(3); see constants.py.
TDIM = wp.constant(TASK_DIM)
PDIM = wp.constant(POINT_DIM)
TAYLOR = wp.constant(1.0e-4)
NEAR_PI = wp.constant(3.1405926536)  # pi - 1e-3
BIG_F = wp.constant(1.0e6)
EPS_F = wp.constant(1.0e-9)
K_REV = wp.constant(KIND_REVOLUTE)
K_PRI = wp.constant(KIND_PRISMATIC)
E_PLANE = wp.constant(0)
E_BOX = wp.constant(1)
E_SPHERE = wp.constant(2)
E_CYL = wp.constant(3)
PAIR_RR = wp.constant(0)


def cuda_available() -> bool:
    return wp.is_cuda_available()


def _pad1(a: np.ndarray) -> np.ndarray:
    """Pad a possibly-empty 1-D array to length 1 so Warp can bind it."""
    return a if a.size else np.zeros(1, dtype=DTYPE)


# --------------------------------------------------------------------------- #
# device functions -- one-to-one with remoroo_lc.spatial
# --------------------------------------------------------------------------- #


@wp.func
def rot_of(T: wp.mat44f) -> wp.mat33f:
    return wp.mat33f(
        T[0, 0], T[0, 1], T[0, 2],
        T[1, 0], T[1, 1], T[1, 2],
        T[2, 0], T[2, 1], T[2, 2],
    )


@wp.func
def pos_of(T: wp.mat44f) -> wp.vec3f:
    return wp.vec3f(T[0, 3], T[1, 3], T[2, 3])


@wp.func
def make_T(R: wp.mat33f, p: wp.vec3f) -> wp.mat44f:
    return wp.mat44f(
        R[0, 0], R[0, 1], R[0, 2], p[0],
        R[1, 0], R[1, 1], R[1, 2], p[1],
        R[2, 0], R[2, 1], R[2, 2], p[2],
        0.0, 0.0, 0.0, 1.0,
    )


@wp.func
def exp_so3(w: wp.vec3f) -> wp.mat33f:
    theta = wp.length(w)
    a = float(0.0)
    b = float(0.0)
    if theta < TAYLOR:
        a = 1.0 - theta * theta / 6.0
        b = 0.5 - theta * theta / 24.0
    else:
        a = wp.sin(theta) / theta
        b = (1.0 - wp.cos(theta)) / (theta * theta)
    K = wp.mat33f(
        0.0, -w[2], w[1],
        w[2], 0.0, -w[0],
        -w[1], w[0], 0.0,
    )
    return wp.identity(n=3, dtype=wp.float32) + a * K + b * (K * K)


@wp.func
def log_so3(R: wp.mat33f) -> wp.vec3f:
    """Mirrors spatial.log_so3, including the atan2 angle and the near-pi branch."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    anti = wp.vec3f(R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1])
    sin_t = 0.5 * wp.length(anti)
    cos_t = wp.clamp((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = wp.atan2(sin_t, cos_t)

    if theta < TAYLOR:
        s = 0.5 * (1.0 + theta * theta / 6.0)
        return s * anti

    if theta > NEAR_PI:
        denom = wp.max(1.0 - cos_t, 1.0e-12)
        a00 = 0.5 * (R[0, 0] + R[0, 0]) - cos_t
        a11 = 0.5 * (R[1, 1] + R[1, 1]) - cos_t
        a22 = 0.5 * (R[2, 2] + R[2, 2]) - cos_t
        a01 = 0.5 * (R[0, 1] + R[1, 0])
        a02 = 0.5 * (R[0, 2] + R[2, 0])
        a12 = 0.5 * (R[1, 2] + R[2, 1])
        x0 = wp.sqrt(wp.max(a00 / denom, 0.0))
        x1 = wp.sqrt(wp.max(a11 / denom, 0.0))
        x2 = wp.sqrt(wp.max(a22 / denom, 0.0))
        k = int(0)
        if x1 > x0:
            k = 1
        if k == 0:
            if x2 > x0:
                k = 2
        else:
            if x2 > x1:
                k = 2
        if k == 0:
            if x0 > 0.0:
                x1 = a01 / (denom * x0)
                x2 = a02 / (denom * x0)
        elif k == 1:
            if x1 > 0.0:
                x0 = a01 / (denom * x1)
                x2 = a12 / (denom * x1)
        else:
            if x2 > 0.0:
                x0 = a02 / (denom * x2)
                x1 = a12 / (denom * x2)
        axis = wp.vec3f(x0, x1, x2)
        if anti[k] < 0.0:
            axis = -axis
        nrm = wp.max(wp.length(axis), 1.0e-12)
        return (axis / nrm) * theta

    s = 0.5 * theta / wp.sin(theta)
    return s * anti


@wp.func
def stop_velocity(d: float, a_max: float, j_max: float) -> float:
    """Mirrors interpolator.stop_velocity."""
    dd = wp.max(d, 0.0)
    v_tri = wp.pow(dd * dd * j_max, 1.0 / 3.0)
    b = a_max * a_max / j_max
    v_trap = 0.5 * (-b + wp.sqrt(b * b + 8.0 * a_max * dd))
    if v_tri <= b:
        return v_tri
    return v_trap


@wp.func
def sign_f(x: float) -> float:
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


# --------------------------------------------------------------------------- #
# Layer 0: forward kinematics and Jacobians
# --------------------------------------------------------------------------- #


@wp.kernel
def k_fk(
    q: wp.array2d(dtype=wp.float32),
    parent: wp.array(dtype=wp.int32),
    qindex: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    locked: wp.array(dtype=wp.float32),
    origin: wp.array(dtype=wp.mat44f),
    axis: wp.array(dtype=wp.vec3f),
    base: wp.array(dtype=wp.mat44f),
    n_links: int,
    link_T: wp.array2d(dtype=wp.mat44f),
    joint_p: wp.array2d(dtype=wp.vec3f),
    joint_z: wp.array2d(dtype=wp.vec3f),
):
    b = wp.tid()
    for l in range(n_links):
        p = parent[l]
        if p < 0:
            link_T[b, l] = base[l]
        else:
            T_joint = link_T[b, p] * origin[l]
            gi = qindex[l]
            value = float(0.0)
            if gi >= 0:
                value = q[b, gi]
            else:
                value = locked[l]
            M = wp.identity(n=4, dtype=wp.float32)
            kd = kind[l]
            if kd == K_REV:
                Rj = exp_so3(axis[l] * value)
                M = make_T(Rj, wp.vec3f(0.0, 0.0, 0.0))
            elif kd == K_PRI:
                M = make_T(wp.identity(n=3, dtype=wp.float32), axis[l] * value)
            link_T[b, l] = T_joint * M
            if gi >= 0:
                joint_p[b, gi] = pos_of(T_joint)
                joint_z[b, gi] = rot_of(T_joint) * axis[l]


@wp.kernel
def k_frames(
    link_T: wp.array2d(dtype=wp.mat44f),
    joint_p: wp.array2d(dtype=wp.vec3f),
    joint_z: wp.array2d(dtype=wp.vec3f),
    joint_kind: wp.array(dtype=wp.int32),
    support: wp.array2d(dtype=wp.int32),
    tcp_link: wp.array(dtype=wp.int32),
    tcp_offset: wp.array(dtype=wp.mat44f),
    sphere_link: wp.array(dtype=wp.int32),
    sphere_centre: wp.array(dtype=wp.vec3f),
    n_joints: int,
    n_tcps: int,
    n_spheres: int,
    tcp_p: wp.array2d(dtype=wp.vec3f),
    tcp_R: wp.array2d(dtype=wp.mat33f),
    Jv: wp.array3d(dtype=wp.vec3f),
    Jw: wp.array3d(dtype=wp.vec3f),
    sphere_c: wp.array2d(dtype=wp.vec3f),
    Js: wp.array3d(dtype=wp.vec3f),
):
    b = wp.tid()
    zero = wp.vec3f(0.0, 0.0, 0.0)
    for i in range(n_tcps):
        li = tcp_link[i]
        T = link_T[b, li] * tcp_offset[i]
        p = pos_of(T)
        tcp_p[b, i] = p
        tcp_R[b, i] = rot_of(T)
        for j in range(n_joints):
            if support[li, j] == 0:
                Jv[b, i, j] = zero
                Jw[b, i, j] = zero
            else:
                if joint_kind[j] == K_REV:
                    Jv[b, i, j] = wp.cross(joint_z[b, j], p - joint_p[b, j])
                    Jw[b, i, j] = joint_z[b, j]
                else:
                    Jv[b, i, j] = joint_z[b, j]
                    Jw[b, i, j] = zero
    for s in range(n_spheres):
        ls = sphere_link[s]
        T = link_T[b, ls]
        c = rot_of(T) * sphere_centre[s] + pos_of(T)
        sphere_c[b, s] = c
        for j in range(n_joints):
            if support[ls, j] == 0:
                Js[b, s, j] = zero
            else:
                if joint_kind[j] == K_REV:
                    Js[b, s, j] = wp.cross(joint_z[b, j], c - joint_p[b, j])
                else:
                    Js[b, s, j] = joint_z[b, j]


# --------------------------------------------------------------------------- #
# Layer 1: chunk interpolator
# --------------------------------------------------------------------------- #


@wp.kernel
def k_interp(
    x: wp.array3d(dtype=wp.float32),
    v: wp.array3d(dtype=wp.float32),
    a: wp.array3d(dtype=wp.float32),
    eff: wp.array2d(dtype=wp.float32),
    waypoints: wp.array4d(dtype=wp.float32),
    eff_waypoints: wp.array3d(dtype=wp.float32),
    anchor: wp.array3d(dtype=wp.float32),
    eff_anchor: wp.array2d(dtype=wp.float32),
    t_chunk: wp.array(dtype=wp.float32),
    n_way: wp.array(dtype=wp.int32),
    task_v: wp.array(dtype=wp.float32),
    task_a: wp.array(dtype=wp.float32),
    task_j: wp.array(dtype=wp.float32),
    eff_rate: wp.array(dtype=wp.float32),
    tcp_base: wp.array(dtype=wp.mat44f),
    R_ref: wp.array2d(dtype=wp.mat33f),
    n_tcps: int,
    eff_dim: int,
    dt_c: float,
    dt_p: float,
    brake: float,
    lag_ticks: float,
    pos_lag_ticks: float,
    track_mode: int,
    way_v: wp.array4d(dtype=wp.float32),
    seam_x: wp.array3d(dtype=wp.float32),
    seam_v: wp.array3d(dtype=wp.float32),
    p_cmd: wp.array2d(dtype=wp.vec3f),
    R_cmd: wp.array2d(dtype=wp.mat33f),
):
    b = wp.tid()
    kk = n_way[b]

    # --- time-interpolated chunk target -------------------------------------- #
    s = t_chunk[b] / dt_p
    if s < 0.0:
        s = 0.0
    if kk > 0:
        if s > float(kk):
            s = float(kk)
    i0 = int(wp.floor(s))
    if kk > 0:
        if i0 > kk - 1:
            i0 = kk - 1
    else:
        i0 = 0
    frac = s - float(i0)

    t_a = lag_ticks * dt_c
    t_v = pos_lag_ticks * dt_c
    if track_mode == 1:
        # Follower: evaluate the chunk Hermite at the END of this tick.  Pure
        # polynomial -- no cusped laws, no clamps; the chunk defines the motion
        # and joint-space safety lives in layer 3.  Mirrors
        # ChunkInterpolator._follower_state.
        if kk == 0:
            for i in range(n_tcps):
                for c in range(TDIM):
                    v[b, i, c] = 0.0
                    a[b, i, c] = 0.0
        else:
            t_next = t_chunk[b] + dt_c
            sn = t_next / dt_p
            for i in range(n_tcps):
                for c in range(TDIM):
                    x0 = float(0.0)
                    v0 = float(0.0)
                    x1 = float(0.0)
                    v1 = float(0.0)
                    tau = float(0.0)
                    seg_t = float(0.0)
                    hold = int(0)
                    if sn < float(kk):
                        j0 = int(wp.floor(sn))
                        if j0 > kk - 1:
                            j0 = kk - 1
                        tau = sn - float(j0)
                        seg_t = dt_p
                        if j0 == 0:
                            x0 = seam_x[b, i, c]
                            v0 = seam_v[b, i, c]
                        else:
                            x0 = waypoints[b, j0 - 1, i, c]
                            v0 = way_v[b, j0 - 1, i, c]
                        x1 = waypoints[b, j0, i, c]
                        v1 = way_v[b, j0, i, c]
                    else:
                        # Runway: one Hermite to (w_K + v_K*T/2, 0) over
                        # T = K*dt_p is exactly constant deceleration v_K/T.
                        seg_t = float(kk) * dt_p
                        tau = (t_next - float(kk) * dt_p) / seg_t
                        x0 = waypoints[b, kk - 1, i, c]
                        v0 = way_v[b, kk - 1, i, c]
                        x1 = x0 + v0 * seg_t * 0.5
                        v1 = 0.0
                        if tau >= 1.0:
                            hold = int(1)
                    if hold == 1:
                        x[b, i, c] = x1
                        v[b, i, c] = 0.0
                        a[b, i, c] = 0.0
                    else:
                        t2 = tau * tau
                        t3 = t2 * tau
                        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
                        h10 = t3 - 2.0 * t2 + tau
                        h01 = -2.0 * t3 + 3.0 * t2
                        h11 = t3 - t2
                        x[b, i, c] = h00 * x0 + h01 * x1 + seg_t * (h10 * v0 + h11 * v1)
                        v[b, i, c] = (
                            (6.0 * t2 - 6.0 * tau) * (x0 - x1) / seg_t
                            + (3.0 * t2 - 4.0 * tau + 1.0) * v0
                            + (3.0 * t2 - 2.0 * tau) * v1
                        )
                        a[b, i, c] = (
                            (12.0 * tau - 6.0) * (x0 - x1) / (seg_t * seg_t)
                            + (6.0 * tau - 4.0) * v0 / seg_t
                            + (6.0 * tau - 2.0) * v1 / seg_t
                        )
    else:
        for i in range(n_tcps):
            for c in range(TDIM):
                lo = float(0.0)
                hi = float(0.0)
                if kk == 0:
                    lo = anchor[b, i, c]
                    hi = anchor[b, i, c]
                else:
                    if i0 == 0:
                        lo = anchor[b, i, c]
                    else:
                        lo = waypoints[b, i0 - 1, i, c]
                    hi = waypoints[b, i0, i, c]
                x_tgt = lo + (hi - lo) * frac

                xv = x[b, i, c]
                vv = v[b, i, c]
                av = a[b, i, c]
                vmax = task_v[c]
                amax = task_a[c]
                jmax = task_j[c]

                e = x_tgt - xv
                v_ref = sign_f(e) * wp.min(
                    wp.min(vmax, stop_velocity(wp.abs(e), amax * brake, jmax * brake)),
                    wp.abs(e) / t_v,
                )
                dv = v_ref - vv
                a_ref = sign_f(dv) * wp.min(
                    wp.min(amax, wp.sqrt(2.0 * jmax * brake * wp.abs(dv))), wp.abs(dv) / t_a
                )
                j = wp.clamp((a_ref - av) / dt_c, -jmax, jmax)
                av = wp.clamp(av + j * dt_c, -amax, amax)
                vv = wp.clamp(vv + av * dt_c, -vmax, vmax)
                xv = xv + vv * dt_c

                x[b, i, c] = xv
                v[b, i, c] = vv
                a[b, i, c] = av

    # --- effector channels ---------------------------------------------------- #
    for c in range(eff_dim):
        lo = float(0.0)
        hi = float(0.0)
        if kk == 0:
            lo = eff_anchor[b, c]
            hi = eff_anchor[b, c]
        else:
            if i0 == 0:
                lo = eff_anchor[b, c]
            else:
                lo = eff_waypoints[b, i0 - 1, c]
            hi = eff_waypoints[b, i0, c]
        tgt = lo + (hi - lo) * frac
        step = eff_rate[c] * dt_c
        d = wp.clamp(tgt - eff[b, c], -step, step)
        eff[b, c] = wp.clamp(eff[b, c] + d, 0.0, 1.0)

    t_chunk[b] = t_chunk[b] + dt_c

    # --- base frame -> world -------------------------------------------------- #
    for i in range(n_tcps):
        pb = wp.vec3f(x[b, i, 0], x[b, i, 1], x[b, i, 2])
        rb = wp.vec3f(x[b, i, PDIM + 0], x[b, i, PDIM + 1], x[b, i, PDIM + 2])
        Rb = R_ref[b, i] * exp_so3(rb)
        Tb = tcp_base[i]
        p_cmd[b, i] = rot_of(Tb) * pb + pos_of(Tb)
        R_cmd[b, i] = rot_of(Tb) * Rb


# --------------------------------------------------------------------------- #
# Layer 2: stacked damped least squares
# --------------------------------------------------------------------------- #


@wp.kernel
def k_diffik(
    tcp_p: wp.array2d(dtype=wp.vec3f),
    tcp_R: wp.array2d(dtype=wp.mat33f),
    p_cmd: wp.array2d(dtype=wp.vec3f),
    R_cmd: wp.array2d(dtype=wp.mat33f),
    Jv: wp.array3d(dtype=wp.vec3f),
    Jw: wp.array3d(dtype=wp.vec3f),
    w_thresh: wp.array(dtype=wp.float32),
    v_cmd: wp.array3d(dtype=wp.float32),
    tcp_base: wp.array(dtype=wp.mat44f),
    qd_max: wp.array(dtype=wp.float32),
    n_joints: int,
    n_tcps: int,
    dt_c: float,
    v_max_lin: float,
    v_max_ang: float,
    kp_lin: float,
    kp_ang: float,
    lambda_min: float,
    lambda_max: float,
    A: wp.array3d(dtype=wp.float32),
    L: wp.array3d(dtype=wp.float32),
    Jf: wp.array3d(dtype=wp.float32),
    vtask: wp.array2d(dtype=wp.float32),
    yvec: wp.array2d(dtype=wp.float32),
    qd_des: wp.array2d(dtype=wp.float32),
    manip: wp.array2d(dtype=wp.float32),
    damping: wp.array2d(dtype=wp.float32),
):
    b = wp.tid()
    m6 = n_tcps * TDIM
    inv_dt = 1.0 / dt_c

    # --- task velocity, norm clamped per TCP --------------------------------- #
    for i in range(n_tcps):
        # Feedforward (Layer 1's own command velocity, base -> world) plus a
        # feedback term at a servo bandwidth.  See DiffIk.task_velocity.
        lin = (p_cmd[b, i] - tcp_p[b, i]) * kp_lin
        ang = log_so3(R_cmd[b, i] * wp.transpose(tcp_R[b, i])) * kp_ang
        Rb = rot_of(tcp_base[i])
        lin = lin + Rb * wp.vec3f(v_cmd[b, i, 0], v_cmd[b, i, 1], v_cmd[b, i, 2])
        ang = ang + Rb * wp.vec3f(
            v_cmd[b, i, PDIM + 0], v_cmd[b, i, PDIM + 1], v_cmd[b, i, PDIM + 2]
        )
        nl = wp.length(lin)
        na = wp.length(ang)
        if nl > v_max_lin:
            lin = lin * (v_max_lin / (nl + EPS_F))
        if na > v_max_ang:
            ang = ang * (v_max_ang / (na + EPS_F))
        for c in range(PDIM):
            vtask[b, i * TDIM + c] = lin[c]
            vtask[b, i * TDIM + PDIM + c] = ang[c]

    # --- materialise the stacked Jacobian once ------------------------------- #
    # Reading it back out of the vec3 arrays inside the Gram triple loop meant
    # 2 * m6 * m6 * n accessor calls per environment -- 3456 on the reference
    # cell -- each one a branch plus a vec3 load plus a component extract.
    # Writing it flat costs m6 * n and turns the Gram into plain float reads.
    for r in range(m6):
        ir = r / TDIM
        rr = r - ir * TDIM
        for j in range(n_joints):
            Jf[b, r, j] = jac_entry(Jv, Jw, b, ir, rr, j)

    # --- Gram, computed ONCE ------------------------------------------------- #
    # The per-TCP Gram J_i J_i^T that manipulability needs is exactly the i-th
    # diagonal block of the stacked J J^T, so computing them separately was
    # computing the same products twice.
    for r in range(m6):
        for c in range(m6):
            acc = float(0.0)
            for j in range(n_joints):
                acc += Jf[b, r, j] * Jf[b, c, j]
            A[b, r, c] = acc

    # --- manipulability from the diagonal blocks ----------------------------- #
    for i in range(n_tcps):
        base = i * TDIM
        det = float(1.0)
        ok_i = int(1)
        for r in range(TDIM):
            sv = A[b, base + r, base + r]
            for t in range(r):
                sv -= L[b, base + r, base + t] * L[b, base + r, base + t]
            if sv <= EPS_F:
                ok_i = 0
                break
            dg = wp.sqrt(sv)
            L[b, base + r, base + r] = dg
            det = det * dg
            for c in range(r + 1, TDIM):
                s2 = A[b, base + c, base + r]
                for t in range(r):
                    s2 -= L[b, base + c, base + t] * L[b, base + r, base + t]
                L[b, base + c, base + r] = s2 / dg
        if ok_i == 1:
            manip[b, i] = det
        else:
            manip[b, i] = 0.0
        frac = wp.max(0.0, 1.0 - manip[b, i] / w_thresh[i])
        damping[b, i] = lambda_min + lambda_max * frac * frac

    # --- damping onto the diagonal ------------------------------------------- #
    for r in range(m6):
        ir = r / TDIM
        lam = damping[b, ir]
        A[b, r, r] = A[b, r, r] + lam * lam

    # --- Cholesky, with the reference's single fixed ridge retry -------------- #
    ok = chol(A, L, b, m6, 1.0e-12)
    if ok == 0:
        for r in range(m6):
            A[b, r, r] = A[b, r, r] + 1.0e-9
        ok = chol(A, L, b, m6, 1.0e-12)
    for r in range(m6):
        yvec[b, r] = 0.0
    if ok == 1:
        chol_solve(L, vtask, yvec, b, m6)

    for j in range(n_joints):
        acc = float(0.0)
        for r in range(m6):
            acc += Jf[b, r, j] * yvec[b, r]
        qd_des[b, j] = acc

    # Feasibility governor: uniform, direction-preserving.  See DiffIk.solve --
    # the damped inverse can ask for 100x what the arm can do near a singular
    # direction, and letting that reach Layer 3 makes its box annihilate the
    # tracking component along with the excess.
    worst = float(0.0)
    for j in range(n_joints):
        frac = wp.abs(qd_des[b, j]) / wp.max(qd_max[j], EPS_F)
        if frac > worst:
            worst = frac
    if worst > 1.0:
        g = 1.0 / worst
        for j in range(n_joints):
            qd_des[b, j] = qd_des[b, j] * g


@wp.func
def jac_entry(
    Jv: wp.array3d(dtype=wp.vec3f),
    Jw: wp.array3d(dtype=wp.vec3f),
    b: int,
    i: int,
    row: int,
    j: int,
) -> float:
    """Element (row, j) of TCP i's TASK_DIM x n Jacobian."""
    if row < PDIM:
        return Jv[b, i, j][row]
    return Jw[b, i, j][row - PDIM]


@wp.func
def chol(
    A: wp.array3d(dtype=wp.float32), L: wp.array3d(dtype=wp.float32), b: int, m: int, floor: float
) -> int:
    for r in range(m):
        for c in range(m):
            L[b, r, c] = 0.0
    for r in range(m):
        s = A[b, r, r]
        for t in range(r):
            s -= L[b, r, t] * L[b, r, t]
        if s <= floor:
            return 0
        d = wp.sqrt(s)
        L[b, r, r] = d
        for c in range(r + 1, m):
            s2 = A[b, c, r]
            for t in range(r):
                s2 -= L[b, c, t] * L[b, r, t]
            L[b, c, r] = s2 / d
    return 1


@wp.func
def chol_solve(
    L: wp.array3d(dtype=wp.float32),
    rhs: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
    b: int,
    m: int,
):
    for r in range(m):
        s = rhs[b, r]
        for t in range(r):
            s -= L[b, r, t] * out[b, t]
        out[b, r] = s / L[b, r, r]
    for rr in range(m):
        r = m - 1 - rr
        s = out[b, r]
        for t in range(r + 1, m):
            s -= L[b, t, r] * out[b, t]
        out[b, r] = s / L[b, r, r]


# --------------------------------------------------------------------------- #
# Layer 3a: constraint rows
# --------------------------------------------------------------------------- #


@wp.func
def env_distance(
    etype: int, pose: wp.mat44f, dims: wp.vec3f, p: wp.vec3f
) -> wp.vec4f:
    """Returns (distance, normal.x, normal.y, normal.z); mirrors walls.env_distance."""
    if etype == E_PLANE:
        n = wp.vec3f(dims[0], dims[1], dims[2])
        d = wp.dot(n, p - pos_of(pose))
        return wp.vec4f(d, n[0], n[1], n[2])
    R = rot_of(pose)
    c = pos_of(pose)
    local = wp.transpose(R) * (p - c)
    if etype == E_SPHERE:
        r = dims[0]
        dist = wp.length(local)
        nl = local / wp.max(dist, EPS_F)
        nw = R * nl
        return wp.vec4f(dist - r, nw[0], nw[1], nw[2])
    if etype == E_BOX:
        closest = wp.vec3f(
            wp.clamp(local[0], -dims[0], dims[0]),
            wp.clamp(local[1], -dims[1], dims[1]),
            wp.clamp(local[2], -dims[2], dims[2]),
        )
        diff = local - closest
        dist = wp.length(diff)
        if dist > EPS_F:
            nw = R * (diff / dist)
            return wp.vec4f(dist, nw[0], nw[1], nw[2])
        g0 = dims[0] - wp.abs(local[0])
        g1 = dims[1] - wp.abs(local[1])
        g2 = dims[2] - wp.abs(local[2])
        k = int(0)
        gm = g0
        if g1 < gm:
            k = 1
            gm = g1
        if g2 < gm:
            k = 2
            gm = g2
        sgn = sign_f(local[k])
        if sgn == 0.0:
            sgn = 1.0
        nl = wp.vec3f(0.0, 0.0, 0.0)
        if k == 0:
            nl = wp.vec3f(sgn, 0.0, 0.0)
        elif k == 1:
            nl = wp.vec3f(0.0, sgn, 0.0)
        else:
            nl = wp.vec3f(0.0, 0.0, sgn)
        nw = R * nl
        return wp.vec4f(-gm, nw[0], nw[1], nw[2])
    # cylinder, axis along local z
    r = dims[0]
    half_h = dims[1]
    rho = wp.sqrt(local[0] * local[0] + local[1] * local[1])
    ux = local[0] / wp.max(rho, EPS_F)
    uy = local[1] / wp.max(rho, EPS_F)
    d_rad = rho - r
    d_ax = wp.abs(local[2]) - half_h
    sz = sign_f(local[2])
    if sz == 0.0:
        sz = 1.0
    if d_rad > 0.0 and d_ax > 0.0:
        dist = wp.sqrt(d_rad * d_rad + d_ax * d_ax)
        nl = wp.vec3f(ux * d_rad, uy * d_rad, sz * d_ax) / wp.max(dist, EPS_F)
        nw = R * nl
        return wp.vec4f(dist, nw[0], nw[1], nw[2])
    if d_rad > d_ax:
        nw = R * wp.vec3f(ux, uy, 0.0)
        return wp.vec4f(d_rad, nw[0], nw[1], nw[2])
    nw = R * wp.vec3f(0.0, 0.0, sz)
    return wp.vec4f(d_ax, nw[0], nw[1], nw[2])


@wp.kernel
def k_walls(
    q: wp.array2d(dtype=wp.float32),
    sphere_c: wp.array2d(dtype=wp.vec3f),
    Js: wp.array3d(dtype=wp.vec3f),
    sphere_radius: wp.array(dtype=wp.float32),
    pair_kind: wp.array(dtype=wp.int32),
    pair_a: wp.array(dtype=wp.int32),
    pair_b: wp.array(dtype=wp.int32),
    env_type: wp.array(dtype=wp.int32),
    env_pose: wp.array(dtype=wp.mat44f),
    env_dims: wp.array(dtype=wp.vec3f),
    q_lo: wp.array(dtype=wp.float32),
    q_hi: wp.array(dtype=wp.float32),
    lam: wp.array2d(dtype=wp.float32),
    n_joints: int,
    n_pairs: int,
    jl_xi: float,
    jl_safe: float,
    jl_infl: float,
    cd_xi: float,
    cd_safe: float,
    cd_infl: float,
    G: wp.array3d(dtype=wp.float32),
    h: wp.array2d(dtype=wp.float32),
    dist_out: wp.array2d(dtype=wp.float32),
    min_pair: wp.array(dtype=wp.float32),
    min_margin: wp.array(dtype=wp.float32),
):
    b = wp.tid()
    span = jl_infl - jl_safe

    for j in range(n_joints):
        for c in range(n_joints):
            G[b, j, c] = 0.0
            G[b, n_joints + j, c] = 0.0
        G[b, j, j] = 1.0
        G[b, n_joints + j, j] = -1.0
        d_up = q_hi[j] - q[b, j]
        d_lo = q[b, j] - q_lo[j]
        dist_out[b, j] = d_up
        dist_out[b, n_joints + j] = d_lo
        if d_up < jl_infl:
            h[b, j] = jl_xi * (d_up - jl_safe) / span
        else:
            h[b, j] = BIG_F
        if d_lo < jl_infl:
            h[b, n_joints + j] = jl_xi * (d_lo - jl_safe) / span
        else:
            h[b, n_joints + j] = BIG_F

    mm = BIG_F
    for j in range(n_joints):
        mm = wp.min(mm, wp.min(dist_out[b, j], dist_out[b, n_joints + j]))
    min_margin[b] = mm

    base = 2 * n_joints
    cspan = cd_infl - cd_safe
    mp = BIG_F
    for p in range(n_pairs):
        row = base + p
        ia = pair_a[p]
        d = float(0.0)
        nrm = wp.vec3f(0.0, 0.0, 0.0)
        ib = pair_b[p]
        if pair_kind[p] == PAIR_RR:
            diff = sphere_c[b, ia] - sphere_c[b, ib]
            dist = wp.length(diff)
            nrm = diff / (dist + EPS_F)
            d = dist - sphere_radius[ia] - sphere_radius[ib]
        else:
            res = env_distance(env_type[ib], env_pose[ib], env_dims[ib], sphere_c[b, ia])
            nrm = wp.vec3f(res[1], res[2], res[3])
            d = res[0] - sphere_radius[ia]
        dist_out[b, row] = d
        if d < cd_infl:
            h[b, row] = cd_xi * (d - cd_safe) / cspan
        else:
            h[b, row] = BIG_F
        mp = wp.min(mp, d)

        # Project the Jacobians only for rows the solver will actually read.
        # The distance is one subtraction and a norm; the row is n dot products
        # over 3-vectors, so filling all of them costs 20x what deciding costs.
        # A pair beyond d_infl with no carried multiplier is parked, and a parked
        # row's G is never touched by the sweep or by the diagnostics.
        if d < cd_infl or lam[b, row] != 0.0:
            if pair_kind[p] == PAIR_RR:
                for j in range(n_joints):
                    G[b, row, j] = -wp.dot(nrm, Js[b, ia, j] - Js[b, ib, j])
            else:
                for j in range(n_joints):
                    G[b, row, j] = -wp.dot(nrm, Js[b, ia, j])
    min_pair[b] = mp


# --------------------------------------------------------------------------- #
# Layer 3b: dual projected Gauss-Seidel, then the box clamp and output
# --------------------------------------------------------------------------- #


@wp.kernel
def k_solve(
    q: wp.array2d(dtype=wp.float32),
    qd_des: wp.array2d(dtype=wp.float32),
    q_rest: wp.array(dtype=wp.float32),
    G: wp.array3d(dtype=wp.float32),
    h: wp.array2d(dtype=wp.float32),
    lam: wp.array2d(dtype=wp.float32),
    qd_max: wp.array(dtype=wp.float32),
    w_joint: wp.array(dtype=wp.float32),
    n_joints: int,
    n_rows: int,
    k_post: float,
    w_post: float,
    rho: float,
    iterations: int,
    dt_c: float,
    qd_out: wp.array2d(dtype=wp.float32),
    q_target: wp.array2d(dtype=wp.float32),
    n_active: wp.array(dtype=wp.int32),
    max_violation: wp.array(dtype=wp.float32),
    slack_norm: wp.array(dtype=wp.float32),
    box_clamped: wp.array(dtype=wp.int32),
    Dscratch: wp.array2d(dtype=wp.float32),
    qdscratch: wp.array2d(dtype=wp.float32),
    live_idx: wp.array2d(dtype=wp.int32),
):
    b = wp.tid()
    inv_rho = 1.0 / rho

    # Compact the live rows once per tick instead of testing every row on every
    # sweep.  A row parked at BIG with lambda = 0 is provably a no-op (see
    # solver.py), and at rest EVERY row is parked -- so the sweep loop was
    # executing 32 * n_rows iterations to do nothing at all.  Measured on the
    # A10G at 4096 environments: 8.9 ms of a 13.3 ms step, on an empty active set.
    #
    # Exact, not approximate: the compacted list preserves row order, so
    # Gauss-Seidel visits the same rows in the same sequence and produces the
    # same bits.
    n_live = int(0)
    for k in range(n_rows):
        if h[b, k] < BIG_F or lam[b, k] != 0.0:
            live_idx[b, n_live] = k
            n_live += 1

    for j in range(n_joints):
        Hinv = 1.0 / (w_joint[j] + w_post)
        qd_post = -k_post * (q[b, j] - q_rest[j])
        qdscratch[b, j] = Hinv * (w_joint[j] * qd_des[b, j] + w_post * qd_post)

    # D is only ever read for a live row, so only live rows pay for it.
    for kk in range(n_live):
        k = live_idx[b, kk]
        acc = float(0.0)
        for j in range(n_joints):
            Hinv = 1.0 / (w_joint[j] + w_post)
            acc += G[b, k, j] * Hinv * G[b, k, j]
        Dscratch[b, k] = wp.max(acc + inv_rho, 1.0e-12)

    for k in range(n_rows):
        lk = lam[b, k]
        if lk != 0.0:
            for j in range(n_joints):
                Hinv = 1.0 / (w_joint[j] + w_post)
                qdscratch[b, j] = qdscratch[b, j] - Hinv * G[b, k, j] * lk

    for _it in range(iterations):
        for kk in range(n_live):
            k = live_idx[b, kk]
            phi = -h[b, k] - lam[b, k] * inv_rho
            for j in range(n_joints):
                phi += G[b, k, j] * qdscratch[b, j]
            new = lam[b, k] + phi / Dscratch[b, k]
            if new < 0.0:
                new = 0.0
            dl = new - lam[b, k]
            if dl != 0.0:
                lam[b, k] = new
                for j in range(n_joints):
                    Hinv = 1.0 / (w_joint[j] + w_post)
                    qdscratch[b, j] = qdscratch[b, j] - Hinv * G[b, k, j] * dl

    # Uniform scaling, not per-joint clipping: clipping bends the direction of
    # q_dot and the collision rows constrain a projection of it, so a bent q_dot
    # can approach a pair the solver had decided to retreat from.  See solver.py.
    over = float(0.0)
    for j in range(n_joints):
        over = wp.max(over, wp.abs(qdscratch[b, j]) / qd_max[j])
    scale = float(1.0)
    nclamp = int(0)
    if over > 1.0:
        scale = 1.0 / over
        for j in range(n_joints):
            if wp.abs(qdscratch[b, j]) > qd_max[j]:
                nclamp += 1
    for j in range(n_joints):
        v = qdscratch[b, j] * scale
        qd_out[b, j] = v
        q_target[b, j] = q[b, j] + v * dt_c
    box_clamped[b] = nclamp

    # Diagnostics over the live rows only.  A parked row sits at h = BIG, so its
    # violation is about -1e6 and can never be the maximum, and only a live row
    # can carry a multiplier.  Reading G for parked rows would also defeat the
    # point of not having filled it.
    na = int(0)
    mv = float(-BIG_F)
    sl = float(0.0)
    for kk in range(n_live):
        k = live_idx[b, kk]
        if lam[b, k] > 0.0:
            na += 1
        acc = -h[b, k]
        for j in range(n_joints):
            acc += G[b, k, j] * qd_out[b, j]
        mv = wp.max(mv, acc)
        s = lam[b, k] * inv_rho
        sl += s * s
    n_active[b] = na
    max_violation[b] = mv
    slack_norm[b] = wp.sqrt(sl)


# --------------------------------------------------------------------------- #
# chunk decoding (host-side shape, device-side maths)
# --------------------------------------------------------------------------- #


@wp.kernel
def k_set_chunk(
    actions: wp.array3d(dtype=wp.float32),
    tcp_p: wp.array2d(dtype=wp.vec3f),
    tcp_R: wp.array2d(dtype=wp.mat33f),
    tcp_base_inv: wp.array(dtype=wp.mat44f),
    R_ref: wp.array2d(dtype=wp.mat33f),
    pose_start: wp.array(dtype=wp.int32),
    eff_start: wp.array(dtype=wp.int32),
    eff_width: wp.array(dtype=wp.int32),
    eff_offset: wp.array(dtype=wp.int32),
    x: wp.array3d(dtype=wp.float32),
    v: wp.array3d(dtype=wp.float32),
    n_tcps: int,
    eff_dim: int,
    k_steps: int,
    delta_mode: int,
    track_mode: int,
    dt_p: float,
    waypoints: wp.array4d(dtype=wp.float32),
    eff_waypoints: wp.array3d(dtype=wp.float32),
    anchor: wp.array3d(dtype=wp.float32),
    eff_anchor: wp.array2d(dtype=wp.float32),
    eff: wp.array2d(dtype=wp.float32),
    t_chunk: wp.array(dtype=wp.float32),
    n_way: wp.array(dtype=wp.int32),
    way_v: wp.array4d(dtype=wp.float32),
    seam_x: wp.array3d(dtype=wp.float32),
    seam_v: wp.array3d(dtype=wp.float32),
):
    b = wp.tid()
    for i in range(n_tcps):
        Ti = tcp_base_inv[i]
        p_base = rot_of(Ti) * tcp_p[b, i] + pos_of(Ti)
        R_base = rot_of(Ti) * tcp_R[b, i]

        r_run = wp.vec3f(x[b, i, PDIM + 0], x[b, i, PDIM + 1], x[b, i, PDIM + 2])
        ref_T = wp.transpose(R_ref[b, i])
        r_anchor = r_run + log_so3((ref_T * R_base) * wp.transpose(exp_so3(r_run)))
        for c in range(PDIM):
            anchor[b, i, c] = p_base[c]
            anchor[b, i, PDIM + c] = r_anchor[c]

        p_prev = p_base
        R_prev = R_base
        r_prev = r_anchor
        for k in range(k_steps):
            o = pose_start[i]
            dp = wp.vec3f(actions[b, k, o], actions[b, k, o + 1], actions[b, k, o + 2])
            dw = wp.vec3f(
                actions[b, k, o + PDIM], actions[b, k, o + PDIM + 1], actions[b, k, o + PDIM + 2]
            )
            p_new = wp.vec3f(0.0, 0.0, 0.0)
            R_new = wp.identity(n=3, dtype=wp.float32)
            if delta_mode == 0:
                p_new = p_prev + dp
                R_new = exp_so3(dw) * R_prev
            else:
                p_new = p_base + dp
                R_new = exp_so3(dw) * R_base
            r_new = r_prev + log_so3((ref_T * R_new) * wp.transpose(exp_so3(r_prev)))
            for c in range(PDIM):
                waypoints[b, k, i, c] = p_new[c]
                waypoints[b, k, i, PDIM + c] = r_new[c]
            p_prev = p_new
            R_prev = R_new
            r_prev = r_new

        for c in range(eff_width[i]):
            for k in range(k_steps):
                eff_waypoints[b, k, eff_offset[i] + c] = actions[b, k, eff_start[i] + c]

    # Loop the CONFIGURED width, not the buffer's.  Effector buffers are padded to
    # length 1 so that a g = 0 cell still has a bindable array; iterating the
    # padding would read past the end of eff_default and take the process out.
    for c in range(eff_dim):
        eff_anchor[b, c] = eff[b, c]
    t_chunk[b] = 0.0
    n_way[b] = k_steps

    # Follower knots: the seam is the command state at chunk arrival (knot 0 of
    # segment 0 -- C1 continuity across chunk replacement), and each waypoint's
    # velocity is read off its neighbours in the decoded chain.  Mirrors the
    # reference exactly; see ChunkInterpolator.set_chunk.
    if track_mode == 1:
        for i in range(n_tcps):
            for c in range(TDIM):
                seam_x[b, i, c] = x[b, i, c]
                seam_v[b, i, c] = v[b, i, c]
                if k_steps == 1:
                    way_v[b, 0, i, c] = (waypoints[b, 0, i, c] - anchor[b, i, c]) / dt_p
                else:
                    for k in range(k_steps):
                        if k == k_steps - 1:
                            way_v[b, k, i, c] = (
                                waypoints[b, k, i, c] - waypoints[b, k - 1, i, c]
                            ) / dt_p
                        else:
                            prev = float(0.0)
                            if k == 0:
                                prev = anchor[b, i, c]
                            else:
                                prev = waypoints[b, k - 1, i, c]
                            way_v[b, k, i, c] = (waypoints[b, k + 1, i, c] - prev) / (
                                2.0 * dt_p
                            )


@wp.kernel
def k_reset(
    tcp_p: wp.array2d(dtype=wp.vec3f),
    tcp_R: wp.array2d(dtype=wp.mat33f),
    tcp_base_inv: wp.array(dtype=wp.mat44f),
    eff_default: wp.array(dtype=wp.float32),
    n_tcps: int,
    eff_dim: int,
    x: wp.array3d(dtype=wp.float32),
    v: wp.array3d(dtype=wp.float32),
    a: wp.array3d(dtype=wp.float32),
    eff: wp.array2d(dtype=wp.float32),
    R_ref: wp.array2d(dtype=wp.mat33f),
    anchor: wp.array3d(dtype=wp.float32),
    eff_anchor: wp.array2d(dtype=wp.float32),
    lam: wp.array2d(dtype=wp.float32),
    t_chunk: wp.array(dtype=wp.float32),
    n_way: wp.array(dtype=wp.int32),
):
    b = wp.tid()
    for i in range(n_tcps):
        Ti = tcp_base_inv[i]
        p_base = rot_of(Ti) * tcp_p[b, i] + pos_of(Ti)
        # r = 0 by construction: the reference orientation IS this one.  Never
        # take log_so3 of an orientation that may be near a half turn.
        R_ref[b, i] = rot_of(Ti) * tcp_R[b, i]
        for c in range(PDIM):
            x[b, i, c] = p_base[c]
            x[b, i, PDIM + c] = 0.0
            anchor[b, i, c] = p_base[c]
            anchor[b, i, PDIM + c] = 0.0
        for c in range(TDIM):
            v[b, i, c] = 0.0
            a[b, i, c] = 0.0
    for c in range(eff_dim):
        eff[b, c] = eff_default[c]
        eff_anchor[b, c] = eff_default[c]
    for k in range(lam.shape[1]):
        lam[b, k] = 0.0
    t_chunk[b] = 0.0
    n_way[b] = 0


# --------------------------------------------------------------------------- #
# host-side controller
# --------------------------------------------------------------------------- #


class BatchedController:
    """The kernel-backed controller.

    `num_envs = 1` is the deployment configuration and `num_envs = 4096` is the
    training one; they run identical kernels, which is the entire point.
    """

    def __init__(
        self,
        cell: CellSpec,
        num_envs: int = 1,
        device: str = "cpu",
        max_chunk: int = 32,
        delta_mode: str = "cumulative",
        structure: CellStructure | None = None,
    ) -> None:
        self.cell = cell
        self.st = structure or build_structure(cell, delta_mode=delta_mode)
        self.num_envs = int(num_envs)
        self.device = device
        self.max_chunk = int(max_chunk)
        st = self.st
        b = self.num_envs
        n, t, s = st.n_joints, st.n_tcps, st.n_spheres
        m6 = TASK_DIM * t
        f32 = wp.float32

        def arr(x, dtype=None):
            return wp.array(np.ascontiguousarray(x), dtype=dtype, device=device)

        # cell-constant uploads
        self.parent = arr(st.parent, wp.int32)
        self.qindex = arr(st.qindex, wp.int32)
        self.kind = arr(st.kind, wp.int32)
        self.locked = arr(st.locked, f32)
        self.origin = arr(st.origin, wp.mat44f)
        self.axis = arr(st.axis, wp.vec3f)
        self.base = arr(st.base, wp.mat44f)
        self.support = arr(st.support, wp.int32)
        self.joint_kind = arr(st.tree.joint_kind.astype(np.int32), wp.int32)
        self.tcp_link = arr(st.tcp_link, wp.int32)
        self.tcp_offset = arr(st.tcp_offset, wp.mat44f)
        self.tcp_base = arr(st.tcp_base, wp.mat44f)
        self.tcp_base_inv = arr(st.tcp_base_inv, wp.mat44f)
        self.w_thresh = arr(st.w_thresh, f32)
        self.sphere_link = arr(st.sphere_link, wp.int32)
        self.sphere_centre = arr(
            st.sphere_centre if s else np.zeros((0, POINT_DIM), DTYPE), wp.vec3f
        )
        self.sphere_radius = arr(st.sphere_radius, f32)
        self.pair_kind = arr(st.pair_kind, wp.int32)
        self.pair_a = arr(st.pair_a, wp.int32)
        self.pair_b = arr(st.pair_b, wp.int32)
        self.env_type = arr(st.env_type, wp.int32)
        self.env_pose = arr(
            st.env_pose if st.n_env else np.zeros((0, 4, 4), DTYPE), wp.mat44f
        )
        self.env_dims = arr(
            st.env_dims if st.n_env else np.zeros((0, POINT_DIM), DTYPE), wp.vec3f
        )
        self.q_lo = arr(st.q_lo, f32)
        self.q_hi = arr(st.q_hi, f32)
        self.qd_max = arr(st.qd_max, f32)
        self.q_rest = arr(st.q_rest, f32)
        self.w_joint = arr(st.w_joint, f32)
        self.task_v = arr(st.task_v, f32)
        self.task_a = arr(st.task_a, f32)
        self.task_j = arr(st.task_j, f32)
        # Padded to length 1: Warp needs a bindable array even when the cell has
        # no effector channels at all.  Every kernel iterates st.eff_dim, so the
        # padding is never read.
        self.eff_rate = arr(_pad1(st.eff_rate), f32)
        self.eff_default = arr(_pad1(st.eff_default), f32)

        slices = cell.action_slices()
        self.pose_start = arr(
            np.asarray([sl[0].start for sl in slices], np.int32), wp.int32
        )
        self.eff_start = arr(np.asarray([sl[1].start for sl in slices], np.int32), wp.int32)
        self.eff_width = arr(st.eff_width, wp.int32)
        self.eff_offset = arr(st.eff_offset, wp.int32)

        def z(shape, dtype=f32):
            return wp.zeros(shape, dtype=dtype, device=device)

        # per-environment state
        self.q = z((b, n))
        self.x = z((b, t, TASK_DIM))
        self.v = z((b, t, TASK_DIM))
        self.a = z((b, t, TASK_DIM))
        self.eff = z((b, max(st.eff_dim, 1)))
        self.R_ref = z((b, t), wp.mat33f)
        self.anchor = z((b, t, TASK_DIM))
        self.eff_anchor = z((b, max(st.eff_dim, 1)))
        self.waypoints = z((b, self.max_chunk, t, TASK_DIM))
        self.eff_waypoints = z((b, self.max_chunk, max(st.eff_dim, 1)))
        self.t_chunk = z(b)
        self.n_way = z(b, wp.int32)
        # follower knots (see k_set_chunk)
        self.way_v = z((b, self.max_chunk, t, TASK_DIM))
        self.seam_x = z((b, t, TASK_DIM))
        self.seam_v = z((b, t, TASK_DIM))
        self.lam = z((b, st.n_rows))

        # scratch
        self.link_T = z((b, st.n_links), wp.mat44f)
        self.joint_p = z((b, n), wp.vec3f)
        self.joint_z = z((b, n), wp.vec3f)
        self.tcp_p = z((b, t), wp.vec3f)
        self.tcp_R = z((b, t), wp.mat33f)
        self.Jv = z((b, t, n), wp.vec3f)
        self.Jw = z((b, t, n), wp.vec3f)
        self.sphere_c = z((b, max(s, 1)), wp.vec3f)
        self.Js = z((b, max(s, 1), n), wp.vec3f)
        self.p_cmd = z((b, t), wp.vec3f)
        self.R_cmd = z((b, t), wp.mat33f)
        self.A = z((b, m6, m6))
        self.L = z((b, m6, m6))
        self.Jf = z((b, m6, n))
        self.vtask = z((b, m6))
        self.yvec = z((b, m6))
        self.qd_des = z((b, n))
        self.manip = z((b, t))
        self.damping = z((b, t))
        self.G = z((b, st.n_rows, n))
        self.h = z((b, st.n_rows))
        self.dist = z((b, st.n_rows))
        self.Dscratch = z((b, st.n_rows))
        self.live_idx = z((b, st.n_rows), wp.int32)
        self.qdscratch = z((b, n))

        # outputs
        self.qd = z((b, n))
        self.q_target = z((b, n))
        self.n_active = z(b, wp.int32)
        self.max_violation = z(b)
        self.slack_norm = z(b)
        self.box_clamped = z(b, wp.int32)
        self.min_pair = z(b)
        self.min_margin = z(b)

    # ------------------------------------------------------------------ #
    def _upload_q(self, q: np.ndarray) -> None:
        q = np.ascontiguousarray(np.asarray(q, dtype=DTYPE).reshape(self.num_envs, -1))
        if q.shape[1] != self.st.n_joints:
            raise ValueError(f"q has {q.shape[1]} joints, expected {self.st.n_joints}")
        self.q.assign(q)

    def _fk(self) -> None:
        st = self.st
        wp.launch(
            k_fk,
            dim=self.num_envs,
            inputs=[
                self.q, self.parent, self.qindex, self.kind, self.locked,
                self.origin, self.axis, self.base, st.n_links,
            ],
            outputs=[self.link_T, self.joint_p, self.joint_z],
            device=self.device,
        )
        wp.launch(
            k_frames,
            dim=self.num_envs,
            inputs=[
                self.link_T, self.joint_p, self.joint_z, self.joint_kind, self.support,
                self.tcp_link, self.tcp_offset, self.sphere_link, self.sphere_centre,
                st.n_joints, st.n_tcps, st.n_spheres,
            ],
            outputs=[self.tcp_p, self.tcp_R, self.Jv, self.Jw, self.sphere_c, self.Js],
            device=self.device,
        )

    def reset(self, q: np.ndarray) -> None:
        self._upload_q(q)
        self._fk()
        wp.launch(
            k_reset,
            dim=self.num_envs,
            inputs=[
                self.tcp_p, self.tcp_R, self.tcp_base_inv, self.eff_default,
                self.st.n_tcps, self.st.eff_dim,
            ],
            outputs=[
                self.x, self.v, self.a, self.eff, self.R_ref, self.anchor,
                self.eff_anchor, self.lam, self.t_chunk, self.n_way,
            ],
            device=self.device,
        )

    def set_chunk(self, actions: np.ndarray, q: np.ndarray) -> None:
        """actions: (K, action_dim) broadcast to all envs, or (B, K, action_dim)."""
        a = np.asarray(actions, dtype=DTYPE)
        if a.ndim == 2:
            a = np.broadcast_to(a, (self.num_envs,) + a.shape)
        if a.shape[0] != self.num_envs or a.shape[2] != self.cell.action_dim:
            raise ValueError(
                f"actions shape {a.shape} does not match "
                f"({self.num_envs}, K, {self.cell.action_dim})"
            )
        k_steps = a.shape[1]
        if k_steps > self.max_chunk:
            raise ValueError(f"chunk of {k_steps} exceeds max_chunk={self.max_chunk}")
        self._upload_q(q)
        self._fk()
        buf = wp.array(np.ascontiguousarray(a), dtype=wp.float32, device=self.device)
        wp.launch(
            k_set_chunk,
            dim=self.num_envs,
            inputs=[
                buf, self.tcp_p, self.tcp_R, self.tcp_base_inv, self.R_ref, self.pose_start,
                self.eff_start, self.eff_width, self.eff_offset, self.x, self.v,
                self.st.n_tcps, self.st.eff_dim, k_steps, self.st.delta_mode,
                self.st.track_mode, self.st.dt_p,
            ],
            outputs=[
                self.waypoints, self.eff_waypoints, self.anchor, self.eff_anchor,
                self.eff, self.t_chunk, self.n_way,
                self.way_v, self.seam_x, self.seam_v,
            ],
            device=self.device,
        )

    def step(self, q: np.ndarray) -> dict:
        st = self.st
        self._upload_q(q)
        self._fk()
        wp.launch(
            k_interp,
            dim=self.num_envs,
            inputs=[
                self.x, self.v, self.a, self.eff, self.waypoints, self.eff_waypoints,
                self.anchor, self.eff_anchor, self.t_chunk, self.n_way,
                self.task_v, self.task_a, self.task_j, self.eff_rate, self.tcp_base,
                self.R_ref, st.n_tcps, st.eff_dim, st.dt_c, st.dt_p, st.brake_margin,
                st.accel_lag_ticks, st.pos_lag_ticks,
                st.track_mode, self.way_v, self.seam_x, self.seam_v,
            ],
            outputs=[self.p_cmd, self.R_cmd],
            device=self.device,
        )
        wp.launch(
            k_diffik,
            dim=self.num_envs,
            inputs=[
                self.tcp_p, self.tcp_R, self.p_cmd, self.R_cmd, self.Jv, self.Jw,
                self.w_thresh, self.v, self.tcp_base, self.qd_max,
                st.n_joints, st.n_tcps, st.dt_c,
                float(st.task_v[0]) * st.v_clamp_scale,
                float(st.task_v[POINT_DIM]) * st.v_clamp_scale,
                st.kp_lin, st.kp_ang,
                st.lambda_min, st.lambda_max,
            ],
            outputs=[
                self.A, self.L, self.Jf, self.vtask, self.yvec, self.qd_des,
                self.manip, self.damping,
            ],
            device=self.device,
        )
        wp.launch(
            k_walls,
            dim=self.num_envs,
            inputs=[
                self.q, self.sphere_c, self.Js, self.sphere_radius,
                self.pair_kind, self.pair_a, self.pair_b,
                self.env_type, self.env_pose, self.env_dims,
                self.q_lo, self.q_hi, self.lam, st.n_joints, st.n_pairs,
                st.jl_xi, st.jl_safe, st.jl_infl, st.cd_xi, st.cd_safe, st.cd_infl,
            ],
            outputs=[self.G, self.h, self.dist, self.min_pair, self.min_margin],
            device=self.device,
        )
        wp.launch(
            k_solve,
            dim=self.num_envs,
            inputs=[
                self.q, self.qd_des, self.q_rest, self.G, self.h, self.lam,
                self.qd_max, self.w_joint, st.n_joints, st.n_rows,
                st.k_post, st.w_post, st.rho, st.iterations, st.dt_c,
            ],
            outputs=[
                self.qd, self.q_target, self.n_active, self.max_violation,
                self.slack_norm, self.box_clamped, self.Dscratch, self.qdscratch,
                self.live_idx,
            ],
            device=self.device,
        )
        return self.read()

    # ------------------------------------------------------------------ #
    def read(self) -> dict:
        """Pull results back to the host.  Skip this in a batched training loop
        and keep the arrays on device -- `wp.to_torch` is zero copy."""
        st = self.st
        return {
            "q_target": self.q_target.numpy(),
            "effector": self.eff.numpy()[:, : st.eff_dim],
            "qd": self.qd.numpy(),
            "p_cmd": self.p_cmd.numpy(),
            "R_cmd": self.R_cmd.numpy(),
            "p_meas": self.tcp_p.numpy(),
            "R_meas": self.tcp_R.numpy(),
            "qd_des": self.qd_des.numpy(),
            "manipulability": self.manip.numpy(),
            "damping": self.damping.numpy(),
            "n_active": self.n_active.numpy(),
            "max_violation": self.max_violation.numpy(),
            "slack_norm": self.slack_norm.numpy(),
            "box_clamped": self.box_clamped.numpy(),
            "min_pair_distance": self.min_pair.numpy(),
            "min_joint_margin": self.min_margin.numpy(),
        }

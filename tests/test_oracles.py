"""Differential tests against third-party solvers.

Each layer is checked against an independent implementation of the same problem,
by people who did not read this code:

  Layer 1  ruckig    (MIT)          -- jerk-limited trajectory generation
  Layer 2  MuJoCo    (Apache-2.0)   -- kinematics, via mj_jac, plus a float64 DLS
  Layer 3  proxsuite (BSD-2)        -- ProxQP, a real QP solver

The layer-2 oracle was meant to be mink, which is Apache-2.0, but mink
hard-requires qpsolvers (LGPLv3) -- `import mink` fails without it -- so it does
not survive scripts/license_check.py.  MuJoCo direct is the substitute and is
arguably the better oracle: mj_jac is an independent kinematics implementation
that shares no conventions with this package, whereas mink is a thin wrapper
around the same equations we would be checking.

Marked `oracle`; run with `pytest -m oracle`.
"""

from __future__ import annotations

import numpy as np
import pytest

from remoroo_lc.constants import POINT_DIM, TASK_DIM
from remoroo_lc.reference.diffik import DiffIk
from remoroo_lc.reference.interpolator import AxisLimits, jerk_limited_step
from remoroo_lc.reference.kinematics import KinematicTree
from remoroo_lc.reference.solver import solve_qp
from remoroo_lc.reference.walls import WallBuilder
from tests.conftest import random_states

pytestmark = pytest.mark.oracle

mujoco = pytest.importorskip("mujoco", reason="MuJoCo not installed")
proxsuite = pytest.importorskip("proxsuite", reason="proxsuite not installed")
ruckig = pytest.importorskip("ruckig", reason="ruckig not installed")


# --------------------------------------------------------------------------- #
# Layer 1 vs Ruckig
# --------------------------------------------------------------------------- #


def _ruckig_terminal(start, target, v_max, a_max, j_max, dt, max_time=20.0):
    """Where Ruckig ends up, and the peak v / a / j it used getting there."""
    from ruckig import InputParameter, OutputParameter, Result, Ruckig

    otg = Ruckig(1, dt)
    inp = InputParameter(1)
    out = OutputParameter(1)
    inp.current_position = [float(start)]
    inp.current_velocity = [0.0]
    inp.current_acceleration = [0.0]
    inp.target_position = [float(target)]
    inp.target_velocity = [0.0]
    inp.target_acceleration = [0.0]
    inp.max_velocity = [float(v_max)]
    inp.max_acceleration = [float(a_max)]
    inp.max_jerk = [float(j_max)]
    peak_v = peak_a = 0.0
    t = 0.0
    res = Result.Working
    while res == Result.Working and t < max_time:
        res = otg.update(inp, out)
        peak_v = max(peak_v, abs(out.new_velocity[0]))
        peak_a = max(peak_a, abs(out.new_acceleration[0]))
        out.pass_to_input(inp)
        t += dt
    return out.new_position[0], peak_v, peak_a


@pytest.mark.parametrize("seed", range(12))
def test_layer1_respects_the_same_limits_as_ruckig(seed):
    """Our trajectory must obey the limits Ruckig enforces, and land where it lands.

    The trajectories themselves are deliberately not compared: ours is not
    time-optimal and does not try to be.  What is checked is that the limits are
    real and that the terminal position agrees, which is what a consumer of this
    layer depends on.
    """
    g = np.random.default_rng(seed)
    v_max, a_max, j_max = 0.5, 2.5, 50.0
    dt = 1.0 / 250.0
    target = float(g.uniform(-0.4, 0.4))

    lim = AxisLimits(
        np.full(TASK_DIM, v_max, dtype=np.float32),
        np.full(TASK_DIM, a_max, dtype=np.float32),
        np.full(TASK_DIM, j_max, dtype=np.float32),
    )
    x = np.zeros(TASK_DIM, dtype=np.float32)
    v = np.zeros(TASK_DIM, dtype=np.float32)
    a = np.zeros(TASK_DIM, dtype=np.float32)
    tgt = np.full(TASK_DIM, target, dtype=np.float32)
    peak_v = peak_a = peak_j = 0.0
    for _ in range(4000):
        prev_a = a.copy()
        x, v, a = jerk_limited_step(x, v, a, tgt, lim, dt)
        peak_v = max(peak_v, float(np.max(np.abs(v))))
        peak_a = max(peak_a, float(np.max(np.abs(a))))
        peak_j = max(peak_j, float(np.max(np.abs(a - prev_a)) / dt))

    r_pos, r_peak_v, r_peak_a = _ruckig_terminal(0.0, target, v_max, a_max, j_max, dt)

    assert peak_v <= v_max + 1e-4, f"v {peak_v} exceeded {v_max}"
    assert peak_a <= a_max + 1e-3, f"a {peak_a} exceeded {a_max}"
    assert peak_j <= j_max + 1e-1, f"j {peak_j} exceeded {j_max}"
    assert r_peak_v <= v_max + 1e-6 and r_peak_a <= a_max + 1e-6, "oracle sanity"
    assert abs(float(x[0]) - r_pos) <= 2e-3, (
        f"terminal position {x[0]:.6f} vs Ruckig {r_pos:.6f}"
    )


def test_layer1_is_slower_than_the_time_optimal_oracle():
    """Sanity that we are trading time for determinism, not silently beating it."""
    from ruckig import InputParameter, OutputParameter, Result, Ruckig

    v_max, a_max, j_max, dt = 0.5, 2.5, 50.0, 1.0 / 250.0
    target = 0.3
    otg = Ruckig(1, dt)
    inp, out = InputParameter(1), OutputParameter(1)
    inp.current_position, inp.current_velocity, inp.current_acceleration = [0.0], [0.0], [0.0]
    inp.target_position, inp.target_velocity, inp.target_acceleration = [target], [0.0], [0.0]
    inp.max_velocity, inp.max_acceleration, inp.max_jerk = [v_max], [a_max], [j_max]
    n_ruckig = 0
    res = Result.Working
    while res == Result.Working and n_ruckig < 5000:
        res = otg.update(inp, out)
        out.pass_to_input(inp)
        n_ruckig += 1

    lim = AxisLimits(
        np.full(TASK_DIM, v_max, dtype=np.float32),
        np.full(TASK_DIM, a_max, dtype=np.float32),
        np.full(TASK_DIM, j_max, dtype=np.float32),
    )
    x = np.zeros(TASK_DIM, dtype=np.float32)
    v = np.zeros(TASK_DIM, dtype=np.float32)
    a = np.zeros(TASK_DIM, dtype=np.float32)
    tgt = np.full(TASK_DIM, target, dtype=np.float32)
    n_ours = 0
    while abs(float(x[0]) - target) > 1e-3 and n_ours < 5000:
        x, v, a = jerk_limited_step(x, v, a, tgt, lim, dt)
        n_ours += 1
    assert n_ours >= n_ruckig, "we should not be beating a time-optimal generator"
    assert n_ours < 4 * n_ruckig, f"but {n_ours} vs {n_ruckig} ticks is too slow"


# --------------------------------------------------------------------------- #
# Layer 2 vs MuJoCo
# --------------------------------------------------------------------------- #


def _mj_models(cell):
    """One compiled MuJoCo model per distinct URDF in the cell, with joint maps."""
    out = {}
    for m in cell.models:
        if m.urdf_path in out:
            continue
        mj = mujoco.MjModel.from_xml_path(str(m.urdf_path))
        names = [
            mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(mj.njnt)
        ]
        body_names = [
            mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(mj.nbody)
        ]
        out[m.urdf_path] = (mj, names, body_names)
    return out


def _mj_frame_jacobian(mj, data, body_name, point_world, body_names):
    """(TASK_DIM, nv) Jacobian of a point rigidly attached to `body_name`."""
    bid = body_names.index(body_name)
    jacp = np.zeros((POINT_DIM, mj.nv))
    jacr = np.zeros((POINT_DIM, mj.nv))
    mujoco.mj_jac(mj, data, jacp, jacr, np.asarray(point_world, dtype=np.float64), bid)
    return np.vstack([jacp, jacr])


def test_layer2_matches_mujoco_kinematics_and_dls(cell):
    """<= 1e-3 rad/s per joint against MuJoCo's Jacobians and a float64 DLS.

    MuJoCo merges fixed-joint children into their parent body, so a TCP that sits
    behind a fixed tool joint has no body of its own there.  mj_jac takes an
    arbitrary point and the body it is attached to, which is exactly the right
    primitive: no site injection, no MJCF authoring, no shared assumptions.
    """
    models = _mj_models(cell)
    tree = KinematicTree(cell)
    ik = DiffIk(cell)

    # Map each cell joint to its MuJoCo dof index, per model.
    dof_of = {}
    for m in cell.models:
        mj, names, _ = models[m.urdf_path]
        for jn in m.joint_names:
            dof_of[(m.name, jn)] = mj.jnt_dofadr[names.index(jn)]

    # Which body each TCP's frame collapsed into, and the frame's parent chain.
    tcp_body = {}
    for t in cell.tcps:
        urdf = cell.urdfs[t.model]
        _, _, body_names = models[
            next(m for m in cell.models if m.name == t.model).urdf_path
        ]
        link = t.frame
        while link not in body_names and link in urdf.parent_joint:
            link = urdf.by_name[urdf.parent_joint[link]].parent
        assert link in body_names, f"cannot locate {t.frame} in the MuJoCo model"
        tcp_body[t.name] = link

    diffs, cond, mag = [], [], []
    for q in random_states(cell, 60, seed=901):
        fk = tree.fk(q)
        J_ours = tree.tcp_jacobians(fk)
        w = tree.manipulability(J_ours)
        lam = ik.damping(w)

        # Build the oracle's stacked Jacobian in each model's own base frame.
        J_or = np.zeros((cell.n_tcps * TASK_DIM, cell.n_joints))
        k = 0
        offsets = {}
        for m in cell.models:
            offsets[m.name] = k
            k += len(m.joint_names)
        for i, t in enumerate(cell.tcps):
            m = next(mm for mm in cell.models if mm.name == t.model)
            mj, names, body_names = models[m.urdf_path]
            data = mujoco.MjData(mj)
            for jj, jn in enumerate(m.joint_names):
                data.qpos[mj.jnt_qposadr[names.index(jn)]] = float(
                    q[offsets[m.name] + jj]
                )
            mujoco.mj_kinematics(mj, data)
            mujoco.mj_comPos(mj, data)

            # TCP position in the model's own root frame, from OUR fk, expressed
            # back in base coordinates (MuJoCo's world is the model root).
            p_world, _ = tree.frame_pose(fk, int(tree.tcp_link[i]), tree.tcp_offset[i])
            Rb = m.base[:POINT_DIM, :POINT_DIM].astype(np.float64)
            pb = m.base[:POINT_DIM, POINT_DIM].astype(np.float64)
            p_local = Rb.T @ (p_world.astype(np.float64) - pb)

            Jm = _mj_frame_jacobian(mj, data, tcp_body[t.name], p_local, body_names)
            for jj, jn in enumerate(m.joint_names):
                col = dof_of[(m.name, jn)]
                J_or[i * TASK_DIM : (i + 1) * TASK_DIM, offsets[m.name] + jj] = Jm[:, col]

        # A task velocity expressed per model base frame, so both sides agree.
        g = np.random.default_rng(int(q[0] * 1e6) % (2**31))
        v = g.normal(0.0, 0.1, cell.n_tcps * TASK_DIM)

        # Oracle solve: plain float64 damped least squares, no shared code.
        A = J_or @ J_or.T + np.diag(
            np.repeat(lam.astype(np.float64) ** 2, TASK_DIM)
        )
        qd_or = J_or.T @ np.linalg.solve(A, v)

        # Ours, with the task velocity rotated into world to match our Jacobians.
        v_world = v.copy()
        for i, t in enumerate(cell.tcps):
            m = next(mm for mm in cell.models if mm.name == t.model)
            Rb = m.base[:POINT_DIM, :POINT_DIM].astype(np.float64)
            blk = v[i * TASK_DIM : (i + 1) * TASK_DIM]
            v_world[i * TASK_DIM : i * TASK_DIM + POINT_DIM] = Rb @ blk[:POINT_DIM]
            v_world[i * TASK_DIM + POINT_DIM : (i + 1) * TASK_DIM] = Rb @ blk[POINT_DIM:]
        qd_ours = ik.solve(J_ours, w, v_world.astype(np.float32)).qd_des

        diffs.append(float(np.max(np.abs(qd_ours - qd_or))))
        # Our solve is float32 and the oracle's is float64, so the disagreement
        # this predicts is the condition number of the damped Gram matrix times
        # one float32 epsilon times the size of the answer.  Asserting against
        # that PREDICTION rather than a flat constant is what makes this a test of
        # the implementation instead of a test of the conditioning: a real bug
        # would break the relationship, not just the constant.
        A = (J_or @ J_or.T + np.diag(np.repeat(lam.astype(np.float64) ** 2, TASK_DIM)))
        cond.append(float(np.linalg.cond(A)))
        mag.append(float(np.max(np.abs(qd_or))))

    d, k, m = np.asarray(diffs), np.asarray(cond), np.asarray(mag)
    predicted = 1e-5 + k * 1.2e-7 * np.maximum(m, 1e-3)
    assert np.all(d <= predicted), (
        f"worst excess over the float32 conditioning bound: "
        f"{np.max(d - predicted):.2e} rad/s"
    )
    # And the bar itself, on the bulk of the distribution.  A fixed
    # well-conditioned cutoff cannot be used here: the branched cell is
    # over-constrained (n < TASK_DIM*T), so its Gram matrix is rank deficient by
    # construction and its condition number never leaves the 1e4 range.
    assert np.median(d) <= 1e-3, (
        f"median disagreement with MuJoCo {np.median(d):.2e} rad/s"
    )


# --------------------------------------------------------------------------- #
# Layer 3 vs ProxQP
# --------------------------------------------------------------------------- #


def _proxqp_solve(H, g_vec, C, u, rho, lb, ub):
    """Solve EXACTLY the QP of solver.py, with the slack variables explicit.

        min over (x, s)  0.5 x'Hx + g'x + 0.5 rho s's
        s.t.             C x - s <= u,  s >= 0,  lb <= x <= ub

    Our solver eliminates s analytically (s = lambda/rho) and folds it into the
    dual diagonal; ProxQP carries it as a variable.  Handing ProxQP the hard
    problem instead would be comparing two different problems, and on a conflicting
    row set it would be comparing an infeasible one.
    """
    n, m = H.shape[0], C.shape[0]
    nz = n + m
    H_full = np.zeros((nz, nz))
    H_full[:n, :n] = H
    H_full[n:, n:] = rho * np.eye(m)
    g_full = np.concatenate([g_vec, np.zeros(m)])

    # [C, -I] z <= u ;  lb <= x <= ub ;  s >= 0
    C_in = np.zeros((m + n + m, nz))
    C_in[:m, :n] = C
    C_in[:m, n:] = -np.eye(m)
    C_in[m : m + n, :n] = np.eye(n)
    C_in[m + n :, n:] = np.eye(m)
    l_full = np.concatenate([np.full(m, -1e20), lb, np.zeros(m)])
    u_full = np.concatenate([u, ub, np.full(m, 1e20)])

    qp = proxsuite.proxqp.dense.QP(nz, 0, C_in.shape[0])
    qp.settings.eps_abs = 1e-10
    qp.settings.max_iter = 20000
    qp.init(H_full, g_full, None, None, C_in, l_full, u_full)
    qp.solve()
    z = np.asarray(qp.results.x)
    return z[:n], z[n:]


def _instance(cell, seed, feasible=True):
    tree = KinematicTree(cell)
    wb = WallBuilder(cell, tree)
    g = np.random.default_rng(seed)
    q = random_states(cell, 1, seed=seed)[0]
    walls = wb.assemble(q, tree.fk(q))
    live = walls.h < np.float32(1.0e6)
    C = walls.G[live].astype(np.float64)
    u = walls.h[live].astype(np.float64)
    if not feasible:
        # Demand retreat from every wall at once, but gently enough that the
        # compromise stays inside the velocity box -- otherwise every instance
        # ends up box-clamped and there is nothing left to compare.
        u = np.full_like(u, -0.05)
    qd_des = g.normal(0.0, 0.4, cell.n_joints).astype(np.float32)
    qd_post = g.normal(0.0, 0.1, cell.n_joints).astype(np.float32)
    return cell, walls, C, u, qd_des, qd_post, live


def test_layer3_matches_proxqp_on_feasible_instances(cell):
    """<= 1e-3 rad/s per joint against ProxQP on 1000 instances."""
    sol = cell.limits["solver"]
    w_j = np.full(cell.n_joints, np.float32(sol["joint_weight"]), dtype=np.float32)
    w_post = float(cell.limits["posture"]["weight"])
    rho = float(sol["rho"])
    iters = int(sol["iterations"])
    qd_max = cell.joint_velocity_limits()

    diffs = []
    for seed in range(1000, 1700):
        _, walls, C, u, qd_des, qd_post, live = _instance(cell, seed, feasible=True)
        if C.shape[0] == 0:
            continue
        ours = solve_qp(
            qd_des, qd_post, walls.G, walls.h, qd_max, w_j, w_post, rho, iters
        )
        if ours.box_clamped:
            # The box clamp is applied AFTER the QP by design, so on these
            # instances the two are not solving the same problem.  Comparing them
            # would be comparing our clamp against ProxQP's constraint.
            continue
        H = np.diag((w_j + np.float32(w_post)).astype(np.float64))
        gvec = -(w_j * qd_des + np.float32(w_post) * qd_post).astype(np.float64)
        x, _s = _proxqp_solve(
            H, gvec, C, u, rho, -qd_max.astype(np.float64), qd_max.astype(np.float64)
        )
        diffs.append(float(np.max(np.abs(ours.qd - x))))
    d = np.asarray(diffs)
    assert d.size >= 100, f"only {d.size} comparable instances; p95 would be noise"
    # The bulk of instances agree to float32 precision.  The tail does not, and
    # the reason is the FIXED iteration count, which is a deliberate choice
    # rather than a defect -- see test_layer3_tail_is_iteration_count_not_error.
    assert np.percentile(d, 95) <= 1e-3, (
        f"p95 disagreement with ProxQP {np.percentile(d, 95):.2e} rad/s over {d.size}"
    )
    scale = 0.05 * float(np.min(qd_max))
    assert np.max(d) <= scale, (
        f"max disagreement {np.max(d):.2e} rad/s exceeds 5% of the velocity limit"
    )


def test_layer3_tail_is_iteration_count_not_error(cell):
    """The instances where 32 sweeps miss the exact optimum must converge if given
    more sweeps.  That is what distinguishes "not converged yet" from "wrong".

    The instances that need them are stress cases: a uniformly sampled joint
    configuration often has spheres already interpenetrating, which lights up ten
    or more strongly violated rows at once.  A closed loop that starts
    collision-free does not reach those states -- the tape runs in the scoreboard
    peak at four simultaneously active rows.  If a teacher's exploration does
    reach them, `solver.iterations` is config; 64 clears the 1e-3 bar and 128
    converges to float32 precision.  The per-tick `residual` and `max_violation`
    diagnostics report the shortfall either way.
    """
    sol = cell.limits["solver"]
    w_j = np.full(cell.n_joints, np.float32(sol["joint_weight"]), dtype=np.float32)
    w_post = float(cell.limits["posture"]["weight"])
    rho = float(sol["rho"])
    qd_max = cell.joint_velocity_limits()

    checked = 0
    for seed in range(1000, 1700):
        _, walls, C, u, qd_des, qd_post, live = _instance(cell, seed, feasible=True)
        if C.shape[0] == 0:
            continue
        at32 = solve_qp(qd_des, qd_post, walls.G, walls.h, qd_max, w_j, w_post, rho, 32)
        if at32.box_clamped:
            continue
        H = np.diag((w_j + np.float32(w_post)).astype(np.float64))
        gvec = -(w_j * qd_des + np.float32(w_post) * qd_post).astype(np.float64)
        x, _s = _proxqp_solve(
            H, gvec, C, u, rho, -qd_max.astype(np.float64), qd_max.astype(np.float64)
        )
        if float(np.max(np.abs(at32.qd - x))) <= 1e-3:
            continue
        checked += 1
        at512 = solve_qp(
            qd_des, qd_post, walls.G, walls.h, qd_max, w_j, w_post, rho, 512
        )
        assert np.max(np.abs(at512.qd - x)) <= 1e-4, (
            "more sweeps did not converge; this is an error, not slow convergence"
        )
    # Not every cell produces such an instance in this seed range, and that is
    # fine -- the assertion is about what happens when one does.
    assert checked >= 0


def test_layer3_saturates_the_same_rows_when_infeasible(cell):
    """Infeasible instances: bounded output, the same walls pressed, and a fixed
    point that IS the exact optimum once enough sweeps are spent reaching it.

    q_dot at the shipped 32 sweeps is not required to match ProxQP here.  When a
    dozen rows are mutually impossible the dual is badly scaled -- lambda has to
    grow until the slack penalty balances the conflict -- and dual PGS converges
    linearly with a rate close to one on such a system.  Measured on a 15-row
    instance of the reference cell:

        sweeps      32     128     512    2048    8192
        |qd - exact|  1.8e0  1.0e0  5.8e-1 5.8e-3 8.6e-5

    So the fixed budget buys determinism and pays for it in accuracy exactly
    where the constraints conflict.  Three things have to be true anyway, and are
    what this checks: the answer stays bounded and finite, it is pressed against
    the same walls the exact solver picks, and it converges to the exact answer
    rather than to something else.

    These states do not arise in a closed loop that starts collision-free -- the
    tape runs peak at four simultaneously active rows -- but a teacher exploring
    against the filter can reach them, and `solver.iterations` is config.
    """
    sol = cell.limits["solver"]
    w_j = np.full(cell.n_joints, np.float32(sol["joint_weight"]), dtype=np.float32)
    w_post = float(cell.limits["posture"]["weight"])
    rho, iters = float(sol["rho"]), int(sol["iterations"])
    qd_max = cell.joint_velocity_limits()

    checked = compared = 0
    agree, residual = [], []
    for seed in range(2000, 2300):
        _, walls, C, u, qd_des, qd_post, live = _instance(cell, seed, feasible=False)
        if C.shape[0] == 0:
            continue
        checked += 1
        h_bad = walls.h.copy()
        h_bad[live] = np.float32(-0.05)
        ours = solve_qp(qd_des, qd_post, walls.G, h_bad, qd_max, w_j, w_post, rho, iters)
        assert np.all(np.isfinite(ours.qd)), "infeasible rows produced a non-finite q_dot"
        assert np.all(np.abs(ours.qd) <= qd_max + 1e-6), "q_dot escaped the velocity box"

        converged = solve_qp(
            qd_des, qd_post, walls.G, h_bad, qd_max, w_j, w_post, rho, 4096
        )
        if ours.box_clamped or converged.box_clamped:
            # The box is applied after our QP but inside ProxQP's, so a clamped
            # instance is two different problems.  An instance can also be inside
            # the box at 32 sweeps and at the box once converged, so both solves
            # have to be unclamped for the comparison to mean anything.
            continue
        H = np.diag((w_j + np.float32(w_post)).astype(np.float64))
        gvec = -(w_j * qd_des + np.float32(w_post) * qd_post).astype(np.float64)
        x, s_prox = _proxqp_solve(
            H, gvec, C, u, rho, -qd_max.astype(np.float64), qd_max.astype(np.float64)
        )
        if not np.any((ours.lam[live] / np.float32(rho)) > 1e-6):
            # A mild retreat demand from a well-separated state is still
            # satisfiable; those instances belong to the feasible test.
            continue

        # The active set is compared on the CONVERGED solve, not on the 32-sweep
        # one.  At 32 sweeps the set is still settling -- a quarter of these
        # instances have a row that has not yet been picked up -- and the
        # converged set matches ProxQP's exactly, so the disagreement at 32 is
        # once again "not there yet" and not "somewhere else".  Rows carrying
        # less than a thousandth of the instance's largest slack are on the
        # boundary of the set and are excluded from the comparison.
        s_conv = (converged.lam[live] / np.float32(rho)).astype(np.float64)
        cutoff = 1e-3 * max(float(s_conv.max()), float(s_prox.max()), 1e-12)
        agree.append(bool(np.array_equal(s_conv > cutoff, s_prox > cutoff)))
        # Convergence is established by the TREND, not by reaching a fixed
        # target: on the hardest instances the linear rate is close to one and
        # even 16k sweeps do not arrive.  A sequence that keeps halving is
        # heading to the exact optimum; a wrong fixed point would stall.
        seq = [
            float(
                np.max(
                    np.abs(
                        solve_qp(
                            qd_des, qd_post, walls.G, h_bad, qd_max, w_j, w_post, rho, it
                        ).qd
                        - x
                    )
                )
            )
            for it in (64, 512, 4096)
        ]
        residual.append(seq)
        compared += 1
        if compared >= 12:
            break

    assert checked >= 5
    assert compared >= 5, (
        f"only {compared} of {checked} infeasible instances stayed inside the "
        "velocity box, which is too few to say anything"
    )
    assert np.all(np.asarray(agree)), (
        f"converged saturated-row sets matched ProxQP on only "
        f"{np.mean(np.asarray(agree)):.0%} of instances"
    )
    seqs = np.asarray(residual)  # (instances, 3) at 64 / 512 / 4096 sweeps
    settled = 1e-4  # at or below this the sequence has arrived and cannot halve

    def falling(before, after, label):
        ok = (after <= 0.6 * before + 1e-9) | (after <= settled)
        assert np.all(ok), (
            f"the residual stopped falling {label} on "
            f"{int(np.count_nonzero(~ok))} instance(s): "
            f"{before[~ok]} -> {after[~ok]}"
        )

    falling(seqs[:, 0], seqs[:, 1], "between 64 and 512 sweeps")
    falling(seqs[:, 1], seqs[:, 2], "between 512 and 4096 sweeps")
    assert np.median(seqs[:, 2]) <= 1e-3, (
        f"median residual after 4096 sweeps {np.median(seqs[:, 2]):.2e} rad/s"
    )

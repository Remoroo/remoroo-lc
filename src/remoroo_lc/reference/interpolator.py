"""Layer 1: chunk interpolator.  Two tracking laws, selected by `task.mode`.

A policy emits a chunk of K actions at policy rate.  This layer turns that into a
task-space pose target per TCP at command rate.

Command state (x, v, a) is carried across chunk boundaries and never reset at a
seam; that carried state is what makes the seam smooth.  x is held in the model's
own base frame -- the same frame the action deltas are expressed in -- with
orientation as a *continuous* rotation vector, unwrapped against the previous
tick rather than re-derived from a matrix each time, so that (v, a) in rotation
coordinates stay meaningful across a full turn.

`point_to_point` is the no-preview law: each waypoint is treated as a place the
axis must be able to STOP, tracked with nested jerk-limited braking laws.  That
is the only safe assumption when the future is unknown -- the target may never
move again -- and it is the right mode for sparse, terminal targets.  Its cost
is structural: it plans a stop toward every waypoint and is interrupted by the
next one, so on a path sampled faster than it can settle it chronically
under-travels.  Measured on the rig's 60 s teach recording (0.43 Hz hand
motion), that is a ~2x amplitude loss.

`follower` is the preview law, for the VLA case this controller exists for: the
policy hands over K = 8-16 future poses at 20 ms spacing, so interior waypoints
are VIA points with known future, not destinations.  The chunk becomes one C1
cubic Hermite through all K waypoints -- knot velocities from central
differences, so the command passes THROUGH each waypoint at path speed instead
of braking at it.  Past the final waypoint it decelerates to rest at constant
deceleration over one chunk duration: the runway.  A policy that stops sending
chunks gets a smooth stop along its own last path, spending at most half the
distance it had already committed to.  There are no speed clamps in this mode
-- the chunk defines the motion, and the real limits live in joint space
(URDF velocity limits, layer 3) where they are measured rather than invented.

Both laws are fixed sequences of arithmetic with no data-dependent iteration,
so two runs on two devices produce identical trajectories.  The follower is the
stronger case: polynomial evaluation has none of the sqrt/cbrt cusps that made
the point-to-point law need its lag constants (see AxisLimits).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from remoroo_lc.constants import DTYPE, EPS, POINT_DIM, TASK_DIM
from remoroo_lc.schema import CellSpec
from remoroo_lc.spatial import exp_so3, log_so3


@dataclass(frozen=True)
class AxisLimits:
    """Per-axis velocity / acceleration / jerk clamps, length TASK_DIM.

    `brake_margin` (kappa) is the fraction of the acceleration and jerk budget the
    braking laws PLAN to use.  At kappa = 1 both laws are exactly time-optimal,
    and in discrete time that means the axis is permanently at the edge of its
    ability to stop: one tick of quantisation error carries it past the target, it
    brakes, overshoots the other way, and hunts forever -- measured at +-3 mm on
    the reference cell.  Holding back a little braking authority makes the axis
    arrive slightly short and creep in instead, which converges.  kappa = 0.6
    settles to 4 um and costs about 10% in time-to-target.
    """

    v_max: np.ndarray
    a_max: np.ndarray
    j_max: np.ndarray
    brake_margin: np.floating = DTYPE(0.6)
    #: Time constant of the acceleration loop, in command ticks.  The jerk-optimal
    #: acceleration reference goes as sqrt(|dv|), which is infinitely steep at
    #: dv = 0, so near the target a velocity error of a few hundred um/s still
    #: commands 0.1 m/s^2 and the acceleration never settles.  Below
    #: 2*j_max*kappa*(lag*dt)^2 of velocity error a linear law takes over instead,
    #: which is a plain first-order lag and does settle.
    accel_lag_ticks: np.floating = DTYPE(4.0)
    #: Time constant of the position loop, in command ticks, for the same reason
    #: one tier up.  `stop_velocity` behaves as |e|^(2/3) near zero, so its slope
    #: is INFINITE at e = 0: a position error of 1e-7 m -- one float32 ulp on a
    #: metre-scale coordinate -- still asks for 1e-4 m/s, and which way it asks is
    #: decided by rounding.  Two float32 implementations then command opposite
    #: accelerations on the first tick and drift apart from there.  Capping the
    #: velocity reference with a linear law bounds that gain at 1/(lag*dt), which
    #: is what makes reference and kernels agree at all, and what keeps CPU and
    #: CUDA agreeing with each other.
    pos_lag_ticks: np.floating = DTYPE(12.0)

    @staticmethod
    def from_limits(limits: dict) -> AxisLimits:
        lin = limits["task"]["linear"]
        ang = limits["task"]["angular"]
        def vec(key: str) -> np.ndarray:
            return np.asarray(
                [lin[key]] * POINT_DIM + [ang[key]] * POINT_DIM, dtype=DTYPE
            )
        return AxisLimits(
            vec("v_max"),
            vec("a_max"),
            vec("j_max"),
            DTYPE(limits["task"].get("brake_margin", 0.6)),
            DTYPE(limits["task"].get("accel_lag_ticks", 4.0)),
            DTYPE(limits["task"].get("pos_lag_ticks", 12.0)),
        )


def _sign(x: np.ndarray) -> np.ndarray:
    return np.sign(x).astype(DTYPE)


def stop_velocity(d: np.ndarray, a_max: np.ndarray, j_max: np.ndarray) -> np.ndarray:
    """Highest speed from which an axis can still stop within distance d.

    Closed form for a jerk-limited stop.  Decelerating from v to rest ramps the
    acceleration down to -a_max and back up to zero, so the deceleration itself
    takes a_max/j_max seconds to establish -- and during that time the axis keeps
    covering ground.  The naive sqrt(2 a_max d) ignores that lag and hands back a
    speed the axis cannot actually shed in time, which shows up as a limit cycle
    around the target rather than as convergence to it.

    Two regimes, selected by distance alone (never by iteration state):
      triangular  (peak decel never reaches a_max):  d = v^1.5 / sqrt(j_max)
      trapezoidal (it does):                         d = v^2/(2 a_max) + v a_max/(2 j_max)
    """
    d = np.maximum(d, DTYPE(0.0))
    v_tri = np.cbrt(d * d * j_max).astype(DTYPE)
    b = (a_max * a_max / j_max).astype(DTYPE)  # speed at which the regime switches
    v_trap = (
        DTYPE(0.5) * (-b + np.sqrt(b * b + DTYPE(8.0) * a_max * d))
    ).astype(DTYPE)
    return np.where(v_tri <= b, v_tri, v_trap).astype(DTYPE)


def jerk_limited_step(
    x: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    x_tgt: np.ndarray,
    lim: AxisLimits,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One tick of the per-axis jerk-limited tracker.

    Two nested braking laws: the velocity reference is capped by the speed from
    which the axis can still stop within the remaining distance under BOTH the
    acceleration and jerk limits (see `stop_velocity`), and the acceleration
    reference is capped by the acceleration that can still be brought to zero
    within the remaining velocity error at j_max.  The three clamps that follow
    make limit satisfaction structural rather than emergent -- |a| <= a_max,
    |v| <= v_max, and |da/dt| <= j_max hold by construction on every tick,
    whatever the input does.
    """
    dtf = DTYPE(dt)
    k = lim.brake_margin
    t_v = DTYPE(lim.pos_lag_ticks) * dtf
    t_a = DTYPE(lim.accel_lag_ticks) * dtf
    e = (x_tgt - x).astype(DTYPE)
    v_ref = _sign(e) * np.minimum(
        np.minimum(lim.v_max, stop_velocity(np.abs(e), lim.a_max * k, lim.j_max * k)),
        np.abs(e) / t_v,
    )
    dv = (v_ref - v).astype(DTYPE)
    a_ref = _sign(dv) * np.minimum(
        np.minimum(lim.a_max, np.sqrt(DTYPE(2.0) * lim.j_max * k * np.abs(dv))),
        np.abs(dv) / t_a,
    )
    j = np.clip((a_ref - a) / dtf, -lim.j_max, lim.j_max).astype(DTYPE)
    a_new = np.clip(a + j * dtf, -lim.a_max, lim.a_max).astype(DTYPE)
    v_new = np.clip(v + a_new * dtf, -lim.v_max, lim.v_max).astype(DTYPE)
    x_new = (x + v_new * dtf).astype(DTYPE)
    return x_new, v_new, a_new


class ChunkInterpolator:
    """Per-TCP command generator, sized entirely from the cell."""

    #: How a chunk's K deltas compose.  "cumulative" integrates them (delta k is
    #: applied on top of waypoint k-1); "per_observation" anchors every delta to
    #: the pose observed at chunk start.  For K = 1 the two agree.  See README
    #: for the note on verifying this against Isaac-GR00T before VLA integration.
    DELTA_MODES = ("cumulative", "per_observation")
    #: Tracking law.  See the module docstring; "follower" is the VLA mode.
    TRACK_MODES = ("point_to_point", "follower")

    def __init__(self, cell: CellSpec, delta_mode: str = "cumulative") -> None:
        if delta_mode not in self.DELTA_MODES:
            raise ValueError(f"delta_mode must be one of {self.DELTA_MODES}")
        self.cell = cell
        self.delta_mode = delta_mode
        self.mode = str(cell.limits["task"].get("mode", "point_to_point"))
        if self.mode not in self.TRACK_MODES:
            raise ValueError(f"task.mode must be one of {self.TRACK_MODES}")
        self.lim = AxisLimits.from_limits(cell.limits)
        rates = cell.limits["rates"]
        self.dt_p = DTYPE(1.0 / float(rates["policy_hz"]))
        self.dt_c = DTYPE(1.0 / float(rates["command_hz"]))
        self.n_tcps = cell.n_tcps
        self.eff_dim = sum(cell.effector_widths)
        self.eff_rate = np.concatenate(
            [np.full(t.effector.width, t.effector.rate_limit, dtype=DTYPE) for t in cell.tcps]
            or [np.zeros(0, dtype=DTYPE)]
        ).astype(DTYPE)

        # Per-TCP reference orientation.  The rotation part of the command state
        # is the rotation vector of the command RELATIVE TO THIS, not relative to
        # the model base frame.
        #
        # Relative to the base frame it would routinely sit near a half turn --
        # any tool pointing down at a table is roughly pi from a base whose z is
        # up -- and log_so3 near pi recovers the axis from the symmetric part of
        # R, whose entries carry absolute error eps.  Taking a square root of that
        # costs half the mantissa: sqrt(1e-7) is 3e-4 rad of axis error, which
        # comes straight back out through exp_so3 as a real orientation command
        # error, and the 1/dt_c task gain turns 3e-4 rad into 0.075 rad/s of
        # angular velocity that nobody asked for.  Anchoring at the reset
        # orientation keeps every log_so3 argument near identity, where it is
        # accurate to full float32 precision.
        self.R_ref = np.stack(
            [np.eye(POINT_DIM, dtype=DTYPE) for _ in range(max(self.n_tcps, 1))]
        )[: self.n_tcps].astype(DTYPE)

        # Command state, in each TCP's model base frame.
        self.x = np.zeros((self.n_tcps, TASK_DIM), dtype=DTYPE)
        self.v = np.zeros((self.n_tcps, TASK_DIM), dtype=DTYPE)
        self.a = np.zeros((self.n_tcps, TASK_DIM), dtype=DTYPE)
        self.eff = np.zeros(self.eff_dim, dtype=DTYPE)

        # Active chunk: absolute waypoints in base frame.
        self.waypoints = np.zeros((0, self.n_tcps, TASK_DIM), dtype=DTYPE)
        self.eff_waypoints = np.zeros((0, self.eff_dim), dtype=DTYPE)
        self.anchor = np.zeros((self.n_tcps, TASK_DIM), dtype=DTYPE)
        self.eff_anchor = np.zeros(self.eff_dim, dtype=DTYPE)
        self.t_chunk = DTYPE(0.0)

        # Follower knots: velocity at each waypoint, and the command state
        # captured at the seam (knot 0 of segment 0), which is what carries C1
        # continuity across chunk replacement.
        self.way_v = np.zeros((0, self.n_tcps, TASK_DIM), dtype=DTYPE)
        self.seam_x = np.zeros((self.n_tcps, TASK_DIM), dtype=DTYPE)
        self.seam_v = np.zeros((self.n_tcps, TASK_DIM), dtype=DTYPE)

    # ------------------------------------------------------------------ #
    def reset(self, p_base: np.ndarray, R_base: np.ndarray) -> None:
        """Initialise command state from the current TCP poses (base frame)."""
        for i in range(self.n_tcps):
            self.x[i, :POINT_DIM] = p_base[i]
            # r = 0 by construction: the reference orientation IS this one.
            self.x[i, POINT_DIM:] = 0.0
            self.R_ref[i] = np.asarray(R_base[i], dtype=DTYPE)
        self.v[:] = 0.0
        self.a[:] = 0.0
        off = 0
        for t in self.cell.tcps:
            g = t.effector.width
            if g:
                self.eff[off : off + g] = np.asarray(t.effector.default, dtype=DTYPE)
            off += g
        self.waypoints = np.zeros((0, self.n_tcps, TASK_DIM), dtype=DTYPE)
        self.eff_waypoints = np.zeros((0, self.eff_dim), dtype=DTYPE)
        self.anchor = self.x.copy()
        self.eff_anchor = self.eff.copy()
        self.t_chunk = DTYPE(0.0)
        self.way_v = np.zeros((0, self.n_tcps, TASK_DIM), dtype=DTYPE)
        self.seam_x = self.x.copy()
        self.seam_v = self.v.copy()

    # ------------------------------------------------------------------ #
    def set_chunk(
        self, actions: np.ndarray, p_base: np.ndarray, R_base: np.ndarray
    ) -> None:
        """Decode a chunk of K actions into absolute base-frame waypoints.

        `actions` is (K, action_dim); `p_base` / `R_base` are the CURRENT measured
        TCP poses in each model's base frame, which is what the relative deltas
        are defined against.
        """
        actions = np.asarray(actions, dtype=DTYPE).reshape(-1, self.cell.action_dim)
        k_steps = actions.shape[0]
        slices = self.cell.action_slices()

        anchor = np.zeros((self.n_tcps, TASK_DIM), dtype=DTYPE)
        anchor_R = [None] * self.n_tcps
        for i in range(self.n_tcps):
            anchor[i, :POINT_DIM] = p_base[i]
            # Unwrap the anchor against the RUNNING command state rather than
            # taking log_so3 of the measured rotation directly.  log_so3 returns a
            # representative in [0, pi]; the command state is a continuous,
            # already-unwrapped rotation vector that may be several turns away
            # from it.  Re-wrapping here would make every chunk seam look like a
            # multi-turn reversal to the tracker, which is a real command the
            # robot would then execute.
            r_prev = self.x[i, POINT_DIM:]
            ref_T = self.R_ref[i].T
            anchor[i, POINT_DIM:] = r_prev + log_so3(
                (ref_T @ np.asarray(R_base[i], dtype=DTYPE)) @ exp_so3(r_prev).T
            )
            anchor_R[i] = np.asarray(R_base[i], dtype=DTYPE)

        wp = np.zeros((k_steps, self.n_tcps, TASK_DIM), dtype=DTYPE)
        eff_wp = np.zeros((k_steps, self.eff_dim), dtype=DTYPE)
        for i in range(self.n_tcps):
            ref_T = self.R_ref[i].T
            R_prev = anchor_R[i]
            p_prev = anchor[i, :POINT_DIM].copy()
            r_prev = anchor[i, POINT_DIM:].copy()
            for k in range(k_steps):
                d = actions[k, slices[i][0]]
                if self.delta_mode == "cumulative":
                    p_new = p_prev + d[:POINT_DIM]
                    R_new = exp_so3(d[POINT_DIM:]) @ R_prev
                else:
                    p_new = anchor[i, :POINT_DIM] + d[:POINT_DIM]
                    R_new = exp_so3(d[POINT_DIM:]) @ anchor_R[i]
                # Unwrap against the previous waypoint, in the reference frame, so
                # the sequence stays continuous through +-pi and every log_so3
                # argument stays near identity.
                r_new = r_prev + log_so3((ref_T @ R_new) @ exp_so3(r_prev).T)
                wp[k, i, :POINT_DIM] = p_new
                wp[k, i, POINT_DIM:] = r_new
                p_prev, R_prev, r_prev = p_new, R_new, r_new
        off = 0
        for i in range(self.n_tcps):
            g = self.cell.tcps[i].effector.width
            if g:
                eff_wp[:, off : off + g] = actions[:, slices[i][1]]
            off += g

        self.waypoints = wp
        self.eff_waypoints = eff_wp
        self.anchor = anchor
        self.eff_anchor = self.eff.copy()
        self.t_chunk = DTYPE(0.0)

        # Follower knot velocities, from the decoded waypoint chain itself.  The
        # chain including its measured anchor is the path the policy declared, so
        # the velocity through waypoint k is read off its neighbours by central
        # difference; the last waypoint takes the backward difference, which is
        # the speed the runway then sheds.  All of it in the unwrapped chart, so
        # rotation differences are plain subtractions.
        if self.mode == "follower":
            knots = np.concatenate([anchor[None], wp], axis=0)  # (K+1, T, 6)
            vk = np.zeros_like(wp)
            if k_steps == 1:
                vk[0] = (knots[1] - knots[0]) / self.dt_p
            else:
                vk[: k_steps - 1] = (knots[2:] - knots[:-2]) / (DTYPE(2.0) * self.dt_p)
                vk[k_steps - 1] = (knots[k_steps] - knots[k_steps - 1]) / self.dt_p
            self.way_v = vk.astype(DTYPE)
            self.seam_x = self.x.copy()
            self.seam_v = self.v.copy()

    # ------------------------------------------------------------------ #
    def _chunk_target(self) -> tuple[np.ndarray, np.ndarray]:
        """Time-interpolated target inside the active chunk."""
        if self.waypoints.shape[0] == 0:
            return self.anchor.copy(), self.eff_anchor.copy()
        k_steps = self.waypoints.shape[0]
        s = float(self.t_chunk / self.dt_p)
        s = min(max(s, 0.0), float(k_steps))
        i0 = int(np.floor(s))
        i0 = min(i0, k_steps - 1)
        frac = DTYPE(s - i0)
        lo = self.anchor if i0 == 0 else self.waypoints[i0 - 1]
        hi = self.waypoints[i0]
        eff_lo = self.eff_anchor if i0 == 0 else self.eff_waypoints[i0 - 1]
        eff_hi = self.eff_waypoints[i0]
        return (
            (lo + (hi - lo) * frac).astype(DTYPE),
            (eff_lo + (eff_hi - eff_lo) * frac).astype(DTYPE),
        )

    def _follower_state(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the chunk spline at absolute chunk time t.

        Segment s covers [s*dt_p, (s+1)*dt_p) between knots s and s+1, where
        knot 0 is the seam (command state at chunk arrival) and knots 1..K are
        the waypoints.  Past knot K a single Hermite segment to
        (w_K + v_K*T/2, 0) over T = K*dt_p is EXACTLY constant deceleration
        v_K/T -- the cubic's t^3 coefficient cancels for that endpoint choice --
        so the runway needs no extra law and no extra constants.  Past the
        runway: hold.
        """
        k_steps = self.waypoints.shape[0]
        dt_p = float(self.dt_p)
        s = t / dt_p
        if s < float(k_steps):
            i0 = min(int(s), k_steps - 1)
            tau = DTYPE(s - i0)
            seg_t = DTYPE(dt_p)
            x0 = self.seam_x if i0 == 0 else self.waypoints[i0 - 1]
            v0 = self.seam_v if i0 == 0 else self.way_v[i0 - 1]
            x1, v1 = self.waypoints[i0], self.way_v[i0]
        else:
            seg_t = DTYPE(k_steps * dt_p)
            tau = DTYPE((t - k_steps * dt_p) / float(seg_t))
            x0, v0 = self.waypoints[k_steps - 1], self.way_v[k_steps - 1]
            x1 = (x0 + v0 * (seg_t * DTYPE(0.5))).astype(DTYPE)
            v1 = np.zeros_like(v0)
            if float(tau) >= 1.0:
                return x1.copy(), np.zeros_like(v0), np.zeros_like(v0)
        t2 = tau * tau
        t3 = t2 * tau
        h00 = DTYPE(2.0) * t3 - DTYPE(3.0) * t2 + DTYPE(1.0)
        h10 = t3 - DTYPE(2.0) * t2 + tau
        h01 = DTYPE(-2.0) * t3 + DTYPE(3.0) * t2
        h11 = t3 - t2
        x = (h00 * x0 + h01 * x1 + seg_t * (h10 * v0 + h11 * v1)).astype(DTYPE)
        d00 = (DTYPE(6.0) * t2 - DTYPE(6.0) * tau) / seg_t
        d10 = DTYPE(3.0) * t2 - DTYPE(4.0) * tau + DTYPE(1.0)
        d11 = DTYPE(3.0) * t2 - DTYPE(2.0) * tau
        v = (d00 * (x0 - x1) + d10 * v0 + d11 * v1).astype(DTYPE)
        a = (
            (DTYPE(12.0) * tau - DTYPE(6.0)) * (x0 - x1) / (seg_t * seg_t)
            + (DTYPE(6.0) * tau - DTYPE(4.0)) * v0 / seg_t
            + (DTYPE(6.0) * tau - DTYPE(2.0)) * v1 / seg_t
        ).astype(DTYPE)
        return x, v, a

    def step(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance one command tick.

        Returns (p_base (T, 3), R_base (T, 3, 3), effector command).
        """
        x_tgt, eff_tgt = self._chunk_target()
        if self.mode == "follower":
            if self.waypoints.shape[0]:
                # The command sent this tick is the pose at the END of the tick.
                x_f, v_f, a_f = self._follower_state(float(self.t_chunk) + float(self.dt_c))
                self.x[:] = x_f
                self.v[:] = v_f
                self.a[:] = a_f
            else:
                self.v[:] = 0.0
                self.a[:] = 0.0
        else:
            for i in range(self.n_tcps):
                self.x[i], self.v[i], self.a[i] = jerk_limited_step(
                    self.x[i], self.v[i], self.a[i], x_tgt[i], self.lim, float(self.dt_c)
                )
        if self.eff_dim:
            step = self.eff_rate * self.dt_c
            delta = np.clip(eff_tgt - self.eff, -step, step)
            self.eff = np.clip(self.eff + delta, DTYPE(0.0), DTYPE(1.0)).astype(DTYPE)
        self.t_chunk = DTYPE(self.t_chunk + self.dt_c)

        p = self.x[:, :POINT_DIM].copy()
        R = np.zeros((self.n_tcps, POINT_DIM, POINT_DIM), dtype=DTYPE)
        for i in range(self.n_tcps):
            R[i] = self.R_ref[i] @ exp_so3(self.x[i, POINT_DIM:])
        return p, R, self.eff.copy()

    # ------------------------------------------------------------------ #
    @property
    def command_velocity(self) -> np.ndarray:
        """Current (T, TASK_DIM) command velocity, for diagnostics."""
        return self.v.copy()

    def limit_report(self) -> dict[str, float]:
        """Worst-case usage of each limit since the last reset, as a fraction."""
        return {
            "v_frac": float(np.max(np.abs(self.v) / (self.lim.v_max + EPS))),
            "a_frac": float(np.max(np.abs(self.a) / (self.lim.a_max + EPS))),
        }

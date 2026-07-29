# remoroo-lc — what was built, what it measures, what to argue with

## What was built

The G0 local controller, complete through M6.

- **The cell contract** ([CELL_SPEC.md](CELL_SPEC.md), `schema.py`) — versioned,
  with a `validate_cell` every shipped config passes in CI. A stdlib-only URDF
  reader, so there is no parser dependency to license-audit.
- **Three layers, twice.** `reference/` in readable NumPy is the specification;
  `kernels/` in Warp is the runtime, compiled to CPU and CUDA from one source,
  one thread per environment.
- **Five cells, every test.** Dual 6-DOF, dual 7-DOF, mixed 6+7 with different
  effector widths, a single effectorless chain, and two chains sharing a trunk.
  The entire suite is parameterised over all five; a static gate reads the core's
  tokens and fails on an adapter import, an embodiment word, or a bare DOF-shaped
  literal.
- **Oracles** — ruckig, MuJoCo, ProxQP — each checked at the tolerance its
  quantity's conditioning permits, with the reason stated.
- **Tapes and a scoreboard** that produce a pass/fail report per cell, plus
  adapters (mock, Isaac Lab, xArm), a sysid fitter validated against the plant,
  and benchmarks.

503 tests green in one run; 9 skipped (CUDA, hardware). `license_check.py` passes.

## Measured

### G0 scoreboard — all five cells PASS

| cell | n | T | act | rows | clean RMS pos | clean RMS rot | hard coll | limit viol | clean wall-active | adv wall-active | min pair | peak clean speed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `branched_trunk` | 11 | 2 | 14 | 222 | 1.83 mm | 0.013° | 0 | 0 | 0.00% | 46% | 14.5 mm | 0.113 m/s |
| `dual_7dof` | 14 | 2 | 14 | 318 | 1.36 mm | 0.000° | 0 | 0 | 0.00% | 67% | 15.1 mm | 0.108 m/s |
| `dual_xarm6` | 12 | 2 | 14 | 267 | 1.81 mm | 0.001° | 0 | 0 | 0.00% | 67% | 15.0 mm | 0.167 m/s |
| `mixed` | 13 | 2 | 15 | 292 | 2.53 mm | 0.002° | 0 | 0 | 0.00% | 67% | 15.0 mm | 0.119 m/s |
| `single_6dof_leg` | 6 | 1 | 6 | 62 | 0.82 mm | 0.000° | 0 | 0 | 0.00% | 91% | 14.5 mm | 0.079 m/s |

Bars: clean RMS ≤ 3 mm and ≤ 1.5°, zero hard violations, clean wall-active ≤ 5%.
All met, none tuned to fit. `min pair` bottoms out at `d_safe` (15 mm) on the
adversarial tapes — the damper holding the line, which is what it is for. The
0.5 mm undershoot on two cells is actuation lag, quantified below.

Jerk p99 runs 148–904 rad/s³ and chatter 67–325 acceleration sign flips per
second while a wall binds. Both are highest on the leg, which spends 91% of its
adversarial tape pressed against the floor. Nothing to compare them against yet —
they are the baseline for G1's filter-activation penalty.

### Performance

Single instance, one core, Apple M-series:

| cell | rows | reference (NumPy) | **Warp CPU** |
| --- | ---: | ---: | ---: |
| `single_6dof_leg` | 62 | 0.354 ms | **0.103 ms** (p99 0.117) |
| `branched_trunk` | 222 | 0.461 ms | **0.111 ms** (p99 0.135) |
| `dual_xarm6` | 267 | 0.502 ms | **0.118 ms** (p99 0.139) |
| `dual_7dof` | 318 | 0.682 ms | **0.124 ms** (p99 0.163) |

Budget was 1 ms mean, 2 ms p99. Met with 8× of headroom.

Batched, on an **A10G** (g5.4xlarge), after the optimisation pass described below:

| cell | rows | CUDA @1 | @1024 | **@4096** | @16384 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `single_6dof_leg` | 62 | 1.36 ms | 1.49 | **1.62** | 3.20 |
| `branched_trunk` | 222 | 1.40 | 2.43 | **2.69** | 7.36 |
| `dual_xarm6` | 267 | 1.43 | 2.75 | **3.12** | 8.10 |
| `mixed` | 292 | 1.48 | 2.90 | **3.21** | 10.07 |
| `dual_7dof` | 318 | 1.56 | 3.30 | **3.91** | 11.92 |

Budget was 5 ms at 4096 environments. Met on every cell.

**This took a 6.6× optimisation to reach.** The first measurement was 13.33 ms on
`dual_xarm6`, of which `k_solve` was 8.87 ms — while the active set was *empty*.
The sweep was executing 32 × n_rows iterations per environment to reach
`continue`. Three changes, all exact (same arithmetic, same order,
bitwise-identical output, verified by the CPU-vs-CUDA numbers being unchanged to
the last digit afterwards):

| | | |
| --- | ---: | --- |
| `k_solve` | 8.87 → **0.34 ms** | compact the live rows once per tick and sweep that; restrict `D` and the diagnostics likewise |
| `k_walls` | 2.40 → **0.45 ms** | only project Jacobians for rows the solver will read — deciding costs 5% of filling |
| `k_diffik` | 1.46 → **0.63 ms** | materialise the stacked Jacobian once instead of 3456 branching accessor calls per env; compute the Gram once, since each per-TCP `J_i J_iᵀ` *is* a diagonal block of the stacked one |

Stressed rather than only best-cased: uniformly random joint states, which put 4
rows live at p50 and 18 at p99, cost **4.68 ms** — still inside the target. Cost
now scales with the ACTIVE SET rather than the row count, which is the property
that was wanted.

Two larger wins remain and were deliberately not taken. `G` is laid out
`[env, row, joint]`, so adjacent threads are 12.8 KB apart and every access burns
its own cache line; env-fastest (SoA) is the textbook fix. And 4096 threads is
~4% of the A10G's thread slots, which is why per-env cost was still falling at
16384 — every stage except the PGS is embarrassingly parallel over (env, pair)
and could have a thread each. The PGS must stay one-thread-per-env; that is the
determinism guarantee.

### The real cell

The rig's `robot.urdf`, calibrated base-to-base, obstacle list and
two-TCP-in-one-model structure all load and validate with **no code change** —
the "real assets drop in unchanged" claim, tested for real. The one blocker is
the fitted sphere set: 1480 spheres is 865,000 pairs, ~1000× the 250 Hz budget.
A set fitted at control resolution is being produced upstream; conservative
decimation of the shape-fitting set is the wrong lever, because a 362 mm link
covered by 3 spheres needs 170 mm radii and then self-collides at its own home
pose (measured: −20 mm clearance standing still).

**Our FK against the vendor controllers', on a 60 s two-arm hand-guided
recording, 593 samples, anchored at the tip where no tool-offset setting can
reach:**

| | pos RMS | pos max | rot RMS |
| --- | ---: | ---: | ---: |
| arm1 | **3.50 mm** | 3.84 mm | 0.535° |
| arm2 | **5.43 mm** | 6.15 mm | 0.507° |

Established before commanding any motion, which is the point of running it first.

Three findings from the capture path, each of which would have silently
corrupted a motion test:

1. **The SDK's report stream is 5.3 Hz** on ports 30001/30002 and 99.7 Hz on
   30003. A hand-guided path recorded through it arrives in 0.09 rad steps every
   172 ms. **Port 30000 is the 250 Hz stream and the SDK does not expose it**, so
   `adapters/xarm_rt.py` reads it directly: 249.3 Hz measured, controller dt p50
   4.026 ms, zero dropped frames, plus the controller's own microsecond timestamp
   so sample timing comes from the robot rather than from our scheduler.
2. **Port 30000's orientation is a rotation vector, not roll-pitch-yaw.** The
   vendor table says only "rad". Read as RPY it is 40° wrong; as a rotation
   vector it matches the SDK's own pose to 0.19°.
3. **The two arms have different tool offsets configured — 200 mm and 0 mm — and
   neither matches the URDF's 171.5 mm.** It does not reach us, because we command
   `servo_j` from joint angles, but anything consuming controller TCP poses
   (teleop, hand-eye, VLA data) is reading a different point on each arm.

The recorded reference: hand-guiding reached **1.69 m/s peak TCP speed**, 3.4×
the controller's configured `v_max` of 0.5 m/s. The replay speed sweep therefore
has a real envelope to characterise against, and `servo_j` imposes no speed limit
of its own — the controller is the only constraint, which is what makes the sweep
a test of it.

### Oracles

| layer | oracle | result |
| --- | --- | --- |
| 1 | ruckig (MIT) | limits respected; terminal position within 2 mm over 12 random targets |
| 2 | MuJoCo `mj_jac` + float64 DLS | median ≤ 1e-3 rad/s per joint; every sample inside the float32 conditioning bound |
| 3 | ProxQP (BSD-2) | p95 ≤ 1e-3 rad/s over 100+ instances per cell; converged active set matches exactly |

mink was the intended layer-2 oracle. It is Apache-2.0 but hard-requires
`qpsolvers`, which is **LGPLv3** — `import mink` fails without it, so there is no
way to use only its permissive parts, and the licence gate rejects it. MuJoCo
direct is the substitute and is the better oracle anyway: it shares no code or
conventions with this package, whereas mink wraps the same equations.

## Bugs this found

Each of these was found by a gate, an oracle, or the cell matrix — not by
inspection — and each was a real defect, not a test that needed loosening.

1. **The velocity box could defeat the collision filter.** Clipping `q̇` per joint
   changes its *direction*, and the collision rows constrain a projection of it,
   so the clamp could turn a commanded retreat into an approach. Two 7-DOF chains
   driven together interpenetrated **29 mm** with the row active and the QP solved
   to convergence throughout. Uniform scaling preserves direction; penetration
   gone (−29.4 mm → +15.1 mm, exactly `d_safe`). This is a deviation from the
   brief's "then the box clamp", and the reason is in `solver.py`.
2. **`log_so3` near π** recovers the axis from the symmetric part of R, so a
   square root costs half the float32 mantissa — 3e-4 rad of axis error. The
   rotation command state was referenced to the model base frame, which is ~π away
   for any tool pointing down at a table, so that error was injected as a real
   orientation command and the 1/dt_c task gain turned it into 0.075 rad/s.
   Referenced to the reset orientation instead; agreement improved 50×. Separately,
   the angle came from `arccos(trace)`, which is stationary at π; now `atan2`.
3. **The diff-IK servo clamp was set to the trajectory speed limit.** At 250 Hz a
   2 mm tracking error already reaches 0.5 m/s, so the correction saturated
   immediately and capped achievable TCP speed at ~0.1 m/s: a 0.2 m/s figure-eight
   tracked at 19 mm. Decoupled → 2.9 mm. It also meant the adversarial floor tape
   stopped 52 mm short of the wall it was supposed to be *stopped by* — the filter
   looked safe because the servo was too weak to test it.
4. **Both time-optimal braking laws have infinite slope at zero error.** One ulp
   of position difference commanded 1e-4 m/s and *rounding decided the sign*;
   reference and kernel commanded opposite accelerations on tick 0. Bounded by
   linear laws near the target. At `brake_margin` 1.0 the interpolator also
   limit-cycled ±3 mm around its target forever.
5. **`k_post = 1` leaves a redundant cell's null space unregulated** (~100 s time
   constant). A 30 s figure-eight drifted through a singularity, saturated the
   velocity box, and tracked at 53 mm RMS with a 400 mm excursion. `k_post = 5`
   holds `w` above 1.4e-4 and brings it to 2.7 mm.
6. Smaller: `CellAdapter` silently dropped effector channels when one model
   carried several TCPs (caught only by the branched cell); the sysid fitter was
   4× low until it used the exact discrete update, and again until it stopped
   fitting a regression row across the join between two recordings; zero-length
   effector arrays segfaulted the kernels on the `g = 0` cell.

## Deviations from the brief, and why

| | |
| --- | --- |
| **Box clamp is a uniform scale, not a per-joint clip** | Per-joint clipping lets the box override the collision filter — 29 mm of measured penetration. `solver.box_mode: clip` restores the literal behaviour. |
| **Layer-2 oracle is MuJoCo, not mink** | mink hard-requires LGPLv3 `qpsolvers`; the licence gate is a hard constraint. |
| **`diffik.v_clamp_scale = 4.0`** | The brief says the task velocity is "clamped" without a value. Clamping it at the trajectory limit caps achievable speed at ~0.1 m/s. |
| **`k_post = 5`** | The brief fixes `w_post = 1e-2` but not `k_post`. At 1 the null space is unregulated. |
| **`brake_margin = 0.6`, `pos_lag_ticks`, `accel_lag_ticks`** | Additions, not changes: the time-optimal laws are unusable in discrete float32 without them. |
| **The reference runs float32, not float64** | So it is a bit-level specification of the kernels rather than a more precise approximation of them. |

## Known gaps

- ~~**No CUDA anywhere.**~~ **CLOSED.** Measured on an A10G: 479 tests pass, the
  4096-environment budget is met on every cell, and CPU-vs-CUDA agrees to 1.8e-7 m
  on TCP position, 3.1e-6 rad on `q_target`, 1.6e-4 rad/s on `q_dot` p99 — with
  **every wall-activation decision identical on every sample**. That is ~40×
  tighter than NumPy-vs-Warp, as the shared-codegen argument predicted.
- **`adapters/xarm.py` has never commanded hardware.** The smoke test is written
  and marked `hw`. Its sibling `adapters/xarm_rt.py` HAS run against the real rig
  (read-only) — see below.
- **`adapters/isaaclab.py` has never run inside Isaac Lab.** Tested against a fake
  environment implementing the same two methods.
- **The GR00T action-field ordering is unverified.** Cumulative vs
  per-observation chunk composition is a config switch (`delta_mode`) precisely
  because this is unresolved. Check against Isaac-GR00T `finetune_new_embodiment`
  before the first VLA integration; it does not block G1.
- **Dual PGS at 32 sweeps does not converge on stress instances** with ≥10
  mutually violated rows — up to 4e-2 rad/s from the exact optimum, converging
  monotonically with more sweeps. Those states need spheres already
  interpenetrating and do not arise in a closed loop that starts clear (the tape
  runs peak at four simultaneously active rows), but a teacher exploring against
  the filter can reach them. `solver.iterations` is config; 64 clears the 1e-3 bar.
- **Every gain, link length and rest posture is a placeholder.** The scoreboard
  numbers describe this controller on *these* arms.
- **The reference implementation is slow by construction** (0.35–0.68 ms/step of
  Python). It is a specification. Do not deploy it.

## The three riskiest assumptions

**1. That the plant model resembles a real servo closely enough for these numbers
to mean anything.** *(Now the clear leader, since assumption 2 has been retired.)* The plant is a second-order spring with a pure delay, and its
gains are invented. Two properties of the *architecture* fall straight out of it
and would change with real hardware: the achieved joint velocity cannot exceed
`1/(1 + delay_ticks)` of the commanded one, because `q_target = q_measured + q̇·dt_c`
re-anchors on a measurement that is `N` ticks stale (at N = 2 that ceiling is 0.33
and these gains reach 0.21); and the resulting task loop has a ~20 ms time
constant, which is where the 1–3 mm of tracking lag comes from. If a real
controller's delay or stiffness differs materially, the tracking numbers move and
the damper margins (`d_safe`, `d_infl`, `ξ`) have to be re-sized against the new
lag — the 0.5 mm of `d_safe` undershoot is exactly that lag showing through. This
is the assumption most likely to be wrong and most consequential at G4.

The rig has now supplied one hard number against it: hand-guiding reached
**1.69 m/s**, against a configured `v_max` of 0.5 m/s and a simulated achievable
envelope near 0.2 m/s. Either the real gains are far stiffer than the placeholders
or the controller will be the binding constraint on the real cell. `sysid_tapes.py`
against the real arms settles it, and that is the single highest-value measurement
still outstanding.

**2. ~~That the amplification applies to CPU vs CUDA too.~~ RETIRED — measured,
and it does not bite.** Layer 2 amplifies TCP position to joint velocity by
~1/(σ_min·dt_c) — about 2.5e3 typically, 1.25e4 at a singularity — so one float32
ulp of disagreement about where a metre-scale TCP is becomes ~3e-4 rad/s of
disagreement about how fast a joint should turn. That is real and it is why
NumPy-vs-Warp cannot be compared tightly. But **CPU vs CUDA runs the same
generated code**, so it only differs by FMA contraction and libm, and the measured
result is 1.8e-7 m on TCP position, 1.6e-4 rad/s on `q_dot` p99, and **identical
wall-activation decisions on every sample of every cell**. The prediction and the
measurement agree. What replaces this as a risk is narrower and stated below.

**2b. ~~That an A10G stands in for the training fleet, and a Mac for the edge
box.~~ RETIRED for the edge box — measured on the rig's Orin, and it changes the
deployment recipe.** Single-instance, all five cells, 200 ticks:

| Orin AGX (12-core aarch64, Tegra `nvgpu`) | mean | p99 |
| --- | --- | --- |
| **warp cpu** | 0.68–0.72 ms | **0.74–0.80 ms** |
| warp cuda, 1 env | 2.98–5.05 ms | 3.85–6.43 ms |
| reference (NumPy) | 2.07–4.07 ms | 2.22–**11.38** ms |

⚠️ **Measurement conditions were not clean, and this cuts one way.** The rig's
Orin was under severe pressure while these ran: load average 231 on 12 cores,
swap 27 of 30 GiB consumed, page cache squeezed to 162 MB, and ~57 GiB of the
64 GiB unaccounted for by any process RSS (top consumer 0.3 GiB) — a Tegra
nvmap/GPU-carveout signature. Uptime 2d5h with the rig stack up since boot, so
this predates and is independent of anything here.

Load can only *inflate* a latency measurement, so the conclusion that matters
survives: **true CPU p99 on a quiet box is ≤ 0.80 ms**, and the 5× budget margin
holds a fortiori. What does *not* survive cleanly is the CPU-vs-CUDA ratio — CPU
and GPU contend differently under this kind of pressure, so "4–6× slower" is
softer than it reads, even though the structural argument below predicts the sign
independently. Re-run on a quiet box before treating the absolute CUDA figures as
load-bearing.

**The edge box runs the CPU backend, not CUDA.** p99 0.80 ms on the worst cell is
5× inside the 4 ms budget at 250 Hz. CUDA at one environment is 4–6× *slower* than
CPU and misses the budget on three of five cells — which is not a defect, it is
the one-thread-per-environment design being read back honestly: at `num_envs=1`
that is a single Tegra GPU thread running a sequential 32-sweep Gauss-Seidel loop,
plus ~6 kernel launches a tick, and a single GPU core loses badly to a single
A78AE core. The property that buys structural determinism is the same property
that makes a batch of one pointless on a GPU. CUDA is for the training fleet;
the cell runs the identical source on CPU.

Two things this also settles: the NumPy reference would **not** hold 250 Hz on the
Orin (p99 11.4 ms on `branched_trunk`), so "the kernels are the runtime, the
reference is the spec" is load-bearing rather than stylistic; and the batched
numbers remain single-GPU-model — the A10G still stands in for a training fleet
nobody has run this on.

What is *not* yet measured is narrower than "aarch64 is untested", which would be
wrong — the 507-test suite, determinism tests included, runs on **arm64** every
time it runs here (Apple M5 Pro). Two architectures are already covered: arm64 on
macOS, and x86-64 on the A10G, which is where CPU↔CUDA agreement was proven
because it is the host with both devices.

The real gap is the *platform*, not the instruction set. Apple Silicon and the
Orin's Cortex-A78AE are both ARMv8, but they do not share a libm or a compiler
toolchain — and libm is exactly what the determinism contract is exposed to,
since transcendentals are not IEEE-mandated and `sin`/`cos`/`atan2` may differ in
the last ulp between implementations. Layer 2 then amplifies a last-ulp
disagreement by ~1/(σ_min·dt_c). So what needs proving is *Tegra glibc against
Apple libm*, not ARM against x86. That run was attempted and could not complete;
see the box-health note above.

**3. That the five-cell matrix spans the space of cells Remoroo will enrol.** It
covers different DOF counts, redundant and over-constrained cells, shared joints,
mixed and absent effectors, and an inverted mount. It does **not** cover: a
prismatic joint (the code path exists and is unexercised by any cell), more than
two TCPs, more than two models, a model spanning two controller boxes, mimic
joints in a load-bearing role, or a cell with more than ~300 collision pairs. The
static gate catches hardcoded *names* and *numbers*; it cannot catch a structural
assumption that happens to hold for all five cells. The first real customer cell
that differs structurally is where that gets tested.

## Where to look next

1. Run `test_cpu_and_cuda_agree` and `bench.py --cuda` on a GPU box. Everything
   else is downstream of that.
2. Run the xArm smoke test on the reference cell, then `sysid_tapes.py` against it,
   and re-run the scoreboard with measured gains. That converts assumption 1 from
   an assumption into a measurement.
3. Settle the GR00T field ordering before G3.

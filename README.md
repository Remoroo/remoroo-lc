# remoroo-lc

**The layer between a learned policy and joint servos — identical in simulation and on hardware.**

A policy emits end-effector pose deltas. A robot needs joint targets at 250 Hz that respect
joint limits, avoid self-collision, and behave the *same way* in your training simulator as
on the real arm. `remoroo-lc` is that layer, and it is the same code in both places: batched
across thousands of environments during training, single-instance on the cell's edge box.

```python
from remoroo_lc import load_cell
from remoroo_lc.kernels.warp_backend import BatchedController

cell = load_cell("my_robot.yaml")
ctrl = BatchedController(cell, num_envs=4096, device="cuda")

ctrl.reset(q)                    # q: (envs, n_joints) measured joint state
ctrl.set_chunk(actions, q)       # actions: (envs, K, action_dim) from your policy
q_target = ctrl.step(q, read=False)   # every control tick; stays on device
```

---

## The problem it solves

If you train a policy in simulation and deploy it on a robot, something has to convert
"move the gripper 2 cm left" into joint commands. That converter is usually written twice —
once in the sim stack, once in the robot stack — and the two are never quite the same.

That difference is invisible. Both look correct. Your metrics look fine. The policy then
behaves differently on hardware and you cannot tell whether the policy is wrong or the
simulator was.

`remoroo-lc` exists so there is only one converter, and so it is *provably* the same one:
the same kernels run on CPU and CUDA, and two runs on the same device produce **identical
bits**. Determinism is not a nice-to-have here, it is the whole argument.

### What you get

| | |
| --- | --- |
| **One controller, both worlds** | Batched on CUDA for training, single-instance on CPU for the robot. Same source, same arithmetic. |
| **Any robot from config** | No robot-specific code. Tool frames, joint sets, limits and rest posture are derived from a URDF. |
| **A safety filter that degrades** | Joint-limit and collision velocity dampers that slow the approach instead of clamping the whole command. |
| **Preview-aware** | Modern policies (GR00T-class VLAs) emit 8–16 future poses at a time. Layer 1 fits one C1 spline through the chunk rather than braking at every waypoint. |
| **Measured, not asserted** | Every number below was measured on real hardware and is reproducible from this repo. |

### Where it will not help

It is a **reactive** controller, not a planner. It has no notion of a goal beyond the chunk
it was handed, and it will happily drive into a local minimum that a planner would route
around. Pair it with a planner if you need one. It also assumes a position-controlled
robot — it emits joint *positions*, not torques.

---

## Measured

Real bimanual 6-DOF cell, 312 collision spheres, tracking a 60 s hand-guided path recorded
at 250 Hz from the robot's own controller.

**Accuracy** — how closely the achieved TCP path follows the recorded one:

| | RMS |
| --- | ---: |
| geometric path error | **3.03 mm** |
| best achievable from 50 Hz action samples | 2.56 mm |

**Latency** — one control tick:

| device | mean | p99 |
| --- | ---: | ---: |
| Jetson Orin AGX (CPU), 1 env | 1.16 ms | 1.33 ms |
| A10G (CUDA), 512 envs | 11.5 ms | 11.6 ms |
| A10G (CUDA), 4096 envs | 15.1 ms | **3.7 µs/env** |

**Generality** — cells generated from third-party URDFs by `scripts/cell_from_urdf.py`,
nothing hand-written, engine unchanged:

| robot | joints | path RMS |
| --- | ---: | ---: |
| UR10e | 6 | 0.98 mm |
| Unitree G1 humanoid | 29 | 1.12 mm |
| Franka Panda | 9 | 4.53 mm |
| dual UR10e | 12 | 11.42 mm |

The pattern is worth knowing before you adopt: **cells with no kinematic redundancy track
worst.** `dual_ur10e` is 12 joints against 12 task dimensions, so every task direction has
exactly one joint-velocity answer and a near-singular direction has no alternative to
saturating. Redundant robots have room to move and do better.

---

## Install

```bash
pip install remoroo-lc            # NumPy reference, CPU
pip install remoroo-lc[kernels]   # + Warp kernels (batched, CUDA)
```

From source:

```bash
git clone https://github.com/Remoroo/remoroo-lc && cd remoroo-lc
python -m venv .venv && .venv/bin/pip install -e ".[kernels,dev]"
cp .env.example .env                    # your cell, your robot's IPs
.venv/bin/python -m pytest              # ~540 tests over a 5-cell matrix
.venv/bin/python scripts/license_check.py   # every dependency permissive
```

The test suite needs no robot and no hardware: it runs entirely on the five
synthetic cells in `configs/cells/`, which are chosen to disagree with each other
(different DOF counts, redundant and over-constrained, shared joints, mixed and
absent effectors).

**Your robot's files stay yours.** `configs/rig/`, `assets/rig/` and
`recordings/` are gitignored -- a cell describes an actual machine, including its
calibrated URDF and fitted collision spheres. What ships is the contract
([CELL_SPEC.md](CELL_SPEC.md)), a validating example
(`configs/rig/bimanual.example.yaml`), placeholder assets, and a generator.

---

## Using it in your stack

### 1. Describe your robot

```bash
python scripts/cell_from_urdf.py /path/to/robot.urdf --name my_robot \
    --tcp tool0 -o my_robot.yaml
```

Everything is derived: the root link, the actuated joints, the tool frames, a rest posture.
Edit the result to add collision spheres, obstacles and a second arm.
[CELL_SPEC.md](CELL_SPEC.md) is the full contract and is versioned — treat it as an API.

### 2. Drive it from your policy

The action vector is frozen, per TCP: **3 position deltas + 3 rotation-vector deltas,
relative to the CURRENT TCP pose**, then one normalised float per effector DOF.
`action_dim` is `sum(6 + g_i)` and is derived from the cell.

```python
ctrl.set_chunk(actions, q)         # K future actions, 20 ms apart
for _ in range(ticks_per_action * K):
    out = ctrl.step(q)             # 250 Hz
    q = robot.read_joint_state()
    robot.command(out["q_target"])
```

Deltas are relative to the *measured* pose, so a policy that re-plans from what it observes
cannot accumulate drift. Set `task.mode: follower` and hand over the whole chunk — the
preview is what makes it track. Past the end of the chunk it decelerates to rest over one
chunk duration, so a policy that stops publishing gets a smooth stop rather than a fault.

### 3. Train against it

```python
ctrl.reset(q, rows=env_ids)        # re-initialise only the environments you reset
ctrl.step(q, read=False)           # no host sync; results stay on device
```

Pass a `wp.array` or torch CUDA tensor for `q` and it is used in place, zero copy.
`remoroo_lc.schema.config_stamp(cell)` returns a hash of the resolved configuration —
stamp your datasets and checkpoints with it so an artifact trained under a different
controller is visibly invalid rather than quietly wrong.

### 4. Talk to hardware

Implement `RobotAdapter` — `connect`, `read_state`, `stream_targets`, `set_effector`,
`limits`, `estop`. Every unit conversion and SDK call lives there;
[`adapters/xarm.py`](src/remoroo_lc/adapters/xarm.py) is a worked example. The core never
learns a brand exists, and `tests/test_agnosticism.py` asserts that statically.

---

## How it works

```
action chunk ──[1] interpolator ──→ task pose targets @ 250 Hz
             ──[2] stacked DLS  ──→ joint velocity
             ──[3] safety filter ─→ filtered joint velocity
             ──────────────────────→ q_target = q_measured + q̇·dt
```

**1 — Chunk interpolator.** Fits one C1 cubic Hermite through the chunk's waypoints, with
knot velocities from central differences, so the command passes *through* each waypoint at
path speed. The older `point_to_point` mode brakes at every waypoint and is still there for
sparse, terminal targets; on a densely sampled path it loses about half the amplitude.

**2 — Stacked damped least squares.** One solve over *every* TCP,
`q̇ = Jᵀ(JJᵀ + Λ)⁻¹v`, never one per chain — that is what makes a shared trunk come out
right. Takes the interpolator's velocity as feedforward, so it does not have to build up
error to produce speed. The result is scaled to fit the URDF's own joint velocity limits
before the safety layer sees it.

**3 — Safety filter.** Joint-limit and collision velocity dampers solved by a fixed number
of dual projected Gauss-Seidel sweeps. Constraint rows are bounded per link pair
(`collision.rows_per_block`), which is what makes the memory footprint independent of how
many collision spheres a robot has.

**Output.** `q_target = q_measured + q̇·dt` — integrated from **measurement**, never from
the previous target. That costs a little tracking lag and buys sim/real parity, which is
the entire point.

Determinism comes from structure, not discipline: one thread per environment, so nothing is
reduced across threads and there is no ordering to vary. Fixed iteration counts, fixed
constraint ordering from config load order, float32 throughout.

---

## Status

Version 0.1. Honest about what is and is not proven:

| | |
| --- | --- |
| controller core, all three layers | measured on real hardware, ~570 tests |
| CPU ↔ CUDA agreement | verified on an A10G |
| four third-party robots incl. a 29-DOF humanoid | tracked, engine unchanged |
| `configs/gains.default.yaml` | **placeholders.** Run `scripts/sysid_tapes.py` on your robot to measure the real values before trusting any dynamic simulation built from them |
| hardware adapters | written and read-only-validated; never commanded motion from this repo |

See [SUMMARY.md](SUMMARY.md) for the measurements, the known gaps, and the assumptions
worth arguing with.

---

## Layout

```
src/remoroo_lc/
  schema.py     the cell contract (CELL_SPEC.md is its prose)
  urdf.py       stdlib-only URDF reader
  spatial.py    SO(3)/SE(3), mirrored exactly by the kernels
  reference/    pure-NumPy specification of all three layers
  kernels/      Warp implementation, CPU + CUDA
  adapters/     the only place a robot brand exists
```

Start with [CELL_SPEC.md](CELL_SPEC.md), then read `reference/` layer by layer — each
module's docstring explains what it does and, where it matters, why the obvious alternative
is wrong.

---

## Contributing

Two rules, both enforced by tests:

1. **No robot-specific code outside `adapters/`.** No hardcoded DOF counts, frame names or
   brands. `tests/test_agnosticism.py` reads the source and fails on violations.
2. **Every dependency must be permissive** (MIT, BSD, Apache-2.0, ISC, Zlib).
   `scripts/license_check.py` fails CI otherwise. This ships inside commercial products.

New behaviour needs a test that would fail without it, and any change to a measured number
needs the measurement.

## License

Apache-2.0 — see [LICENSE](LICENSE). Commercial use is fine, including embedding this in a
product, and the licence includes an explicit patent grant.

# remoroo-lc

Remoroo's local reactive controller. It sits between a learned policy — a PPO
teacher in simulation, GR00T N1.7 on real robots — and the joint servos of any
robot cell, and it runs the same code in both places: batched across thousands of
Isaac Lab environments during training, and single-instance at 250 Hz on the
cell's edge box.

Two requirements drive every decision in here:

**Determinism.** The same kernels on the same device produce identical bits. Fixed
iteration counts, fixed constraint ordering, float32 throughout, one thread per
environment so there is no reduction to reorder. Any behavioural difference
between the two worlds becomes an irreducible gap at G4, so determinism beats
optimality everywhere.

**Embodiment agnosticism.** The core contains no robot-specific code, no hardcoded
DOF, no frame names. A cell is defined entirely by config. Chains may share
joints; a TCP may have no effector at all. Enforced by a static gate and by
running the entire test suite over five different cells.

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

.venv/bin/python scripts/license_check.py     # dependency licences
.venv/bin/python -m pytest                    # 507 tests over the cell matrix
.venv/bin/python scripts/run_g0.py            # the G0 scoreboard -> reports/
.venv/bin/python scripts/bench.py             # performance -> reports/bench.json
```

```python
from remoroo_lc import load_cell
from remoroo_lc.kernels.warp_backend import BatchedController

cell = load_cell("configs/cells/dual_xarm6.yaml")
ctrl = BatchedController(cell, num_envs=1, device="cpu")

ctrl.reset(q[None])                      # q: (n_joints,) measured
ctrl.set_chunk(actions, q[None])         # actions: (K, cell.action_dim)
out = ctrl.step(q[None])                 # every command tick
out["q_target"], out["effector"]
```

`remoroo_lc.reference.Controller` is the same thing in readable NumPy. It is the
specification the kernels are checked against, not the runtime.

## What it does, in three layers

| | | |
| --- | --- | --- |
| **1** | chunk interpolator | K actions at 16 Hz → task-space pose targets at 250 Hz, under per-axis velocity/acceleration/jerk clamps. Command state carries across chunk seams. |
| **2** | stacked damped IK | ONE damped least-squares solve over every TCP in the cell, `q̇ = Jᵀ(JJᵀ + Λ)⁻¹v`. Not one per chain — that is what makes a shared trunk come out right. |
| **3** | QP safety filter | joint-limit and collision velocity dampers, solved by 32 fixed sweeps of dual projected Gauss-Seidel, then the velocity box. |

Output: `q_target = q_measured + q̇·dt_c`. Integrating from **measurement**, never
from the previous target — that costs a little tracking lag and buys sim/real
parity, which is the entire point.

## Real assets drop in unchanged

`assets/` holds placeholder URDFs and sphere files with invented link lengths.
Every one of them is an exact, validating instance of the contract in
[CELL_SPEC.md](CELL_SPEC.md). To use a real cell:

1. Point `models[].urdf` and `models[].spheres` at what `remoroo setup` emitted.
2. `python scripts/calibrate_wthresh.py <cell>.yaml --write` — `w_thresh` scales
   with link length and cannot be carried over from another robot.
3. `python scripts/sysid_tapes.py <cell>.yaml --out gains.yaml` — measure `kp`,
   `kd` and the command delay rather than inheriting the placeholders.
4. Choose a `rest` posture (see CELL_SPEC §7) and `python scripts/run_g0.py <cell>.yaml`.

Nothing in `src/` changes.

## Adding a robot brand

One adapter file, one cell yaml, sphere files, one sysid run. Nothing else.

1. `src/remoroo_lc/adapters/<brand>.py` implementing `RobotAdapter`: `connect`,
   `disconnect`, `read_state`, `stream_targets`, `set_effector`, `limits`,
   `estop`. Every unit conversion, SDK call, effector mapping and rate constraint
   lives there — [`xarm.py`](src/remoroo_lc/adapters/xarm.py) is the worked
   example.
2. A cell yaml naming the models after the adapters that drive them.
3. Sphere files.
4. A sysid run for that cell's gains.

The core never learns a brand exists;
`tests/test_agnosticism.py` asserts it statically. If a brand ever requires a
change outside `adapters/`, that is a bug in the abstraction.

## Isaac Lab

```python
from remoroo_lc.adapters.isaaclab import IsaacLabAdapter

adapter = IsaacLabAdapter(cell, env, device="cuda", joint_order=perm)
adapter.reset()
adapter.set_chunk(actions)          # (num_envs, K, action_dim)
q_target = adapter.step_device()    # stays on device
```

`env` needs only `num_envs`, `read_joint_state()` and `write_joint_targets()`; the
seam is tested against a fake environment, no simulator required.

**`joint_order` is not optional in practice.** Isaac Lab orders joints by its own
articulation parse and this package orders them by the cell file. Getting it wrong
produces a controller that works and a robot that moves the wrong joints, which
looks fine in aggregate metrics. Pass the permutation explicitly.

Warp arrays and torch tensors interop zero-copy through the CUDA array interface
(`wp.to_torch` / `wp.from_torch`), so a training loop should use `step_device()`
and never round-trip through the host.

## Hardware smoke test

`tests/test_adapters.py::test_xarm_servo_smoke` streams a 5 s clean tape in servo
mode at a quarter of the shipped task limits, and halts on any hard violation or
a solve p99 above 4 ms.

Marked `@pytest.mark.hw`. **Never run in CI.** It needs a real cell, a real estop
within reach, and someone watching:

```bash
REMOROO_LC_HW_CELL=/path/to/cell.yaml \
REMOROO_LC_HW_HOSTS=$REMOROO_LC_HOSTS,$REMOROO_LC_HOSTS \
  pytest -m hw
```

## Backend

Warp (Apache-2.0, verified by `scripts/license_check.py`). One thread per
environment: Layer 3 is a sequential Gauss-Seidel sweep in which each row reads
what the previous row just wrote, so as batched tensor operations it would be
32 × m tiny kernel launches per tick and launch-bound at any batch size. Inside a
thread it is a loop. It also means no cross-thread reduction anywhere, which is
why bitwise determinism is structural rather than requested.

**Pick the device by batch size, not by what hardware is present.** Training runs
batched on CUDA. The cell's edge box runs the same source on **CPU** — measured on
the rig's Orin, `device="cpu"` is 0.68 ms mean / 0.80 ms p99, while `device="cuda"`
at one environment is 3.0–5.1 ms. A batch of one is a single GPU thread plus six
kernel launches; there is nothing there for a GPU to do. Same corollary on the
training side: CUDA only pays once the batch is large.

A PyTorch fallback with identical semantics is the documented alternative if the
Warp licence ever fails the gate. It is not currently needed and is not
implemented.

## Placeholder vs production

| | status |
| --- | --- |
| `assets/urdf/*.urdf`, `assets/spheres/*.json` | **placeholder** — invented link lengths; valid instances of the contract |
| `configs/gains.default.yaml` | **placeholder** — plausible, not measured. Real values from `sysid_tapes.py` |
| `configs/cells/*.yaml` `rest` postures | **placeholder** — chosen for these placeholder arms |
| `configs/cells/*.yaml` `w_thresh` | **calibrated** for the placeholder assets; recalibrate for real ones |
| `configs/limits.yaml` | **defaults** — several tuned against measurement; see SUMMARY.md |
| `src/remoroo_lc/**` | production |
| `adapters/xarm.py` | **written, never commanded hardware** |
| `adapters/xarm_rt.py` | **run against the real rig, read-only** — 249 Hz, zero dropped frames |
| `adapters/isaaclab.py` | **written, tested against a fake env**, never run in Isaac Lab |
| CUDA kernel path | **validated on an A10G** — 479 tests pass, CPU/CUDA agree, 4096-env budget met |
| Jetson Orin (deployment target) | **benchmarked** — run the **CPU** backend there: p99 0.80 ms against a 4 ms budget. CUDA at 1 env is 4–6× slower; see SUMMARY.md §2b |

## Layout

```
src/remoroo_lc/
  schema.py         the cell contract (CELL_SPEC.md is its prose)
  urdf.py           stdlib-only URDF reader
  spatial.py        SO(3)/SE(3), mirrored exactly by the kernels
  reference/        pure-NumPy specification of all three layers
  kernels/          Warp implementation, CPU + CUDA
  plant.py          second-order per-joint plant for headless testing
  tapes.py          action tapes, generated per cell
  scoreboard.py     G0 metrics and reports
  adapters/         the only place a brand exists
```

## Reading order

[CELL_SPEC.md](CELL_SPEC.md) for the contract, then `reference/` layer by layer —
each module's docstring explains what it does and, where it matters, why the
obvious alternative is wrong. [SUMMARY.md](SUMMARY.md) has the measured numbers,
the known gaps, and the assumptions worth arguing with.

# The cell contract, version 1.0

A **cell** is everything `remoroo-lc` needs to control a customer's robot cell, and
nothing about the task it will do. `remoroo setup` emits it; `remoroo-lc` consumes
it; `remoroo world` will consume it to build the imagined Isaac Lab world. It is a
product contract, not scaffolding — treat a change to it the way you would treat a
change to a wire protocol.

The executable definition of everything below is
[`src/remoroo_lc/schema.py`](src/remoroo_lc/schema.py). `validate_cell(cell)` is
exported so an emitter can check what it produced before shipping it; every cell
config in this repository is validated by
`tests/test_schema.py::test_every_shipped_cell_validates`.

---

## 1. The abstraction

The unit of control is a **TCP**: a task frame at the end of a kinematic chain in
a fixed-base kinematic tree.

A cell is:

- one or more **models**, each a URDF mounted at a configured base transform;
- a list of **TCPs**, each naming a model and a frame within it, and each
  optionally carrying an **effector** of configured command width `g` (1 for a
  parallel jaw, 0 for a bare frame or a foot, more if an effector needs more);
- **environment primitives** the robot must not hit;
- **limits** and **gains**.

Chains may share joints. A trunk feeding two limbs is one model with two TCPs,
and the joint they share appears once in the cell's joint vector. `g = 0` is a
first-class case: a leg is a chain with no effector, and it flows through every
layer unchanged.

**Fixed base is a v1 boundary.** No floating base, no contact dynamics.

## 2. Files

```
cell.yaml                  # this contract
limits.yaml                # task-space limits, damper margins, solver settings
gains.yaml                 # per-joint plant/actuator gains, from a sysid run
<model>.urdf               # one per model
<model>_spheres.json       # one per model, cuRobo layout
```

Paths inside `cell.yaml` resolve against the directory `cell.yaml` is in.

## 3. cell.yaml

```yaml
lc_spec_version: "1.0"     # REQUIRED.  Major version must be understood by the loader.
name: dual_xarm6           # REQUIRED.  Identifies the cell; tapes are stamped with it.

limits: ../limits.yaml     # path, or an inline mapping
gains:  ../gains.yaml
limits_override:           # optional, deep-merged over `limits`
  collision_damper: { d_infl_m: 0.04 }

models:                    # REQUIRED, at least one
  - name: left             # REQUIRED, unique.  Names the adapter that drives it.
    urdf: ../arm.urdf      # REQUIRED
    spheres: ../arm_spheres.json     # optional, but a model with none is invisible
                                     #   to the collision layer
    base:                  # optional; identity if absent.  World -> model root link.
      xyz: [0.0, 0.35, 0.08]
      rpy: [0.0, 0.0, -0.35]         # OR quat: [w, x, y, z] -- not both
    joints: [j1, j2, ...]  # optional.  Actuated joints, in the order they occupy
                           #   the cell's joint vector.  Absent means "every moving
                           #   joint that is neither locked nor a mimic, in the
                           #   URDF's depth-first order".  PRESENT BUT EMPTY IS AN
                           #   ERROR, not a request to guess.
    locked_joints:         # optional.  Held at a fixed value, never actuated.
      finger_joint: 0.0    #   This is where an effector's own DOF goes.
    rest: [0.0, 1.761, ...]  # optional posture target for this model's joints.
                             #   Absent means mid-range.  See section 7.

tcps:                      # REQUIRED, at least one
  - name: left_tcp         # REQUIRED, unique
    model: left            # REQUIRED, must name a model
    frame: tool0           # REQUIRED, must be a link of that model
    offset:                # optional fixed transform from `frame` to the TCP
      xyz: [0, 0, 0.02]
    effector:              # optional.  Absent means width 0.
      kind: parallel_jaw   #   free string; the core never reads it.  The ADAPTER
                           #   does, and decides what the normalised value means.
      width: 1             #   number of command floats, >= 0
      rate_limit: 4.0      #   units/s on the normalised [0, 1] command
      default: [0.0]       #   value at reset; must have `width` entries
    damping:
      w_thresh: 0.000178   # manipulability threshold; see section 6

environment:               # optional
  - { name: table, type: plane, point: [0, 0, 0], normal: [0, 0, 1] }
  - { name: wall, type: box, xyz: [-0.5, 0, 0.75], rpy: [0, 0, 0], dims: [0.06, 2.0, 1.5] }
  - { name: bulb, type: sphere, xyz: [0.2, 0, 0.4], radius: 0.05 }
  - { name: post, type: cylinder, xyz: [0, 0.3, 0.5], rpy: [0, 0, 0], radius: 0.05, height: 1.0 }

collision:                 # optional; defaults shown
  self_pairs: true         # check sphere pairs within each model
  cross_model: true        # check sphere pairs between models
  environment: true        # check every sphere against every primitive
  ignore_pairs: []         # [[linkA, linkB], ...] -- see section 5
  max_pairs: null          # hard cap; exceeding it is an ERROR, never a silent truncation
```

## 4. URDF

Read with a stdlib-only parser ([`src/remoroo_lc/urdf.py`](src/remoroo_lc/urdf.py)).
It uses only the kinematic tree and ignores inertial, visual, collision, material
and transmission tags — although including inertial and collision geometry is
recommended, because third-party tools (MuJoCo, and therefore the layer-2 oracle)
need them.

Requirements:

- exactly one root link, and a single connected tree (no cycles, no forest);
- every `revolute` and `prismatic` joint has a `<limit>` with `lower`, `upper` and
  a **positive `velocity`** — the velocity limit is load-bearing (it is the QP's
  box) and there is no safe default for it;
- `continuous` joints are treated as revolute with range ±π;
- `mimic` joints are permitted and contribute a fixed offset;
- any branching factor, and fixed joints anywhere.

Every moving joint must be accounted for: actuated (in `joints`), locked (in
`locked_joints`), or a mimic. An unaccounted joint is an error, because the
alternative is a controller that quietly does not know about part of the robot.

## 5. Sphere files

cuRobo layout, JSON or YAML, either nested or flat:

```json
{"robot_cfg": {"kinematics": {"collision_spheres": {
  "link1": [{"center": [0.0, 0.0, 0.05], "radius": 0.045}]
}}}}
```

Centres are in the **link frame**, metres. Every link named must exist in that
model's URDF.

**Pair selection is derived, not authored.** Links welded together by fixed joints
are one rigid body; pairs within a rigid body, and between two rigid bodies joined
by a single joint, are excluded automatically from the joint graph. `ignore_pairs`
exists for the cases that rule does not cover — a wrist whose two-apart links
genuinely cannot touch — and should stay short.

The pair list and its ORDER are fixed at load and are a property of the cell.
Rows keep their slot when inactive. This is what makes the solver's warm start
meaningful and its output reproducible.

## 6. Calibrated values

Two numbers in a cell file are **measured, not chosen**:

| field | what it is | produced by |
| --- | --- | --- |
| `tcps[].damping.w_thresh` | where the DLS damping starts to rise, in the units of `sqrt(det(J J^T))` | `scripts/calibrate_wthresh.py` |
| `gains.yaml` `per_joint` | each joint's `kp`, `kd`, and the cell's command delay | `scripts/sysid_tapes.py` |

`w_thresh` carries units of length³ and therefore scales with the chain's own link
lengths. A value tuned for one robot is meaningless on the next one; the script
takes the 5th percentile of the measure over uniform random joint samples.

## 7. Rest posture

`models[].rest` is the posture the Layer-3 term resolves the null space toward. It
must be inside the joint limits (validated). Choose one that is collision-free and
away from singularities — the shipped cells' postures clear their nearest sphere
pair by 45–72 mm and sit 15° or more from every limit.

Absent, it defaults to joint mid-range, which is valid but usually not sensible.

## 8. What the loader guarantees

After `load_cell` returns without raising:

- every referenced file exists and parses;
- every TCP's frame is a link of its model, and every joint on its path is either
  actuated or locked;
- every actuated joint has a non-empty range and a positive velocity limit;
- the rest posture is inside the joint limits;
- sphere files name only links that exist;
- `n_joints`, `action_dim`, `task_dim` and the constraint row count are all
  derived, and are the only sizes anything downstream uses.

## 9. Action schema (frozen)

Per TCP, in `tcps` order:

```
[ dx, dy, dz, wx, wy, wz,  e_1 ... e_g ]
```

- `dx, dy, dz` — position delta relative to the CURRENT TCP position, in that
  TCP's **model base frame**;
- `wx, wy, wz` — rotation vector composed on SO(3) against the current TCP
  orientation, in the same frame;
- `e_i` — `g` effector floats, **normalised absolute** in [0, 1]. The adapter maps
  them to hardware; 0 and 1 mean whatever that adapter says they mean.

Total width is `sum over TCPs of (6 + g_i)`, independent of every chain's joint
count.

A chunk of K actions composes **cumulatively** by default: waypoint k is delta k
applied on top of waypoint k−1, anchored at the pose measured when the chunk
arrived. `per_observation` mode anchors every delta to that same measured pose.
For K = 1 they agree. See the README's note about verifying field ordering against
Isaac-GR00T before the first VLA integration.

## 10. Versioning

`lc_spec_version` is `MAJOR.MINOR`.

- **MAJOR** — a previously valid file becomes invalid, or its meaning changes.
  The loader refuses majors it does not implement.
- **MINOR** — backward-compatible additions. An older loader ignores fields it
  does not know; a newer loader supplies defaults for fields an older emitter
  omitted.

This document describes **1.0**.

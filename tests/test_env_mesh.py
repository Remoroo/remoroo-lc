"""A mesh obstacle is CARRIED by the schema and REFUSED by the collision layer.

The incident: corner_cell -- the real Siemens rig cell, measured 2026-09-23 --
declares its obstacle world as one STL (`meshes/cell_obstacles.stl`, scale 0.001,
pose straight off the rig's own cell.yaml).  `schema._env_primitive` raised on
`type: mesh`, so the whole measured cell could not be loaded by ANYTHING in this
package: not the kinematics, not the limits, not scripts/sysid_tapes.py, which
reads `models[].joints` and the rates and never looks at the environment at all.
Refusing a measurement at the front door took the entire cell offline to buy a
refusal that belonged three layers down.

So the three legs below are the whole contract:
  (a) the mesh LOADS, and its file, scale and pose survive intact;
  (b) every layer that builds or consults collision geometry REFUSES it by name;
  (c) the four primitive types are untouched -- same codes, same dims, same
      poses, same pair list.  Leg (c) is a pin against a regression, and the
      numbers in it were captured by running this repo at 0873a26 (the commit
      before mesh support) via scripts in the session log; they are measured,
      not chosen.

⚠ The dangerous outcome here was never a crash.  corner_cell declares a mesh and
NO `spheres:` file, so the environment row loops emit zero rows either way: a
mesh silently treated as "nothing" hands the caller a collision world that looks
complete, and a mesh silently treated as a box puts an invented obstacle inside
the QP.  `test_refusal_does_not_depend_on_there_being_rows` is the leg that keeps
the first of those from ever passing for the wrong reason.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from remoroo_lc.kernels.structure import (
    ENV_BOX,
    ENV_CYLINDER,
    ENV_PLANE,
    ENV_SPHERE,
    build_structure,
)
from remoroo_lc.reference.controller import Controller
from remoroo_lc.reference.kinematics import KinematicTree
from remoroo_lc.reference.walls import (
    WallBuilder,
    build_pair_list,
    build_sphere_set,
    env_distance,
    env_distance_batch,
)
from remoroo_lc.schema import (
    CellSpecError,
    EnvMesh,
    EnvPrimitive,
    MeshCollisionUnsupported,
    load_cell,
)
from remoroo_lc.spatial import make_transform, quat_to_mat, rpy_to_mat
from tests.conftest import CELL_DIR, PRIMITIVE_ENVIRONMENT, write_cell

#: corner_cell's own obstacle entry, verbatim -- pose and all, including the
#: quaternion's floating-point dust.  Re-typing a rounded pose is how a measured
#: obstacle quietly moves, and this test exists to prove the pose survives.
MESH_NAME = "cell_obstacles"
MESH_XYZ = [-0.23603686025897397, 0.47207188939545697, -0.054133673953825234]
MESH_QUAT = [
    0.7071067811865476,
    -5.549857282387571e-17,
    -1.181662190535732e-18,
    -0.7071067811865475,
]
MESH_SCALE = [0.001, 0.001, 0.001]

#: A real (if trivial) STL, because the loader checks that the path it resolves
#: points at a file.  Nothing in lc opens it.
STL_TEXT = """solid cell_obstacles
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1000 0 0
    vertex 0 1000 0
  endloop
endfacet
endsolid cell_obstacles
"""


def _mesh_entry(rel_file: str = "meshes/cell_obstacles.stl") -> dict:
    return {
        "name": MESH_NAME,
        "type": "mesh",
        "file": rel_file,
        "scale": list(MESH_SCALE),
        "xyz": list(MESH_XYZ),
        "quat": list(MESH_QUAT),
    }


def _write_mesh_cell(tmp_path, *, environment, drop_spheres=False, collision=None):
    """A dual_xarm6 cell in `tmp_path` whose environment is `environment`.

    The mesh file is written at `tmp_path/meshes/`, one directory BELOW the cell
    file, and referenced relatively -- so a loader that resolved the path against
    the process cwd instead of the cell file's directory would fail this.
    """
    tmp_path = Path(tmp_path)
    (tmp_path / "meshes").mkdir(parents=True, exist_ok=True)
    (tmp_path / "meshes" / "cell_obstacles.stl").write_text(STL_TEXT, encoding="utf-8")

    def mutate(raw):
        raw["environment"] = environment
        if collision is not None:
            raw.setdefault("collision", {}).update(collision)
        if drop_spheres:
            for m in raw["models"]:
                m.pop("spheres", None)

    return write_cell(tmp_path, CELL_DIR / "dual_xarm6.yaml", mutate)


def _digest(*arrays) -> str:
    """Hash of integer structure only.  Deliberately no float arrays: a float32
    digest would pin this machine's libm and BLAS, not the cell's structure."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        assert a.dtype.kind in "iub", f"digest is for integer structure, got {a.dtype}"
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# (a) it loads, and the measurement survives
# --------------------------------------------------------------------------- #


def test_mesh_obstacle_loads_and_keeps_file_scale_and_pose(tmp_path):
    path = _write_mesh_cell(tmp_path, environment=[_mesh_entry()])
    cell = load_cell(path)

    assert len(cell.environment) == 1
    mesh = cell.environment[0]
    assert isinstance(mesh, EnvMesh)
    assert not isinstance(mesh, EnvPrimitive)
    assert mesh.name == MESH_NAME
    assert mesh.ptype == "mesh"

    # ⚠ No `dims`.  A mesh that carried one could be read by primitive code by
    # accident, and the accident would be a zero-size box in the planner.
    assert not hasattr(mesh, "dims")

    # The path is resolved against the CELL FILE's directory, absolute, and real.
    assert mesh.mesh_path.is_absolute()
    assert mesh.mesh_path == (tmp_path / "meshes" / "cell_obstacles.stl").resolve()
    assert mesh.mesh_path.is_file()

    assert np.array_equal(mesh.scale, np.asarray(MESH_SCALE, dtype=np.float32))

    expected = make_transform(
        quat_to_mat(np.asarray(MESH_QUAT, dtype=np.float32)),
        np.asarray(MESH_XYZ, dtype=np.float32),
    )
    assert mesh.pose.shape == (4, 4)
    assert np.array_equal(mesh.pose, expected)
    # The pose is the measurement, so check it against the numbers themselves and
    # not only against the helper that produced it.
    assert np.allclose(mesh.pose[:3, 3], MESH_XYZ, rtol=0, atol=1e-7)


def test_a_mesh_alongside_primitives_leaves_the_primitives_identical(tmp_path):
    """Adding a mesh entry must not perturb the entries around it."""
    without = load_cell(_write_mesh_cell(tmp_path / "a", environment=PRIMITIVE_ENVIRONMENT))
    with_mesh = load_cell(
        _write_mesh_cell(
            tmp_path / "b", environment=[*PRIMITIVE_ENVIRONMENT, _mesh_entry()]
        )
    )
    assert len(with_mesh.environment) == len(without.environment) + 1
    for a, b in zip(without.environment, with_mesh.environment):
        assert (a.name, a.ptype) == (b.name, b.ptype)
        assert np.array_equal(a.pose, b.pose)
        assert np.array_equal(a.dims, b.dims)
    assert isinstance(with_mesh.environment[-1], EnvMesh)


def test_a_mesh_obstacle_needs_its_measurements(tmp_path):
    """file, scale: both required.  An absent measurement must fail, not be guessed."""
    no_file = _mesh_entry()
    no_file.pop("file")
    with pytest.raises(CellSpecError, match="needs `file:`"):
        load_cell(_write_mesh_cell(tmp_path / "a", environment=[no_file]))

    no_scale = _mesh_entry()
    no_scale.pop("scale")
    with pytest.raises(CellSpecError, match="needs an explicit `scale:`"):
        load_cell(_write_mesh_cell(tmp_path / "b", environment=[no_scale]))

    with pytest.raises(CellSpecError, match="does not exist"):
        load_cell(
            _write_mesh_cell(tmp_path / "c", environment=[_mesh_entry("meshes/nope.stl")])
        )

    bad_scale = _mesh_entry()
    bad_scale["scale"] = [0.001, 0.0, 0.001]
    with pytest.raises(CellSpecError, match="must be positive"):
        load_cell(_write_mesh_cell(tmp_path / "d", environment=[bad_scale]))


def test_an_unknown_environment_type_is_still_rejected(tmp_path):
    """Carrying a mesh must not turn the type field into a free-for-all."""
    entry = _mesh_entry()
    entry["type"] = "capsule"
    with pytest.raises(CellSpecError, match="type must be one of"):
        load_cell(_write_mesh_cell(tmp_path, environment=[entry]))


# --------------------------------------------------------------------------- #
# (b) every collision consumer refuses it, by name
# --------------------------------------------------------------------------- #


def _assert_names_the_obstacle(exc_info, *, mesh_path_fragment="cell_obstacles.stl"):
    msg = str(exc_info.value)
    assert MESH_NAME in msg, msg
    assert mesh_path_fragment in msg, msg
    assert "no mesh collision support" in msg, msg
    # It must say what to do instead, in both directions.
    assert "primitives" in msg, msg
    assert "does not consult the environment" in msg, msg


def test_build_structure_refuses_a_mesh_by_name(tmp_path):
    cell = load_cell(
        _write_mesh_cell(tmp_path, environment=[*PRIMITIVE_ENVIRONMENT, _mesh_entry()])
    )
    with pytest.raises(MeshCollisionUnsupported) as exc:
        build_structure(cell)
    _assert_names_the_obstacle(exc)
    assert "structure.build_structure" in str(exc.value)


def test_pair_list_wall_builder_and_controller_refuse_a_mesh(tmp_path):
    cell = load_cell(_write_mesh_cell(tmp_path, environment=[_mesh_entry()]))
    tree = KinematicTree(cell)

    with pytest.raises(MeshCollisionUnsupported) as exc:
        build_pair_list(cell, tree, build_sphere_set(cell, tree))
    _assert_names_the_obstacle(exc)
    assert "walls.build_pair_list" in str(exc.value)

    # WallBuilder and Controller reach it through build_pair_list; neither may
    # hand back a constraint set for a cell whose obstacles lc cannot see.
    with pytest.raises(MeshCollisionUnsupported):
        WallBuilder(cell, tree)
    with pytest.raises(MeshCollisionUnsupported):
        Controller(cell)


def test_env_distance_refuses_a_mesh_rather_than_calling_it_an_unknown_type(tmp_path):
    cell = load_cell(_write_mesh_cell(tmp_path, environment=[_mesh_entry()]))
    mesh = cell.environment[0]
    p = np.zeros(3, dtype=np.float32)

    with pytest.raises(MeshCollisionUnsupported) as exc:
        env_distance(mesh, p)
    _assert_names_the_obstacle(exc)
    assert "unknown primitive type" not in str(exc.value)

    with pytest.raises(MeshCollisionUnsupported) as exc:
        env_distance_batch(mesh, p[None, :])
    _assert_names_the_obstacle(exc)


def test_refusal_does_not_depend_on_there_being_rows(tmp_path):
    """THE silent-wrong case, and the reason the gate is at the door.

    corner_cell declares a mesh obstacle and no `spheres:` file, so there are no
    spheres to pair the obstacle with and the environment row loop emits nothing
    whatever the obstacle is.  A refusal placed inside that loop could never
    fire, and the caller would get an empty-but-plausible pair list.
    """
    cell = load_cell(
        _write_mesh_cell(tmp_path, environment=[_mesh_entry()], drop_spheres=True)
    )
    tree = KinematicTree(cell)
    assert len(build_sphere_set(cell, tree)) == 0, "this leg needs a sphereless cell"
    with pytest.raises(MeshCollisionUnsupported):
        build_pair_list(cell, tree, build_sphere_set(cell, tree))
    with pytest.raises(MeshCollisionUnsupported):
        build_structure(cell)


def test_disabling_environment_collision_does_not_make_a_mesh_acceptable(tmp_path):
    """Pins the documented rule: the refusal is unconditional.

    `build_structure` must put a type code in `env_type` for EVERY obstacle, and
    warp_backend.env_distance treats an unrecognised code as a cylinder, so there
    is no placeholder that is not a phantom obstacle.  One rule at every door
    beats a flag that means one thing on the reference path and another on the
    kernel path.
    """
    cell = load_cell(
        _write_mesh_cell(
            tmp_path, environment=[_mesh_entry()], collision={"environment": False}
        )
    )
    assert cell.collision["environment"] is False
    with pytest.raises(MeshCollisionUnsupported):
        build_structure(cell)
    with pytest.raises(MeshCollisionUnsupported):
        Controller(cell)


def test_the_refusal_is_catchable_as_a_cell_spec_error(tmp_path):
    """Existing `except CellSpecError` / `except ValueError` handlers keep working."""
    cell = load_cell(_write_mesh_cell(tmp_path, environment=[_mesh_entry()]))
    assert issubclass(MeshCollisionUnsupported, CellSpecError)
    assert issubclass(MeshCollisionUnsupported, ValueError)
    with pytest.raises(CellSpecError):
        build_structure(cell)


# --------------------------------------------------------------------------- #
# (c) the primitives are untouched
# --------------------------------------------------------------------------- #

#: Flattened structure of every shipped cell, captured by running this repo at
#: 0873a26 -- the commit BEFORE mesh support -- so a change to the primitive path
#: fails here rather than being discovered on a rig.  Only integer structure is
#: pinned; dims and poses are checked by value below, because a float32 digest
#: would pin this machine's libm rather than the cell.
#:     name -> (n_env, env_type, n_pairs, n_rows, dropped_static_env, pairs_digest)
SHIPPED_BASELINE = {
    "branched_trunk": (1, [ENV_PLANE], 198, 220, 2, "6f8e5823a82c67a6"),
    "dual_7dof": (2, [ENV_PLANE, ENV_BOX], 282, 310, 8, "9b726aec3c16851e"),
    "dual_xarm6": (2, [ENV_PLANE, ENV_BOX], 235, 259, 8, "c63442bfc0d5a3e3"),
    "mixed": (2, [ENV_PLANE, ENV_BOX], 258, 284, 8, "cc2c1838844ca442"),
    "single_6dof_leg": (1, [ENV_PLANE], 48, 60, 2, "fc02188b1a508833"),
}

#: Same, for a dual_xarm6 whose environment is `PRIMITIVE_ENVIRONMENT` -- one of
#: every primitive type, because no shipped cell declares a sphere or a cylinder.
ALL_FOUR_BASELINE = (
    4,
    [ENV_PLANE, ENV_BOX, ENV_SPHERE, ENV_CYLINDER],
    271,
    295,
    16,
    "1554257f75fb80a7",
)


def _flat(cell):
    st = build_structure(cell)
    tree = KinematicTree(cell)
    wb = WallBuilder(cell, tree)
    return (
        st.n_env,
        st.env_type.tolist(),
        st.n_pairs,
        st.n_rows,
        wb.pairs.dropped_static_env,
        _digest(wb.pairs.kind, wb.pairs.a, wb.pairs.b),
    ), st


def test_shipped_cells_flatten_exactly_as_before(cell, cell_path):
    got, _ = _flat(cell)
    assert got == SHIPPED_BASELINE[cell_path.stem]


def test_all_four_primitive_types_flatten_exactly_as_before(tmp_path):
    cell = load_cell(_write_mesh_cell(tmp_path, environment=PRIMITIVE_ENVIRONMENT))
    got, st = _flat(cell)
    assert got == ALL_FOUR_BASELINE

    # And the geometry itself, recomputed from the declared numbers rather than
    # copied out of the loader: plane keeps its normal in `dims`, a box's dims are
    # HALF-extents, a sphere is [r], a cylinder is [r, half-height].
    table, wall, bulb, post = PRIMITIVE_ENVIRONMENT
    expect_dims = np.zeros((4, 3), dtype=np.float32)
    expect_dims[0] = table["normal"]
    expect_dims[1] = np.asarray(wall["dims"], dtype=np.float32) * np.float32(0.5)
    expect_dims[2, 0] = bulb["radius"]
    expect_dims[3, :2] = [post["radius"], np.float32(post["height"]) * np.float32(0.5)]
    assert np.array_equal(st.env_dims, expect_dims)

    expect_pose = np.stack(
        [
            make_transform(np.eye(3, dtype=np.float32), np.asarray(table["point"], np.float32)),
            make_transform(
                rpy_to_mat(np.asarray(wall["rpy"], np.float32)),
                np.asarray(wall["xyz"], np.float32),
            ),
            make_transform(np.eye(3, dtype=np.float32), np.asarray(bulb["xyz"], np.float32)),
            make_transform(
                rpy_to_mat(np.asarray(post["rpy"], np.float32)),
                np.asarray(post["xyz"], np.float32),
            ),
        ]
    )
    assert np.allclose(st.env_pose, expect_pose, rtol=0, atol=1e-7)


def test_primitive_only_cells_still_assemble_and_are_reproducible(tmp_path):
    """The primitive path must still produce constraint rows, not just survive."""
    cell = load_cell(_write_mesh_cell(tmp_path, environment=PRIMITIVE_ENVIRONMENT))
    tree = KinematicTree(cell)
    q = cell.rest_posture()
    a = WallBuilder(cell, tree).assemble(q, tree.fk(q))
    b = WallBuilder(cell, tree).assemble(q, tree.fk(q))
    assert a.G.shape == b.G.shape == (ALL_FOUR_BASELINE[3], cell.n_joints)
    for x, y in ((a.G, b.G), (a.h, b.h), (a.active, b.active), (a.distance, b.distance)):
        assert np.array_equal(x, y)
    assert int(np.count_nonzero(a.active)) > 0

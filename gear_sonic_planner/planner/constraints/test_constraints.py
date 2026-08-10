"""Stage-1 test suite for the constraint layer.

Runs with NO OMPL, NO planner, NO viewer, NO MuJoCo -- NumPy, SciPy and
Pinocchio only.  Executable directly (``python test_constraints.py``), as a
module (``python -m planner.constraints.test_constraints`` from
``gear_sonic_planner/``, or ``python -m
gear_sonic_planner.planner.constraints.test_constraints`` from the repo
root), and pytest-compatible (every test is a ``test_*`` function).

Dataset (T7): path comes from the module constant :data:`DATASET_PATH`,
overridable via the ``GEAR_SONIC_PLANNER_DATASET`` environment variable.  If
the file is absent every T7 test SKIPs with a clear message; T1-T6 run
regardless.

Model fixtures
--------------
Two constraint stacks are exercised:

* ``ctx()`` -- the Stage-1 FIXED-BASE G1 model (pelvis root, ``nq == nv``).
  All Stage-1 behaviour (T1-T7) still runs on it unchanged.
* ``ff_ctx()`` -- the Stage-2 FREE-FLYER G1 model
  (``RobotModel(..., set_floating_base=True)``); the base enters the
  planning space as ``[position(3), rotation vector(3)]``, base first
  (MCVAMP Digit convention).

Stage-2 replacements of Stage-1's fixed-base substitutes:

* T1.4: the free-flyer is now the one ADMITTED multi-DoF joint, so the
  rejection path is exercised with a synthetic SPHERICAL joint instead.
* T1.6: idx_q != idx_v is now asserted on the REAL free-flyer G1 model; the
  synthetic free-flyer model substitute is gone.
* T3.5 / T3.6 ("translate/rotate the floating base"): now REAL base
  perturbations of the free-flyer planning vector; the Stage-1
  target-mutation substitute is gone.
* T7.3b re-runs the T7.3 feet measurement on the free-flyer stack and
  reports the before/after goal errors.

Seeding: all RNG derives from SEED below; the runner prints it on start and
on every failure.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pinocchio as pin

# Make the repo root importable regardless of how this file is invoked
# (script, -m from gear_sonic_planner/, -m from the repo root, pytest).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gear_sonic_planner.planner.constraints.base_constraint import (  # noqa: E402
    Constraint,
    numeric_jacobian,
)
from gear_sonic_planner.planner.constraints.com_constraint import (  # noqa: E402
    CoMConstraint,
)
from gear_sonic_planner.planner.constraints.composable_constraint import (  # noqa: E402
    ComposedConstraint,
)
from gear_sonic_planner.planner.constraints.embedding import (  # noqa: E402
    PlanningEmbedder,
    check_embedder,
)
from gear_sonic_planner.planner.constraints.feet_constraint import (  # noqa: E402
    FeetConstraint,
)
from gear_sonic_planner.planner.constraints.planner_constraints import (  # noqa: E402
    G1Constraints,
)

SEED = 20260808
DATASET_PATH = os.environ.get(
    "GEAR_SONIC_PLANNER_DATASET",
    str(_REPO_ROOT / "test_dataset" / "reference_trajectory_sym.npz"),
)


class SkipTest(Exception):
    """Raised by a test to mark itself skipped (runner reports SKIP)."""


# Free-form notes surfaced in the final summary table, keyed by test name.
_NOTES: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Shared fixtures (built lazily, once)
# ---------------------------------------------------------------------------

_CTX: dict | None = None


def ctx() -> dict:
    """Robot model, embedder and nominal configs shared by all tests."""
    global _CTX
    if _CTX is None:
        from decoupled_wbc.control.robot_model.instantiation.g1 import (
            instantiate_g1_robot_model,
        )

        robot_model = instantiate_g1_robot_model()
        model = robot_model.pinocchio_wrapper.model
        # Planning space: every 1-DoF body joint, in Pinocchio order,
        # excluding the hand joints.  No dimension literals anywhere: the
        # size is len(planning_joint_names).
        planning_joint_names = [
            name for name in list(model.names)[1:] if "hand" not in name
        ]
        q_nominal = robot_model.pinocchio_wrapper.q0.copy()
        embedder = PlanningEmbedder(model, planning_joint_names, q_nominal)
        lower = model.lowerPositionLimit[embedder.idx_q].copy()
        upper = model.upperPositionLimit[embedder.idx_q].copy()
        # Guard against unlimited joints for uniform sampling.
        lower = np.where(np.isfinite(lower), lower, -np.pi)
        upper = np.where(np.isfinite(upper), upper, np.pi)
        _CTX = {
            "robot_model": robot_model,
            "model": model,
            "planning_joint_names": planning_joint_names,
            "embedder": embedder,
            "q_nominal": q_nominal,
            "q0_plan": embedder.extract(q_nominal),
            "lower": lower,
            "upper": upper,
        }
    return _CTX


_FF_CTX: dict | None = None


def ff_ctx() -> dict:
    """Free-flyer robot model, embedder and reference configs (Stage 2).

    The base is the FIRST planning joint (MCVAMP Digit layout: floating
    base first, then the joints); hands stay in q_nominal.
    """
    global _FF_CTX
    if _FF_CTX is None:
        from decoupled_wbc.control.robot_model.robot_model import RobotModel

        urdf_path = (
            _REPO_ROOT
            / "decoupled_wbc/control/robot_model/model_data/g1"
            / "g1_29dof_with_hand.urdf"
        )
        robot_model = RobotModel(
            str(urdf_path), str(urdf_path.parent), set_floating_base=True
        )
        model = robot_model.pinocchio_wrapper.model
        names = list(model.names)
        # names[0] is "universe", names[1] the free-flyer root.
        planning_joint_names = [names[1]] + [
            name for name in names[2:] if "hand" not in name
        ]
        q_nominal = pin.neutral(model)
        embedder = PlanningEmbedder(model, planning_joint_names, q_nominal)
        # Reference standing config: fixed-base nominal joints on a base
        # hovering at the fixed-base model's pelvis height, identity
        # orientation.
        c = ctx()
        base_reference = np.concatenate(
            [np.array([0.0, 0.0, 0.793]), np.zeros(3)]
        )
        q_reference_plan = np.concatenate([base_reference, c["q0_plan"]])
        _FF_CTX = {
            "robot_model": robot_model,
            "model": model,
            "planning_joint_names": planning_joint_names,
            "embedder": embedder,
            "q_nominal": q_nominal,
            "q_reference_plan": q_reference_plan,
            "base_slice": embedder.base_plan_slice,
        }
    return _FF_CTX


def ff_dataset_frame_to_q_plan(data: dict, frame: int) -> np.ndarray:
    """Dataset frame -> free-flyer planning vector [pos, rotvec, joints]."""
    w, x, y, z = np.asarray(data["base_quat"][frame], dtype=float)
    rotation_vector = pin.log3(pin.Quaternion(w, x, y, z).matrix())
    return np.concatenate(
        [
            np.asarray(data["base_pos"][frame], dtype=float),
            rotation_vector,
            np.asarray(data["joint_pos"][frame], dtype=float),
        ]
    )


def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def generic_config(scale: float = 0.15) -> np.ndarray:
    """Nominal standing config plus a small seeded perturbation.

    Small enough to keep the support polygon well-defined, large enough to
    avoid the straight-knee singularity of the exact nominal config.
    """
    c = ctx()
    q = c["q0_plan"] + rng().uniform(-scale, scale, c["embedder"].n_plan)
    return np.clip(q, c["lower"], c["upper"])


def build_spherical_model() -> pin.Model:
    """Small synthetic model with a spherical root and three 1-DoF joints.

    Stage 2 admits the free-flyer into the planning space, so the
    multi-DoF rejection path (T1.4) is exercised with a spherical joint
    (nq = 4, nv = 3) instead of the Stage-1 synthetic free-flyer.
    """
    model = pin.Model()
    root = model.addJoint(
        0, pin.JointModelSpherical(), pin.SE3.Identity(), "spherical_root"
    )
    model.appendBodyToJoint(root, pin.Inertia.Random(), pin.SE3.Identity())
    parent = root
    offset = pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.2]))
    for i in range(3):
        parent = model.addJoint(
            parent, pin.JointModelRY(), offset, f"link_joint_{i}"
        )
        model.appendBodyToJoint(
            parent, pin.Inertia.Random(), pin.SE3.Identity()
        )
    return model


def hip_pitch_lean(scale: float) -> np.ndarray:
    """Nominal config with both hip pitches deflected by ``scale`` rad.

    On the fixed-base model this swings both legs (and therefore the support
    polygon) away from the essentially unchanged CoM -- the mechanism the
    CoM-constraint tests use to push the robot out of stability.
    """
    c = ctx()
    names = c["planning_joint_names"]
    q = c["q0_plan"].copy()
    for name in ("left_hip_pitch_joint", "right_hip_pitch_joint"):
        q[names.index(name)] += scale
    return np.clip(q, c["lower"], c["upper"])


def _config_repr(q: np.ndarray) -> str:
    return np.array2string(
        np.asarray(q, dtype=float), precision=4, max_line_width=120
    )


# ---------------------------------------------------------------------------
# Synthetic constraints for T2 (test the tester)
# ---------------------------------------------------------------------------


class _LinearConstraint(Constraint):
    """error(q) = A @ q -- known Jacobian A."""

    def __init__(self, A: np.ndarray):
        self._A = np.asarray(A, dtype=float)

    @property
    def n_rows(self) -> int:
        return self._A.shape[0]

    @property
    def n_plan(self) -> int:
        return self._A.shape[1]

    def error(self, q_plan) -> np.ndarray:
        return self._A @ np.asarray(q_plan, dtype=float)

    def jacobian(self, q_plan) -> np.ndarray:
        return self._A.copy()


class _QuadraticConstraint(Constraint):
    """error_k(q) = 0.5 q^T Q_k q + b_k^T q with symmetric Q_k."""

    def __init__(self, Qs: list[np.ndarray], bs: list[np.ndarray]):
        self._Qs = [0.5 * (Q + Q.T) for Q in map(np.asarray, Qs)]
        self._bs = [np.asarray(b, dtype=float) for b in bs]

    @property
    def n_rows(self) -> int:
        return len(self._Qs)

    @property
    def n_plan(self) -> int:
        return self._Qs[0].shape[0]

    def error(self, q_plan) -> np.ndarray:
        q = np.asarray(q_plan, dtype=float)
        return np.array(
            [0.5 * q @ Q @ q + b @ q for Q, b in zip(self._Qs, self._bs)]
        )

    def jacobian(self, q_plan) -> np.ndarray:
        q = np.asarray(q_plan, dtype=float)
        return np.vstack([Q @ q + b for Q, b in zip(self._Qs, self._bs)])


# ===========================================================================
# T1 -- Embedding
# ===========================================================================


def test_T1_1_round_trip_exact():
    c = ctx()
    q_plan = rng().uniform(c["lower"], c["upper"])
    check_embedder(c["embedder"], q_plan)
    assert np.array_equal(
        c["embedder"].extract(c["embedder"](q_plan)), q_plan
    )


def test_T1_2_non_planning_entries_preserved():
    c = ctx()
    model, names = c["model"], c["planning_joint_names"]
    # Distinctive values in every non-planning entry of q_nominal.
    q_nominal = c["q_nominal"].copy()
    planning_idx = set(c["embedder"].idx_q.tolist())
    non_planning_idx = [
        i for i in range(model.nq) if i not in planning_idx
    ]
    assert non_planning_idx, "expected non-planning entries (hand joints)"
    for k, i in enumerate(non_planning_idx):
        q_nominal[i] = 0.777 + 1e-3 * k
    embedder = PlanningEmbedder(model, names, q_nominal)
    q_full = embedder(rng().uniform(c["lower"], c["upper"]))
    assert np.array_equal(
        q_full[non_planning_idx], q_nominal[non_planning_idx]
    ), "non-planning entries were not preserved byte-for-byte"


def test_T1_1b_round_trip_free_flyer():
    """Round trip with the BASE in the planning space (Stage 2, A.3).

    The rotation vector passes through exp3/log3, so equality holds to
    float rounding (1e-12) rather than bit-exactly; the fixed-base round
    trip above stays bit-exact.
    """
    f = ff_ctx()
    generator = rng()
    q_plan = f["q_reference_plan"].copy()
    base = f["base_slice"]
    q_plan[base.start : base.start + 3] += generator.uniform(-0.5, 0.5, 3)
    # Rotation vector well inside the chart (angle < pi).
    q_plan[base.start + 3 : base.stop] = generator.uniform(-1.0, 1.0, 3)
    check_embedder(f["embedder"], q_plan)
    # And the full-configuration side: re-embedding the extraction must
    # reproduce the same base pose (compare rotations as matrices -- the
    # quaternion double cover makes sign comparison meaningless).
    q_full = f["embedder"](q_plan)
    q_again = f["embedder"](f["embedder"].extract(q_full))
    assert np.max(np.abs(q_again - q_full)) < 1e-12


def test_T1_2b_non_planning_entries_preserved_free_flyer():
    """Hand entries survive embedding with the base in the planning space."""
    f = ff_ctx()
    model = f["model"]
    embedder = PlanningEmbedder(
        model, f["planning_joint_names"], f["q_nominal"]
    )
    q_full = embedder(f["q_reference_plan"])
    planned_q_idx = set(embedder.revolute_idx_q.tolist())
    base_joint = model.joints[
        model.getJointId(f["planning_joint_names"][0])
    ]
    planned_q_idx.update(
        range(base_joint.idx_q, base_joint.idx_q + base_joint.nq)
    )
    non_planning = [i for i in range(model.nq) if i not in planned_q_idx]
    assert non_planning, "expected hand entries outside the planning space"
    assert np.array_equal(
        q_full[non_planning], f["q_nominal"][non_planning]
    )


def test_T1_3_unknown_joint_raises_with_name():
    c = ctx()
    bad_name = "definitely_not_a_joint"
    try:
        PlanningEmbedder(
            c["model"],
            c["planning_joint_names"] + [bad_name],
            c["q_nominal"],
        )
    except KeyError as exc:
        assert bad_name in str(exc), f"message does not name joint: {exc}"
    else:
        raise AssertionError("unknown joint name did not raise")


def test_T1_4_multi_dof_joint_raises_with_name_and_nq():
    """Non-free-flyer multi-DoF joints still raise (Stage-2 update).

    Stage 2 admits the free-flyer as the ONE exception to the nq == 1
    rule, so the rejection path is exercised with a spherical joint; the
    free-flyer acceptance is asserted alongside on the real G1 model.
    """
    model = build_spherical_model()
    root_joint = model.joints[model.getJointId("spherical_root")]
    try:
        PlanningEmbedder(model, ["spherical_root"], pin.neutral(model))
    except ValueError as exc:
        message = str(exc)
        assert "spherical_root" in message, message
        assert str(root_joint.nq) in message, (
            f"message does not state nq={root_joint.nq}: {message}"
        )
    else:
        raise AssertionError("multi-DoF joint did not raise")
    # The free-flyer, by contrast, is admitted (6 planning slots).
    f = ff_ctx()
    base_only = PlanningEmbedder(
        f["model"], [f["planning_joint_names"][0]], f["q_nominal"]
    )
    assert base_only.n_plan == 6 and base_only.has_base


def test_T1_5_wrong_nominal_length_raises_with_both_lengths():
    c = ctx()
    bad = np.zeros(c["model"].nq + 3)
    try:
        PlanningEmbedder(c["model"], c["planning_joint_names"], bad)
    except ValueError as exc:
        message = str(exc)
        assert str(c["model"].nq) in message, message
        assert str(bad.shape[0]) in message, message
    else:
        raise AssertionError("wrong-length q_nominal did not raise")


def test_T1_6_floating_base_q_and_v_indices_differ():
    """idx_q != idx_v on the REAL free-flyer G1 (Stage-2 update).

    Stage 1 had to exercise this on a synthetic free-flyer model; the
    planning model is now genuinely floating-base, so the synthetic
    substitute is gone.
    """
    f = ff_ctx()
    model, embedder = f["model"], f["embedder"]
    idx_q = embedder.revolute_idx_q
    idx_v = np.array(
        [
            model.joints[model.getJointId(name)].idx_v
            for name in f["planning_joint_names"][1:]
        ]
    )
    assert not np.array_equal(idx_q, idx_v), (
        "idx_q == idx_v on a floating-base model: the quaternion offset "
        "has been missed and every Jacobian column selection downstream "
        "would be wrong"
    )
    # The free-flyer occupies 7 q-entries but 6 v-entries.
    assert np.all(idx_q - idx_v == 1)
    # select_columns must pick v-columns: mark each column with its index.
    # Expected: the base's 6 tangent columns first, then each revolute
    # joint's idx_v, in planning order.
    base_joint = model.joints[
        model.getJointId(f["planning_joint_names"][0])
    ]
    expected = np.concatenate(
        [np.arange(base_joint.idx_v, base_joint.idx_v + 6), idx_v]
    ).astype(float)
    J_full = np.tile(np.arange(model.nv, dtype=float), (6, 1))
    assert np.array_equal(
        embedder.select_columns(J_full), np.tile(expected, (6, 1))
    ), "select_columns did not select tangent-space (v) columns"
    # idx_q / idx_v are undefined with the base planned: must fail loudly.
    for attribute in ("idx_q", "idx_v"):
        try:
            getattr(embedder, attribute)
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                f"{attribute} should raise with the base in the planning "
                f"space"
            )


def test_T1_7_select_columns_shape():
    c = ctx()
    J_full = rng().standard_normal((6, c["model"].nv))
    selected = c["embedder"].select_columns(J_full)
    assert selected.shape == (6, c["embedder"].n_plan)
    assert np.array_equal(selected, J_full[:, c["embedder"].idx_v])


def test_T1_8_n_plan_matches_name_count():
    c = ctx()
    assert c["embedder"].n_plan == len(c["planning_joint_names"])


# ===========================================================================
# T2 -- numeric_jacobian itself (test the tester)
# ===========================================================================


def test_T2_1_numeric_jacobian_recovers_linear_map():
    generator = rng()
    A = generator.standard_normal((4, 7))
    constraint = _LinearConstraint(A)
    J = numeric_jacobian(constraint, generator.standard_normal(7))
    assert np.allclose(J, A, atol=1e-6), (
        f"numeric jacobian is broken; max err {np.max(np.abs(J - A))}"
    )


def test_T2_2_numeric_jacobian_matches_quadratic_gradient():
    generator = rng()
    dim = 5
    constraint = _QuadraticConstraint(
        Qs=[generator.standard_normal((dim, dim)) for _ in range(3)],
        bs=[generator.standard_normal(dim) for _ in range(3)],
    )
    for _ in range(3):
        q = generator.standard_normal(dim)
        J_numeric = numeric_jacobian(constraint, q)
        J_analytic = constraint.jacobian(q)
        # Central differences are exact for quadratics up to rounding.
        assert np.allclose(J_numeric, J_analytic, atol=1e-7), (
            f"max err {np.max(np.abs(J_numeric - J_analytic))}"
        )


def test_T2_3_is_satisfied_thresholds():
    constraint = _LinearConstraint(np.eye(2))
    tol = 1e-3
    inside = np.array([tol * 0.999, 0.0])
    outside = np.array([tol * 1.001, 0.0])
    assert constraint.is_satisfied(inside, tol=tol)
    assert not constraint.is_satisfied(outside, tol=tol)
    # Default tolerance is 1e-3 as well.
    assert constraint.is_satisfied(inside)
    assert not constraint.is_satisfied(outside)


# ===========================================================================
# T3 -- FeetConstraint
# ===========================================================================


def _feet_constraint() -> FeetConstraint:
    c = ctx()
    return FeetConstraint(c["robot_model"], c["embedder"], c["q0_plan"])


def test_T3_1_n_rows():
    assert _feet_constraint().n_rows == 12


def test_T3_2_zero_error_at_reference():
    c = ctx()
    feet = _feet_constraint()
    error = feet.error(c["q0_plan"])
    assert np.max(np.abs(error)) < 1e-9, (
        f"error at the reference config is not ~0: {error}"
    )


def test_T3_3_analytic_vs_numeric_jacobian():
    feet = _feet_constraint()
    generator = rng()
    c = ctx()
    max_discrepancy = 0.0
    for _ in range(5):
        q = np.clip(
            c["q0_plan"]
            + generator.uniform(-0.15, 0.15, c["embedder"].n_plan),
            c["lower"],
            c["upper"],
        )
        J_analytic = feet.jacobian(q)
        J_numeric = numeric_jacobian(feet, q)
        max_discrepancy = max(
            max_discrepancy, float(np.max(np.abs(J_analytic - J_numeric)))
        )
        assert np.allclose(J_analytic, J_numeric, atol=1e-5), (
            f"analytic/numeric mismatch: max abs diff "
            f"{np.max(np.abs(J_analytic - J_numeric))} at config "
            f"{_config_repr(q)}"
        )
    _NOTES["test_T3_3_analytic_vs_numeric_jacobian"] = (
        f"max analytic-vs-numeric discrepancy {max_discrepancy:.3e}"
    )
    print(f"    [T3.3] max analytic-vs-numeric discrepancy: "
          f"{max_discrepancy:.3e}")


def test_T3_4_kinematic_chain():
    """Leg joints move the feet error; arm joints must not."""
    c = ctx()
    names = c["planning_joint_names"]
    feet = _feet_constraint()
    q_leg = c["q0_plan"].copy()
    q_leg[names.index("left_knee_joint")] += 0.2
    error_leg = feet.error(q_leg)
    assert np.linalg.norm(error_leg) > 1e-3, (
        "perturbing a leg joint left the feet error at zero -- the "
        "constraint is not reading the kinematic chain"
    )
    q_arm = c["q0_plan"].copy()
    q_arm[names.index("left_elbow_joint")] += 0.4
    q_arm[names.index("right_shoulder_pitch_joint")] += 0.4
    error_arm = feet.error(q_arm)
    assert np.max(np.abs(error_arm)) < 1e-10, (
        f"perturbing arm joints changed the feet error: {error_arm}"
    )


def _ff_feet_constraint() -> FeetConstraint:
    f = ff_ctx()
    return FeetConstraint(
        f["robot_model"], f["embedder"], f["q_reference_plan"]
    )


def test_T3_5_base_translation_units_and_sign():
    """+1cm base x-translation moves the feet error by 0.01 (Stage-2).

    Now a REAL base perturbation of the free-flyer planning vector --
    Stage 1's target-mutation substitute is gone.  Base-only motion must
    move the feet error; joints are untouched.
    """
    f = ff_ctx()
    feet = _ff_feet_constraint()
    shift = np.array([0.01, 0.0, 0.0])
    q_perturbed = f["q_reference_plan"].copy()
    q_perturbed[f["base_slice"].start : f["base_slice"].start + 3] += shift
    error = feet.error(q_perturbed)
    targets = dict(zip(("left", "right"), feet.targets))
    for foot, rows in (("left", error[0:6]), ("right", error[6:12])):
        translational, rotational = rows[:3], rows[3:]
        assert abs(np.linalg.norm(translational) - 0.01) < 1e-8, (
            f"{foot}: translational magnitude "
            f"{np.linalg.norm(translational)} != 0.01"
        )
        # Sign: the error must point along +x expressed in the target frame.
        expected_direction = targets[foot].rotation.T @ shift
        assert translational @ expected_direction > 0, (
            f"{foot}: translational error has the wrong sign"
        )
        assert np.max(np.abs(rotational)) < 1e-12, (
            f"{foot}: pure base translation produced rotational error "
            f"{rotational}"
        )


def test_T3_6_base_rotation_row_ordering():
    """Small base yaw lands in the rotational rows (3:6 per foot; Stage-2).

    Now a REAL base perturbation (see T3.5).  Row ordering asserted (and
    implemented): per foot, rows 0-2 are TRANSLATIONAL and rows 3-5
    ROTATIONAL -- Pinocchio Motion.vector order ``[linear, angular]``.
    A base yaw also translates the feet slightly (they sit off the yaw
    axis, radius ~0.15 m), so the translational rows are small but not
    zero -- the ordering is proven by the rotational rows carrying the
    full angle while the translational rows stay an order of magnitude
    below it.
    """
    f = ff_ctx()
    feet = _ff_feet_constraint()
    angle = 0.01
    q_perturbed = f["q_reference_plan"].copy()
    q_perturbed[f["base_slice"].start + 5] += angle  # rotvec z at identity
    error = feet.error(q_perturbed)
    for foot, rows in (("left", error[0:6]), ("right", error[6:12])):
        translational, rotational = rows[:3], rows[3:]
        assert abs(np.linalg.norm(rotational) - angle) < 1e-6, (
            f"{foot}: rotational magnitude {np.linalg.norm(rotational)} "
            f"!= {angle}"
        )
        assert np.linalg.norm(translational) < 0.3 * angle, (
            f"{foot}: base yaw produced translational error "
            f"{translational} larger than the off-axis lever arm explains"
        )
        assert np.linalg.norm(rotational) > 3.0 * np.linalg.norm(
            translational
        ), f"{foot}: rotation did not land in rows 3:6 -- ordering broken"


def test_T3_7_jacobian_full_row_rank():
    feet = _feet_constraint()
    J = feet.jacobian(generic_config())
    rank = np.linalg.matrix_rank(J)
    assert rank == feet.n_rows, (
        f"feet Jacobian rank {rank} < {feet.n_rows}: duplicated or dead "
        f"columns in the index map"
    )


def test_T3_8_determinism():
    q = generic_config()
    feet = _feet_constraint()
    assert np.array_equal(feet.error(q), feet.error(q)), (
        "two error() calls differ: cached FK state is leaking between calls"
    )
    assert np.array_equal(feet.jacobian(q), feet.jacobian(q))


def test_T3_9_input_not_mutated():
    feet = _feet_constraint()
    q = generic_config()
    q_snapshot = q.copy()
    feet.error(q)
    feet.jacobian(q)
    assert np.array_equal(q, q_snapshot), "input array was mutated"


# ===========================================================================
# T4 -- CoMConstraint
# ===========================================================================


def _com_constraint() -> CoMConstraint:
    c = ctx()
    return CoMConstraint(c["robot_model"], c["embedder"], margin=0.05)


def test_T4_1_n_rows():
    assert _com_constraint().n_rows == 1


def test_T4_2_error_exactly_zero_when_stable():
    c = ctx()
    com = _com_constraint()
    assert com.margin_at(c["q0_plan"]) > com.margin, (
        "test premise broken: nominal config is not comfortably stable"
    )
    error = com.error(c["q0_plan"])
    assert error[0] == 0.0, (
        f"hinge must be EXACTLY 0.0 when stable, got {error[0]!r}"
    )


def test_T4_3_error_positive_when_unstable():
    com = _com_constraint()
    q_bad = hip_pitch_lean(0.9)
    assert com.margin_at(q_bad) < 0.0, (
        "test premise broken: lean config did not push the CoM outside "
        "the support polygon"
    )
    assert com.error(q_bad)[0] > 0.0


def test_T4_4_monotone_error_growth_outside():
    com = _com_constraint()
    scales = np.linspace(0.0, 0.9, 16)
    errors = [com.error(hip_pitch_lean(s))[0] for s in scales]
    violating = [e for e in errors if e > 0.0]
    assert len(violating) >= 4, (
        f"need >= 4 violating steps to test monotonicity, got "
        f"{len(violating)}; errors: {errors}"
    )
    deltas = np.diff(violating)
    assert np.all(deltas > 0.0), (
        f"error is not monotonically increasing outside the polygon: "
        f"{violating}"
    )


def test_T4_5_continuity_across_boundary():
    com = _com_constraint()
    scales = np.linspace(0.0, 0.9, 16)
    margins = [com.margin_at(hip_pitch_lean(s)) for s in scales]
    crossing = next(
        i for i in range(len(margins) - 1)
        if (margins[i] - com.margin) * (margins[i + 1] - com.margin) <= 0.0
    )
    dense = np.linspace(scales[crossing], scales[crossing + 1], 400)
    errors = np.array([com.error(hip_pitch_lean(s))[0] for s in dense])
    assert np.any(errors == 0.0) and np.any(errors > 0.0), (
        "dense sweep does not straddle the boundary"
    )
    max_jump = float(np.max(np.abs(np.diff(errors))))
    assert max_jump < 5e-3, (
        f"error jumps by {max_jump} across one dense step -- "
        f"discontinuous at the boundary"
    )


def test_T4_6_jacobian_zero_inside_nonzero_outside():
    c = ctx()
    com = _com_constraint()
    J_stable = com.jacobian(c["q0_plan"])
    assert J_stable.shape == (1, com.n_plan)
    assert np.all(J_stable == 0.0), (
        "subgradient inside the satisfied region must be exactly zero"
    )
    J_violating = com.jacobian(hip_pitch_lean(0.9))
    assert np.linalg.norm(J_violating) > 0.0, (
        "Jacobian is zero for a violating config"
    )


def test_T4_7_margin_at_equals_backend_stability():
    c = ctx()
    com = _com_constraint()
    backend = G1Constraints(c["robot_model"])
    for q in (c["q0_plan"], generic_config(), hip_pitch_lean(0.9)):
        assert com.margin_at(q) == backend.compute_stability(
            c["embedder"](q)
        )


def test_T4_8_is_stable_iff_error_zero():
    com = _com_constraint()
    for s in np.linspace(0.0, 0.9, 10):
        q = hip_pitch_lean(s)
        assert com.is_stable(q) == (com.error(q)[0] == 0.0), (
            f"is_stable and error==0 disagree at lean scale {s}"
        )


# ===========================================================================
# T5 -- ComposedConstraint
# ===========================================================================


def test_T5_1_n_rows_sum():
    assert ComposedConstraint(
        _feet_constraint(), _com_constraint()
    ).n_rows == 13


def test_T5_2_jacobian_shape():
    c = ctx()
    composed = ComposedConstraint(_feet_constraint(), _com_constraint())
    assert composed.jacobian(generic_config()).shape == (
        13,
        c["embedder"].n_plan,
    )


def test_T5_3_error_is_concatenation():
    feet, com = _feet_constraint(), _com_constraint()
    composed = ComposedConstraint(feet, com)
    for q in (generic_config(), hip_pitch_lean(0.9)):
        assert np.array_equal(
            composed.error(q),
            np.concatenate([feet.error(q), com.error(q)]),
        )


def test_T5_4_jacobian_is_vstack():
    feet, com = _feet_constraint(), _com_constraint()
    composed = ComposedConstraint(feet, com)
    for q in (generic_config(), hip_pitch_lean(0.9)):
        assert np.array_equal(
            composed.jacobian(q),
            np.vstack([feet.jacobian(q), com.jacobian(q)]),
        )


def test_T5_5_order_preserved():
    feet, com = _feet_constraint(), _com_constraint()
    q = hip_pitch_lean(0.9)  # both blocks non-zero here
    forward = ComposedConstraint(feet, com).error(q)
    swapped = ComposedConstraint(com, feet).error(q)
    assert np.array_equal(forward[: feet.n_rows], swapped[com.n_rows :])
    assert np.array_equal(forward[feet.n_rows :], swapped[: com.n_rows])


def test_T5_6_per_constraint_error_keys_and_shapes():
    feet, com = _feet_constraint(), _com_constraint()
    q = generic_config()
    report = ComposedConstraint(feet, com).per_constraint_error(q)
    assert set(report) == {"FeetConstraint", "CoMConstraint"}, (
        f"keys do not identify the constraints: {sorted(report)}"
    )
    assert report["FeetConstraint"].shape == (feet.n_rows,)
    assert report["CoMConstraint"].shape == (com.n_rows,)
    assert np.array_equal(report["FeetConstraint"], feet.error(q))
    # Duplicate classes must still get unique, identifying keys.
    duplicated = ComposedConstraint(feet, _feet_constraint())
    keys = list(duplicated.per_constraint_error(q))
    assert len(set(keys)) == 2, f"duplicate keys: {keys}"
    assert all("FeetConstraint" in key for key in keys)


def test_T5_7_n_plan_disagreement_raises_with_both_values():
    small = _LinearConstraint(np.eye(3))
    large = _LinearConstraint(np.eye(4))
    try:
        ComposedConstraint(small, large)
    except ValueError as exc:
        message = str(exc)
        assert "3" in message and "4" in message, (
            f"message does not name both n_plan values: {message}"
        )
    else:
        raise AssertionError("n_plan disagreement did not raise")


def test_T5_8_empty_composition_raises():
    """Documented choice: empty composition raises (n_plan undefined)."""
    try:
        ComposedConstraint()
    except ValueError:
        pass
    else:
        raise AssertionError("ComposedConstraint() did not raise")


# ===========================================================================
# T6 -- Cross-cutting contract
# ===========================================================================


def _contract_instances() -> list[Constraint]:
    return [
        _feet_constraint(),
        _com_constraint(),
        ComposedConstraint(_feet_constraint(), _com_constraint()),
    ]


def test_T6_1_error_is_1d():
    q = generic_config()
    for constraint in _contract_instances():
        error = constraint.error(q)
        assert error.shape == (constraint.n_rows,), (
            f"{type(constraint).__name__}: error shape {error.shape} "
            f"is not ({constraint.n_rows},)"
        )


def test_T6_2_error_dtype_float64():
    q = generic_config()
    for constraint in _contract_instances():
        assert constraint.error(q).dtype == np.float64, (
            type(constraint).__name__
        )


def test_T6_3_jacobian_shape():
    q = generic_config()
    for constraint in _contract_instances():
        J = constraint.jacobian(q)
        assert J.shape == (constraint.n_rows, constraint.n_plan), (
            f"{type(constraint).__name__}: jacobian shape {J.shape}"
        )


def test_T6_4_no_nan_inf_over_random_configs():
    c = ctx()
    generator = rng()
    constraints = _contract_instances()
    for trial in range(20):
        q = generator.uniform(c["lower"], c["upper"])
        for constraint in constraints:
            error = constraint.error(q)
            J = constraint.jacobian(q)
            assert np.all(np.isfinite(error)), (
                f"{type(constraint).__name__}: non-finite error at "
                f"trial {trial}, config {_config_repr(q)}"
            )
            assert np.all(np.isfinite(J)), (
                f"{type(constraint).__name__}: non-finite jacobian at "
                f"trial {trial}, config {_config_repr(q)}"
            )


def test_T6_5_accepts_plain_list():
    q_list = generic_config().tolist()
    assert isinstance(q_list, list)
    for constraint in _contract_instances():
        assert constraint.error(q_list).shape == (constraint.n_rows,)
        assert constraint.jacobian(q_list).shape == (
            constraint.n_rows,
            constraint.n_plan,
        )
        constraint.is_satisfied(q_list)


def test_T6_6_input_never_mutated():
    q = generic_config()
    snapshot = q.copy()
    for constraint in _contract_instances():
        constraint.error(q)
        constraint.jacobian(q)
        assert np.array_equal(q, snapshot), (
            f"{type(constraint).__name__} mutated its input"
        )


# ===========================================================================
# T7 -- Validation against the reference trajectory dataset
# ===========================================================================

_DATASET: dict | None = None

# Deterministic expansion of the dataset's abbreviated MuJoCo joint names to
# full Pinocchio joint names.  Explicit table -- if a name is missing the
# test stops rather than guessing.
_ABBREVIATIONS = {
    "sh_pitch": "shoulder_pitch",
    "sh_roll": "shoulder_roll",
    "sh_yaw": "shoulder_yaw",
    "wr_roll": "wrist_roll",
    "wr_pitch": "wrist_pitch",
    "wr_yaw": "wrist_yaw",
    "ankle_p": "ankle_pitch",
    "ankle_r": "ankle_roll",
}
_SIDES = {"L": "left", "R": "right"}


def _expand_dataset_joint_name(abbreviated: str) -> str:
    if abbreviated.startswith(("L_", "R_")):
        side = _SIDES[abbreviated[0]]
        rest = abbreviated[2:]
        rest = _ABBREVIATIONS.get(rest, rest)
        return f"{side}_{rest}_joint"
    return f"{abbreviated}_joint"


def dataset() -> dict:
    """Load, inspect and validate the dataset mapping ONCE (T7.0, T7.9).

    Raises :class:`SkipTest` when the file is absent so every T7 test skips
    with a clear message while T1-T6 run regardless.
    """
    global _DATASET
    if _DATASET is not None:
        return _DATASET
    if not os.path.exists(DATASET_PATH):
        raise SkipTest(
            f"dataset not found: {DATASET_PATH} (set "
            f"GEAR_SONIC_PLANNER_DATASET to override); T1-T6 are unaffected"
        )

    c = ctx()
    raw = np.load(DATASET_PATH, allow_pickle=True)
    print(f"    [T7.0] dataset {DATASET_PATH}")
    for key in raw.files:
        array = raw[key]
        print(f"    [T7.0]   {key}: shape {getattr(array, 'shape', None)} "
              f"dtype {array.dtype}")

    joint_pos = np.asarray(raw["joint_pos"], dtype=float)
    dataset_names = [str(n) for n in raw["joint_names"]]
    expanded = [_expand_dataset_joint_name(n) for n in dataset_names]

    print(f"    [T7.0] trajectory array: 'joint_pos', "
          f"{joint_pos.shape[0]} frames, width {joint_pos.shape[1]}")
    print(f"    [T7.0] n_plan = {c['embedder'].n_plan}; width matches: "
          f"{joint_pos.shape[1] == c['embedder'].n_plan}")
    print(f"    [T7.0] floating base: NOT in joint_pos; stored separately "
          f"as base_pos (xyz) + base_quat (wxyz, MuJoCo convention; "
          f"first component ~1 near identity)")
    print(f"    [T7.0] joint_order flag: {raw['joint_order']!r} "
          f"(MuJoCo ordering, abbreviated names)")

    # STOP conditions (spec T7.0): width mismatch or unmappable ordering.
    if joint_pos.shape[1] != c["embedder"].n_plan:
        raise AssertionError(
            f"STOP: dataset width {joint_pos.shape[1]} != n_plan "
            f"{c['embedder'].n_plan}; refusing to invent a mapping"
        )
    if expanded != c["planning_joint_names"]:
        mismatches = [
            (i, e, p)
            for i, (e, p) in enumerate(
                zip(expanded, c["planning_joint_names"])
            )
            if e != p
        ]
        raise AssertionError(
            f"STOP: dataset joint ordering does not match the planning "
            f"space; first mismatches: {mismatches[:5]}"
        )
    print(f"    [T7.0] expanded dataset names match planning_joint_names "
          f"1:1 in order -- ordering established, no reordering applied")

    _DATASET = {
        "joint_pos": joint_pos,
        "base_pos": np.asarray(raw["base_pos"], dtype=float),
        "base_quat": np.asarray(raw["base_quat"], dtype=float),
        "n_frames": joint_pos.shape[0],
    }
    return _DATASET


def test_T7_0_inspect_dataset():
    dataset()  # all inspection, printing and STOP checks happen inside


def test_T7_1_frames_finite_and_within_limits():
    c = ctx()
    data = dataset()
    joint_pos = data["joint_pos"]
    assert np.all(np.isfinite(joint_pos)), "dataset contains NaN/Inf"
    assert np.all(np.isfinite(data["base_pos"]))
    assert np.all(np.isfinite(data["base_quat"]))
    # float32 storage -> allow a tiny tolerance; report violators, no clipping.
    tolerance = 1e-6
    below = joint_pos < (c["lower"] - tolerance)
    above = joint_pos > (c["upper"] + tolerance)
    violating_frames = np.where(np.any(below | above, axis=1))[0]
    if violating_frames.size:
        for frame in violating_frames[:10]:
            joints = np.where((below | above)[frame])[0]
            details = [
                (c["planning_joint_names"][j], float(joint_pos[frame, j]),
                 float(c["lower"][j]), float(c["upper"][j]))
                for j in joints
            ]
            print(f"    [T7.1] frame {frame} violates limits: {details}")
        print(f"    [T7.1] total violating frames: {violating_frames.size} "
              f"(range {violating_frames.min()}..{violating_frames.max()})")
    assert violating_frames.size == 0, (
        f"{violating_frames.size} frames violate joint limits: "
        f"{violating_frames.tolist()[:20]}"
    )


def _select_pairs() -> list[tuple[int, int]]:
    """10 seeded (start, goal) pairs: ends, 4 adjacent, 5 distant."""
    data = dataset()
    n_frames = data["n_frames"]
    generator = np.random.default_rng(SEED)
    pairs = [(0, n_frames - 1)]
    for start in sorted(generator.integers(0, n_frames - 1, size=4)):
        pairs.append((int(start), int(start) + 1))
    minimum_gap = n_frames // 3
    while len(pairs) < 10:
        start = int(generator.integers(0, n_frames - minimum_gap - 1))
        goal = int(generator.integers(start + minimum_gap, n_frames))
        pairs.append((start, goal))
    return pairs


def test_T7_2_pair_selection_reproducible():
    pairs = _select_pairs()
    print(f"    [T7.2] seed {SEED}; selected (start, goal) pairs: {pairs}")
    data = dataset()
    assert len(pairs) == 10
    assert pairs[0] == (0, data["n_frames"] - 1)
    for start, goal in pairs[1:5]:
        assert goal == start + 1
    for start, goal in pairs[5:]:
        assert goal - start >= data["n_frames"] // 3
    assert pairs == _select_pairs(), "pair selection is not reproducible"


def test_T7_3_feet_constraint_vs_dataset():
    """Feet targets from START frame, error at start (~0) and at goal.

    Case (b) -- the demo moving its feet relative to the pelvis -- is a
    legitimate FINDING, not a test failure; tolerances are not loosened to
    hide it.  A world-frame diagnostic (base_pos/base_quat composed with
    pelvis-frame FK) distinguishes 'feet planted in the world while the
    base moves' from 'feet actually stepping'.
    """
    c = ctx()
    data = dataset()
    joint_pos = data["joint_pos"]
    pairs = _select_pairs()

    goal_errors = {}
    for start, goal in pairs:
        feet = FeetConstraint(
            c["robot_model"], c["embedder"], joint_pos[start]
        )
        error_start = feet.error(joint_pos[start])
        assert np.max(np.abs(error_start)) < 1e-9, (
            f"pair ({start}, {goal}): error at the target-defining start "
            f"frame is {np.max(np.abs(error_start))} -- definitional "
            f"failure"
        )
        error_goal = feet.error(joint_pos[goal])
        goal_errors[(start, goal)] = error_goal
        left, right = error_goal[0:6], error_goal[6:12]
        print(
            f"    [T7.3] pair ({start:3d}, {goal:3d}): |e_goal| "
            f"L trans {np.linalg.norm(left[:3]):.4f} "
            f"rot {np.linalg.norm(left[3:]):.4f} | "
            f"R trans {np.linalg.norm(right[:3]):.4f} "
            f"rot {np.linalg.norm(right[3:]):.4f}"
        )

    # World-frame diagnostic: are the feet planted in the WORLD?
    backend = G1Constraints(c["robot_model"])
    world_positions = {"left": [], "right": []}
    for frame in range(0, data["n_frames"], 10):
        base_rotation = pin.Quaternion(
            *data["base_quat"][frame]  # (w, x, y, z) -- MuJoCo convention
        ).matrix()
        base_pose = pin.SE3(base_rotation, data["base_pos"][frame])
        left, right = backend.compute_feet_poses(
            c["embedder"](joint_pos[frame])
        )
        world_positions["left"].append((base_pose * left).translation.copy())
        world_positions["right"].append(
            (base_pose * right).translation.copy()
        )
    drift = {
        foot: float(
            np.max(
                np.linalg.norm(np.asarray(positions) - positions[0], axis=1)
            )
        )
        for foot, positions in world_positions.items()
    }
    print(f"    [T7.3] world-frame foot drift over trajectory "
          f"(max vs frame 0): {drift}")

    max_goal_error = max(
        float(np.max(np.abs(e))) for e in goal_errors.values()
    )
    if max_goal_error < 1e-3:
        verdict = (
            "case (a): goal error ~0 for all pairs -- the demo keeps the "
            "feet planted; the feet constraint is consistent with this data"
        )
    else:
        verdict = (
            f"case (b): the demo shifts the feet relative to the pelvis "
            f"(max |e_goal| component {max_goal_error:.4f} rad|m), so a "
            f"fixed-feet constraint cannot represent this motion as-is; "
            f"world drift {drift} shows the feet stay planted in the WORLD "
            f"while the BASE moves -- a fixed-base-model artifact, not "
            f"actual stepping"
        )
    print(f"    [T7.3] {verdict}")
    _NOTES["test_T7_3_feet_constraint_vs_dataset"] = verdict.split(":")[0]


def test_T7_3b_feet_constraint_free_flyer():
    """T7.3 re-run on the FREE-FLYER stack (Stage 2, A.3).

    With the base in the planning space the feet error is measured in the
    WORLD frame, so the base motion that dominated the fixed-base numbers
    drops out.  Numbers are reported for comparison with T7.3 -- no
    pass/fail threshold is asserted, per the stage instructions.  The
    analytic-vs-numeric Jacobian gap is also reported: with the base
    planned, the base block is the local-twist Jacobian, exact only at
    base identity (see embedding.py), so a non-zero gap here is the
    documented approximation, not a bug.
    """
    f = ff_ctx()
    data = dataset()
    pairs = _select_pairs()

    worst_goal_error = 0.0
    for start, goal in pairs:
        q_start = ff_dataset_frame_to_q_plan(data, start)
        q_goal = ff_dataset_frame_to_q_plan(data, goal)
        feet = FeetConstraint(f["robot_model"], f["embedder"], q_start)
        error_start = feet.error(q_start)
        assert np.max(np.abs(error_start)) < 1e-9, (
            f"pair ({start}, {goal}): free-flyer error at the "
            f"target-defining start frame is "
            f"{np.max(np.abs(error_start))} -- definitional failure"
        )
        error_goal = feet.error(q_goal)
        worst_goal_error = max(
            worst_goal_error, float(np.max(np.abs(error_goal)))
        )
        left, right = error_goal[0:6], error_goal[6:12]
        print(
            f"    [T7.3b] pair ({start:3d}, {goal:3d}): |e_goal| "
            f"L trans {np.linalg.norm(left[:3]):.4f} "
            f"rot {np.linalg.norm(left[3:]):.4f} | "
            f"R trans {np.linalg.norm(right[:3]):.4f} "
            f"rot {np.linalg.norm(right[3:]):.4f}"
        )

    # Documented base-block Jacobian approximation, quantified on real
    # frames (report only).
    feet = FeetConstraint(
        f["robot_model"],
        f["embedder"],
        ff_dataset_frame_to_q_plan(data, 0),
    )
    jacobian_gap = 0.0
    for frame in np.linspace(0, data["n_frames"] - 1, 5).astype(int):
        q = ff_dataset_frame_to_q_plan(data, int(frame))
        jacobian_gap = max(
            jacobian_gap,
            float(
                np.max(np.abs(feet.jacobian(q) - numeric_jacobian(feet, q)))
            ),
        )
    print(f"    [T7.3b] max free-flyer goal-error component across pairs: "
          f"{worst_goal_error:.4f} (fixed-base T7.3 reached ~0.77)")
    print(f"    [T7.3b] analytic-vs-numeric Jacobian gap on real frames "
          f"(base-block chart approximation): {jacobian_gap:.3e}")
    _NOTES["test_T7_3b_feet_constraint_free_flyer"] = (
        f"max |e_goal| {worst_goal_error:.4f} (was ~0.77 fixed-base); "
        f"jac gap {jacobian_gap:.2e}"
    )


def test_T7_4_feet_error_over_whole_trajectory():
    """Targets from frame 0; 12-row error norm tabulated over all frames."""
    c = ctx()
    data = dataset()
    joint_pos = data["joint_pos"]
    feet = FeetConstraint(c["robot_model"], c["embedder"], joint_pos[0])
    norms = np.array(
        [
            np.linalg.norm(feet.error(joint_pos[frame]))
            for frame in range(data["n_frames"])
        ]
    )
    assert np.all(np.isfinite(norms))
    worst = int(np.argmax(norms))
    print(f"    [T7.4] feet error norm vs frame-0 targets: "
          f"max {norms.max():.4f} at frame {worst}, mean {norms.mean():.4f}")
    quarters = np.array_split(np.arange(data["n_frames"]), 4)
    for k, idx in enumerate(quarters):
        print(f"    [T7.4]   quarter {k} (frames {idx[0]}-{idx[-1]}): "
              f"max {norms[idx].max():.4f}, mean {norms[idx].mean():.4f}")
    _NOTES["test_T7_4_feet_error_over_whole_trajectory"] = (
        f"max {norms.max():.4f} @ frame {worst}, mean {norms.mean():.4f}"
    )


def test_T7_5_com_stability_over_whole_trajectory():
    """CoM margin on every frame; a human demo should be mostly stable."""
    c = ctx()
    data = dataset()
    joint_pos = data["joint_pos"]
    com = _com_constraint()
    margins = np.array(
        [
            com.margin_at(joint_pos[frame])
            for frame in range(data["n_frames"])
        ]
    )
    errors = np.maximum(0.0, com.margin - margins)
    stable = int(np.sum(errors == 0.0))
    violating = int(np.sum(errors > 0.0))
    fraction = stable / data["n_frames"]
    print(f"    [T7.5] margin required: {com.margin}; stable frames "
          f"(error == 0.0): {stable}/{data['n_frames']} "
          f"({100.0 * fraction:.1f}%), violating: {violating}")
    worst = np.argsort(margins)[:5]
    for frame in worst:
        print(f"    [T7.5]   worst frame {int(frame)}: margin "
              f"{margins[frame]:.4f} (error {errors[frame]:.4f})")
    if fraction < 0.9:
        print("    [T7.5] NOTE: a substantial share of demo frames is not "
              "CoM-stable under the fixed-base pelvis-frame model -- "
              "information about the data, reported as-is")
    _NOTES["test_T7_5_com_stability_over_whole_trajectory"] = (
        f"{stable}/{data['n_frames']} stable "
        f"({100.0 * fraction:.1f}%), worst margin {margins.min():.4f} "
        f"@ frame {int(np.argmin(margins))}"
    )


def test_T7_6_jacobian_on_real_configs():
    """Analytic vs numeric feet Jacobian at 20 frames across the demo.

    Real configurations reach near-singular poses that perturbations
    around nominal do not -- the test most likely to expose a frame-
    convention error.
    """
    c = ctx()
    data = dataset()
    joint_pos = data["joint_pos"]
    frames = np.unique(
        np.linspace(0, data["n_frames"] - 1, 20).astype(int)
    )
    feet = FeetConstraint(c["robot_model"], c["embedder"], joint_pos[0])
    max_discrepancy, worst_frame = 0.0, -1
    for frame in frames:
        q = joint_pos[frame]
        J_analytic = feet.jacobian(q)
        J_numeric = numeric_jacobian(feet, q)
        discrepancy = float(np.max(np.abs(J_analytic - J_numeric)))
        if discrepancy > max_discrepancy:
            max_discrepancy, worst_frame = discrepancy, int(frame)
        assert np.allclose(J_analytic, J_numeric, atol=1e-5), (
            f"analytic/numeric mismatch {discrepancy} at frame {frame}, "
            f"config {_config_repr(q)}"
        )
    print(f"    [T7.6] {len(frames)} real frames checked; max discrepancy "
          f"{max_discrepancy:.3e} at frame {worst_frame}")
    _NOTES["test_T7_6_jacobian_on_real_configs"] = (
        f"max discrepancy {max_discrepancy:.3e} @ frame {worst_frame}"
    )


def test_T7_7_composed_over_whole_trajectory():
    """Composed(feet, com) on every frame: shape, finiteness, dominance."""
    c = ctx()
    data = dataset()
    joint_pos = data["joint_pos"]
    composed = ComposedConstraint(
        FeetConstraint(c["robot_model"], c["embedder"], joint_pos[0]),
        _com_constraint(),
    )
    dominant = []
    for frame in range(data["n_frames"]):
        error = composed.error(joint_pos[frame])
        assert error.shape == (13,), f"frame {frame}: shape {error.shape}"
        assert np.all(np.isfinite(error)), f"frame {frame}: non-finite"
        per = composed.per_constraint_error(joint_pos[frame])
        feet_norm = np.linalg.norm(per["FeetConstraint"])
        com_norm = np.linalg.norm(per["CoMConstraint"])
        dominant.append("feet" if feet_norm >= com_norm else "com")
    # Run-length summary of per-frame dominance (299 raw lines is noise).
    runs, start_frame = [], 0
    for frame in range(1, len(dominant) + 1):
        if frame == len(dominant) or dominant[frame] != dominant[start_frame]:
            runs.append((dominant[start_frame], start_frame, frame - 1))
            start_frame = frame
    summary = ", ".join(
        f"{label}: {lo}-{hi}" for label, lo, hi in runs
    )
    print(f"    [T7.7] per-frame dominant constraint (run-length): "
          f"{summary}")
    counts = {label: dominant.count(label) for label in ("feet", "com")}
    print(f"    [T7.7] dominance counts: {counts}")
    _NOTES["test_T7_7_composed_over_whole_trajectory"] = (
        f"dominance {counts}"
    )


def test_T7_8_straight_line_interpolation_baseline():
    """Max composed error along naive straight lines -- Stage-1 baseline.

    This is exactly the quantity Stage 2's projection has to correct;
    large values are expected and are the motivation for projection.
    """
    c = ctx()
    data = dataset()
    joint_pos = data["joint_pos"]
    pairs = _select_pairs()
    com = _com_constraint()
    baseline = {}
    for start, goal in pairs:
        feet = FeetConstraint(
            c["robot_model"], c["embedder"], joint_pos[start]
        )
        composed = ComposedConstraint(feet, com)
        # 20 interior interpolation points (endpoints excluded).
        max_norm = 0.0
        for t in np.linspace(0.0, 1.0, 22)[1:-1]:
            q = (1.0 - t) * joint_pos[start] + t * joint_pos[goal]
            error = composed.error(q)
            assert np.all(np.isfinite(error))
            max_norm = max(max_norm, float(np.linalg.norm(error)))
        baseline[(start, goal)] = max_norm
        print(f"    [T7.8] pair ({start:3d}, {goal:3d}): max composed "
              f"error along straight line = {max_norm:.4f}")
    overall = max(baseline.values())
    print(f"    [T7.8] Stage-1 baseline (overall max): {overall:.4f}")
    _NOTES["test_T7_8_straight_line_interpolation_baseline"] = (
        f"overall max {overall:.4f}"
    )


def test_T7_9_dataset_path_override_and_skip():
    """DATASET_PATH is a module constant with an env override, and T7
    skips cleanly when the file is absent (exercised here directly)."""
    assert isinstance(DATASET_PATH, str)
    # The skip path must raise SkipTest with an actionable message.
    global _DATASET
    saved_dataset, saved_path = _DATASET, DATASET_PATH
    try:
        _DATASET = None
        globals()["DATASET_PATH"] = "/nonexistent/definitely_missing.npz"
        try:
            dataset()
        except SkipTest as exc:
            assert "GEAR_SONIC_PLANNER_DATASET" in str(exc)
        else:
            raise AssertionError(
                "missing dataset did not raise SkipTest"
            )
    finally:
        globals()["DATASET_PATH"] = saved_path
        _DATASET = saved_dataset


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> int:
    tests = [
        (name, fn)
        for name, fn in list(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    print(f"Running {len(tests)} tests (SEED = {SEED})\n")
    results: list[tuple[str, str]] = []
    for name, fn in tests:
        try:
            fn()
        except SkipTest as exc:
            results.append((name, "SKIP"))
            print(f"SKIP  {name}: {exc}")
        except Exception:
            results.append((name, "FAIL"))
            print(f"FAIL  {name}   (SEED = {SEED})")
            traceback.print_exc()
        else:
            results.append((name, "PASS"))
            print(f"PASS  {name}")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for name, status in results:
        note = _NOTES.get(name, "")
        print(f"  {status:<5} {name:<50} {note}")
    passed = sum(1 for _, s in results if s == "PASS")
    skipped = sum(1 for _, s in results if s == "SKIP")
    failures = [n for n, s in results if s == "FAIL"]
    print(f"\n{passed} passed, {skipped} skipped, {len(failures)} failed "
          f"of {len(results)}")
    if failures:
        print(f"FAILED: {failures}  (SEED = {SEED})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

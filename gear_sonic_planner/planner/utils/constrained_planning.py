from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin

import ompl.base as ob
import ompl.geometric as og
import ompl.util as ou

from decoupled_wbc.control.robot_model.robot_model import RobotModel
from gear_sonic_planner.planner.constraints.com_constraint import (
    CoMConstraint,
)
from gear_sonic_planner.planner.constraints.embedding import PlanningEmbedder
from gear_sonic_planner.planner.constraints.feet_constraint import (
    FeetConstraint,
)

# Base planning-slot bounds (see ConstrainedOMPLPlanner.set_up_ompl): the
# free-flyer has no meaningful position limits in the URDF, so the base
# position is bounded to a workspace box around the reference configuration,
# and the rotation-vector slots to +-pi (the chart's validity region).
BASE_POSITION_HALF_RANGE = 1.0


class FeetManifoldConstraint(ob.Constraint):
    """OMPL constraint adapter over the Stage-1 FeetConstraint.

    Only the FEET equality constraint defines the manifold: co-dimension 12,
    every row an equality cutting one dimension, which is what
    ProjectedStateSpace's manifold-dimension bookkeeping
    (ambient - co-dimension) assumes.  The CoM hinge is NOT included here --
    its satisfied set is full-dimensional (a region, not a surface), so
    declaring it as a constraint row would corrupt that bookkeeping; it
    lives in the validity checker instead.

    Tolerance defaults to 1e-3 rad|m rather than OMPL's 1e-4: every Newton
    iteration of the projection runs a Python FK + Jacobian round trip, and
    the extra order of magnitude roughly doubles the iterations for accuracy
    far below what the downstream controller tracks.
    """

    def __init__(
        self,
        feet_constraint: FeetConstraint,
        tolerance: float = 1e-3,
    ):
        super().__init__(feet_constraint.n_plan, feet_constraint.n_rows)
        self.feet_constraint = feet_constraint
        self.setTolerance(tolerance)

    def function(self, x, out):
        """Fill ``out`` with the 12-row feet pose error at ``x``."""
        out[:] = self.feet_constraint.error(np.asarray(x, dtype=float))

    def jacobian(self, x, out):
        """Fill ``out`` with the (12, n_plan) feet Jacobian at ``x``."""
        J = self.feet_constraint.jacobian(np.asarray(x, dtype=float))
        for row in range(J.shape[0]):
            out[row][:] = J[row]


class ProjectingStateSampler(ob.StateSampler):
    """Ambient-uniform sampler that projects onto the feet manifold.

    Workaround for this OMPL build: stock ``ProjectedStateSpace`` samplers
    project every sample onto the constraint manifold
    (``ProjectedStateSampler`` wraps the ambient sampler with
    ``constraint->project``), but the build's sampler returns raw ambient
    samples (measured feet errors of 2-5 on samples that should satisfy
    1e-3).  Off-manifold sample targets make every tree extension's
    discrete geodesic abort at the first step, so planners create no
    states at all.  Installing this sampler on the AMBIENT space restores
    the stock semantics: uniform in bounds, then projected.

    This is NOT a reference-biased sampler -- no trajectory data is
    involved; it reimplements the neutral upstream behavior.
    """

    def __init__(self, space, constraint, lower, upper, seed=None):
        super().__init__(space)
        self.constraint = constraint
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.n_dof = self.lower.shape[0]
        self.rng = np.random.default_rng(seed)

    def _write(self, state, x):
        # Planner-side states arrive as the constrained wrapper type
        # (bulk-writable via copy()); ambient RealVector states have no
        # copy() and are written elementwise.
        try:
            state.copy(x)
        except (AttributeError, TypeError):
            for i in range(self.n_dof):
                state[i] = float(x[i])

    def sampleUniform(self, state):
        x = self.rng.uniform(self.lower, self.upper)
        for _ in range(100):
            if self.constraint.project(x):
                break
            x = self.rng.uniform(self.lower, self.upper)
        self._write(state, np.clip(x, self.lower, self.upper))

    def sampleUniformNear(self, state, near, distance):
        center = np.array([near[i] for i in range(self.n_dof)])
        x = center + self.rng.uniform(-distance, distance, self.n_dof)
        self.constraint.project(x)
        self._write(state, np.clip(x, self.lower, self.upper))

    def sampleGaussian(self, state, mean, stdDev):
        center = np.array([mean[i] for i in range(self.n_dof)])
        x = center + self.rng.normal(0.0, stdDev, self.n_dof)
        self.constraint.project(x)
        self._write(state, np.clip(x, self.lower, self.upper))


class ConstrainedOMPLPlanner:
    """
    OMPL constrained geometric planner for whole-body paths.

    Plans in a ProjectedStateSpace whose manifold is the Stage-1 feet
    constraint (both feet pinned to their reference poses); static stability
    (the CoM hinge) is enforced in the validity checker.  The planning model
    is built from the URDF with a free-flyer root, so the base pose is part
    of the planning space (position + rotation vector; see
    constraints/embedding.py for the representation).

    No optimization objective is set: this is pure constrained motion
    planning, start -> goal.

    Two workarounds for defects in this OMPL build are active and loudly
    documented: ProjectingStateSampler (see its docstring) and the
    degenerate-solution-path recovery in plan() -- the build corrupts
    constrained solution paths on readout (every stored state collapses to
    the goal; C++ path.length() itself reports 0 for a solved 1.3-rad
    problem), so a zero-length "exact" path between distinct endpoints is
    re-materialized from the space's own geodesic when, and only when, the
    direct motion is verifiably valid.
    """

    def __init__(
        self,
        robot_model,
        urdf_path: str,
        planning_joint_names: list[str],
        q_nominal: np.ndarray,
        q_reference: np.ndarray,
        planner: str = "AORRTC",
        validity_resolution: float = 0.01,
        extend_range: float | None = None,
        com_margin: float = 0.05,
        projection_delta: float = 0.05,
        projection_lambda: float = 2.0,
        log: bool = True,
    ):
        """Initialize Planner"""
        # Caller's robot model, kept for parity with OMPLGeometricPlanner's
        # `robot` and for the (currently disabled) collision hookup; all
        # kinematics below run on the free-flyer planning model.
        self.robot_model = robot_model

        # Planning model: URDF with a free-flyer root (A.1).  RobotModel
        # forwards root_joint=pin.JointModelFreeFlyer() to BuildFromURDF and
        # loads the collision geometry the CoM support polygon needs.
        self.planning_robot_model = RobotModel(
            urdf_path,
            str(Path(urdf_path).resolve().parent),
            set_floating_base=True,
        )
        self.pin_model = self.planning_robot_model.pinocchio_wrapper.model

        self.embedder = PlanningEmbedder(
            self.pin_model, planning_joint_names, q_nominal
        )
        self.n_dof = self.embedder.n_plan

        q_reference = np.asarray(q_reference, dtype=float).reshape(-1)
        if q_reference.shape[0] != self.n_dof:
            raise ValueError(
                f"q_reference has {q_reference.shape[0]} DoF, expected "
                f"{self.n_dof}"
            )
        self.q_reference = q_reference.copy()

        # Constraints: feet define the manifold, the CoM hinge gates
        # validity (see FeetManifoldConstraint for why they are split).
        self.feet_constraint = FeetConstraint(
            self.planning_robot_model, self.embedder, self.q_reference
        )
        self.com_constraint = CoMConstraint(
            self.planning_robot_model, self.embedder, margin=com_margin
        )

        # Set up OMPL planner
        self.planner_name = planner
        self.validity_resolution = validity_resolution
        self.projection_delta = projection_delta
        self.projection_lambda = projection_lambda
        self.ss, self.si = self.set_up_ompl()
        self.pdef = self.ss.getProblemDefinition()
        self.planner = self.ss.getPlanner()
        if extend_range is not None:
            self.planner.setRange(extend_range)

        self.log = log
        if not log:
            ou.setLogLevel(ou.LOG_ERROR)

        # Filled by plan(): projection corrections and constraint drift.
        self.last_plan_stats: dict = {}

    def set_up_ompl(self):
        """Setup OMPL constrained planner"""
        # Ambient space over the planning joints
        space = ob.RealVectorStateSpace(self.n_dof)
        bounds = ob.RealVectorBounds(self.n_dof)
        low = np.empty(self.n_dof)
        high = np.empty(self.n_dof)
        # 1-DoF joints: URDF limits, with the same +-2pi fallback for
        # infinite limits as OMPLGeometricPlanner.set_up_ompl.
        revolute_positions = self.embedder.revolute_plan_positions
        revolute_idx_q = self.embedder.revolute_idx_q
        low[revolute_positions] = self.pin_model.lowerPositionLimit[
            revolute_idx_q
        ]
        high[revolute_positions] = self.pin_model.upperPositionLimit[
            revolute_idx_q
        ]
        # Free-flyer slots: the URDF gives the base no meaningful limits, so
        # bound the position to a workspace box around the reference config
        # and the rotation vector to +-pi (the chart's validity region).
        base = self.embedder.base_plan_slice
        if base is not None:
            base_position = slice(base.start, base.start + 3)
            base_rotation = slice(base.start + 3, base.stop)
            low[base_position] = (
                self.q_reference[base_position] - BASE_POSITION_HALF_RANGE
            )
            high[base_position] = (
                self.q_reference[base_position] + BASE_POSITION_HALF_RANGE
            )
            low[base_rotation] = -np.pi
            high[base_rotation] = np.pi
        for i in range(self.n_dof):
            # in case limit is infinite, set to -2pi, 2pi
            if low[i] == -np.inf:
                low[i] = -2 * np.pi
            if high[i] == np.inf:
                high[i] = 2 * np.pi
            bounds.setLow(i, float(low[i]))
            bounds.setHigh(i, float(high[i]))
        space.setBounds(bounds)
        self._bounds_low = low.copy()
        self._bounds_high = high.copy()

        # Constrained space: project onto the feet manifold
        self.constraint = FeetManifoldConstraint(self.feet_constraint)
        # Build-bug workaround: make the ambient sampler project (see
        # ProjectingStateSampler).  The allocator must outlive the space,
        # hence the attribute reference.
        self._sampler_allocator = lambda sampler_space: ProjectingStateSampler(
            sampler_space, self.constraint, self._bounds_low, self._bounds_high
        )
        space.setStateSamplerAllocator(self._sampler_allocator)
        self.constrained_space = ob.ProjectedStateSpace(
            space, self.constraint
        )
        # Delta is the geodesic step size; lambda caps geodesic length
        # relative to ambient distance before a motion is rejected.
        self.constrained_space.setDelta(self.projection_delta)
        self.constrained_space.setLambda(self.projection_lambda)
        csi = ob.ConstrainedSpaceInformation(self.constrained_space)

        # Simple Setup
        ss = og.SimpleSetup(csi)
        csi.setStateValidityCheckingResolution(self.validity_resolution)
        ss.setStateValidityChecker(self.validity_checker)
        # Deliberately no optimization objective: pure constrained motion
        # planning (similarity/optimization is out of scope in this stage).

        # Set planner
        planner = getattr(og, self.planner_name)(csi)
        ss.setPlanner(planner)
        return ss, csi

    def validity_checker(self, state: ob.State):
        """Check if the state is valid

        Bounds, static stability (the CoM hinge lives here, not in the
        manifold constraint), and -- once re-enabled -- collision.
        """
        # The nanobind layer hands the checker the constrained state type
        # directly, and indexing reads the wrapped Eigen map.  (In C++ this
        # is the spot where as<ConstrainedStateSpace::StateType>() is
        # mandatory: as<RealVectorStateSpace::StateType> is an unchecked
        # static_cast in release builds and silently reinterprets memory.)
        q = np.array([state[i] for i in range(self.n_dof)], dtype=float)

        # TEMPORARILY DISABLED FOR SIMULATION TESTING ONLY.
        # Restore this before real-robot deployment; otherwise OMPL may return
        # paths that collide with the robot or the environment.
        # (carried over, still disabled, from ompl_planning.py's
        # validity_checker: in_contact = self.robot.in_contact())
        in_contact = False
        # Check if in bounds
        in_bounds = self.si.satisfiesBounds(state)
        return in_bounds and not in_contact and self.com_constraint.is_stable(q)

    def _project_configuration(
        self, q_plan: np.ndarray, label: str
    ) -> tuple[np.ndarray, float]:
        """Project a configuration onto the feet manifold.

        Returns the projected configuration and the correction magnitude.
        Raises if the projection fails: rooting the tree off the manifold
        makes every subsequent geodesic invalid.
        """
        projected = q_plan.copy()
        if not self.constraint.project(projected):
            raise RuntimeError(
                f"Failed to project the {label} configuration onto the "
                f"feet manifold (residual "
                f"{np.linalg.norm(self.feet_constraint.error(projected)):.3e})"
            )
        return projected, float(np.linalg.norm(projected - q_plan))

    def _extract_path_states(self, path) -> np.ndarray:
        """Path states as an (N, n_dof) array."""
        return np.array(
            [
                [s[i] for i in range(self.n_dof)]
                for s in path.getStates()
            ],
            dtype=float,
        )

    def _max_feet_error_over(self, waypoints: np.ndarray) -> float:
        """Max feet-error norm over waypoint rows."""
        return max(
            float(np.linalg.norm(self.feet_constraint.error(q)))
            for q in waypoints
        )

    def _recover_degenerate_path(
        self,
        start_state,
        goal_state,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> np.ndarray:
        """Re-materialize a solution the build corrupted on readout.

        This OMPL build returns constrained solution paths whose stored
        states have all collapsed to the goal (C++ path.length() == 0 for
        a solved problem with distinct endpoints).  When that happens the
        route information is lost, but for a single-connect solution the
        path IS the manifold geodesic between the endpoints -- so it is
        rebuilt here from the space's own interpolate(), guarded by an
        explicit checkMotion() so a corrupt multi-vertex solution cannot
        be silently replaced by an invalid straight connect.
        """
        if not self.si.checkMotion(start_state, goal_state):
            raise RuntimeError(
                "The solver returned a corrupted (zero-length) solution "
                "path and the direct start-goal geodesic is not valid, so "
                "the actual route cannot be recovered from this OMPL "
                "build's output"
            )
        distance = float(self.si.distance(start_state, goal_state))
        n_segments = max(2, int(np.ceil(distance / self.projection_delta)))
        interpolated = self.si.allocState()
        waypoints = [start.copy()]
        for t in np.linspace(0.0, 1.0, n_segments + 1)[1:-1]:
            self.constrained_space.interpolate(
                start_state, goal_state, float(t), interpolated
            )
            waypoints.append(
                np.array(
                    [interpolated[i] for i in range(self.n_dof)],
                    dtype=float,
                )
            )
        waypoints.append(goal.copy())
        return np.array(waypoints)

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        goal_type: str = "whole_body",
        timeout: float = 10.0,
        smooth_path: bool = True,
        shortcut_path: bool = True,
    ) -> np.ndarray:
        """Plan a constrained path from start to goal.

        Only "whole_body" (a full-configuration goal in the planning joint
        space) is supported in this stage; other goal types raise
        NotImplementedError.
        """
        start = np.asarray(start, dtype=float).reshape(-1)
        goal = np.asarray(goal, dtype=float).reshape(-1)
        if start.shape[0] != self.n_dof:
            raise ValueError(
                f"start has {start.shape[0]} DoF, expected {self.n_dof}"
            )
        if goal_type != "whole_body":
            raise NotImplementedError(
                f"goal_type '{goal_type}' is not implemented yet"
            )
        if goal.shape[0] != self.n_dof:
            raise ValueError(
                f"goal_type '{goal_type}' requires a full "
                f"{self.n_dof}-DoF configuration, got {goal.shape[0]}"
            )

        # Project both endpoints onto the manifold before use: an
        # unprojected start roots the tree off the manifold.
        start, start_correction = self._project_configuration(start, "start")
        goal, goal_correction = self._project_configuration(goal, "goal")
        self.last_plan_stats = {
            "start_projection_correction": start_correction,
            "goal_projection_correction": goal_correction,
        }
        if self.log:
            print(
                f"[ConstrainedOMPLPlanner] projection corrections: "
                f"start {start_correction:.3e}, goal {goal_correction:.3e}"
            )

        # Convert start and goal to OMPL states
        start_state = self.si.allocState()
        start_state.copy(start)
        self.ss.setStartState(start_state)

        goal_state = self.si.allocState()
        goal_state.copy(goal)
        self.ss.setGoalState(goal_state)

        # Set up the planner
        self.ss.setup()

        # Solve
        waypoints = np.array([start])
        status = self.ss.solve(float(timeout))
        if status.asString() == "Exact solution":
            path = self.ss.getSolutionPath()
            extracted = self._extract_path_states(path)
            path_length = float(
                sum(
                    np.linalg.norm(extracted[i + 1] - extracted[i])
                    for i in range(extracted.shape[0] - 1)
                )
            )
            degenerate = (
                path_length <= self.constraint.getTolerance()
                and float(np.linalg.norm(goal - start))
                > self.constraint.getTolerance()
            )
            self.last_plan_stats["degenerate_path_recovered"] = degenerate
            if degenerate:
                # Build-bug workaround; see _recover_degenerate_path.
                waypoints = self._recover_degenerate_path(
                    start_state, goal_state, start, goal
                )
                # The path object is corrupt, so simplification would
                # operate on garbage; the recovered geodesic is reported
                # for both measurements.
                error_before_simplify = self._max_feet_error_over(waypoints)
                error_after_simplify = error_before_simplify
            else:
                error_before_simplify = self._max_feet_error_over(extracted)
                if smooth_path:
                    ps = og.PathSimplifier(self.si)
                    if shortcut_path:
                        try:
                            ps.ropeShortcutPath(path)
                        except Exception:
                            ps.shortcutPath(path)
                    ps.smoothBSpline(path)
                # Simplification can shortcut off the manifold: measure
                # the constraint drift it introduced rather than trusting
                # it.
                waypoints = self._extract_path_states(path)
                error_after_simplify = self._max_feet_error_over(waypoints)
            self.last_plan_stats["max_feet_error_before_simplify"] = (
                error_before_simplify
            )
            self.last_plan_stats["max_feet_error_after_simplify"] = (
                error_after_simplify
            )
            if self.log:
                if degenerate:
                    print(
                        "[ConstrainedOMPLPlanner] solver returned a "
                        "corrupted zero-length path (known build defect); "
                        "recovered the geodesic between the projected "
                        "endpoints instead"
                    )
                print(
                    f"[ConstrainedOMPLPlanner] max feet error along path: "
                    f"{error_before_simplify:.3e} before simplification, "
                    f"{error_after_simplify:.3e} after"
                )

        self.ss.clear()
        return waypoints


def load_start_goal_pairs(
    path: str,
    start_key: str = "starts",
    goal_key: str = "goals",
) -> tuple[np.ndarray, np.ndarray]:
    """Load start/goal configuration pairs from an .npz file.

    The file is expected to hold two float arrays of shape (N, n_plan) --
    ``start_key`` and ``goal_key`` -- with rows in the planner's planning
    joint order (base slots first when the base is planned).  The file does
    not exist yet as of this stage; this loader defines the contract without
    inventing a path.
    """
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Start/goal pair file not found: {path}"
        )
    data = np.load(path, allow_pickle=True)
    for key in (start_key, goal_key):
        if key not in data.files:
            raise KeyError(
                f"Key '{key}' not found in {path}; available keys: "
                f"{data.files}"
            )
    starts = np.asarray(data[start_key], dtype=float)
    goals = np.asarray(data[goal_key], dtype=float)
    if starts.ndim != 2 or starts.shape != goals.shape:
        raise ValueError(
            f"Expected matching (N, n_plan) arrays, got {start_key} "
            f"{starts.shape} and {goal_key} {goals.shape}"
        )
    return starts, goals


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class ConstrainedPlanningDemoConfig:
    """Run one constrained whole-body plan end to end."""

    urdf_path: str = str(
        _REPO_ROOT
        / "decoupled_wbc/control/robot_model/model_data/g1"
        / "g1_29dof_with_hand.urdf"
    )
    trajectory_path: str = str(
        _REPO_ROOT / "test_dataset" / "reference_trajectory_sym.npz"
    )
    start_frame: int = 0
    # Frame 240: ~1.3 rad from frame 0 in the planning space with a
    # comfortable world-frame CoM margin.  Frames 88-213 are unusable as
    # endpoints -- their waist_pitch exceeds the URDF limit (the dataset
    # defect reported by test T7.1), so the goal state fails the bounds
    # check.
    goal_frame: int = 240
    planner: str = "AORRTC"
    timeout: float = 10.0
    # The reference demo stands with a ~0.034 m static margin, so the
    # planner-default 0.05 would reject its own start state.
    com_margin: float = 0.03


def _dataset_frame_to_q_plan(data, frame: int) -> np.ndarray:
    """Dataset frame -> planning vector [base pos, base rotvec, joints]."""
    w, x, y, z = np.asarray(data["base_quat"][frame], dtype=float)
    rotation_vector = pin.log3(pin.Quaternion(w, x, y, z).matrix())
    return np.concatenate(
        [
            np.asarray(data["base_pos"][frame], dtype=float),
            rotation_vector,
            np.asarray(data["joint_pos"][frame], dtype=float),
        ]
    )


def main(config: ConstrainedPlanningDemoConfig) -> int:
    if not Path(config.urdf_path).exists():
        print(f"Robot model not found: {config.urdf_path}", file=sys.stderr)
        return 1
    if not Path(config.trajectory_path).exists():
        print(
            f"Trajectory dataset not found: {config.trajectory_path}",
            file=sys.stderr,
        )
        return 1

    data = np.load(config.trajectory_path, allow_pickle=True)

    # Free-flyer planning model: base first (MCVAMP convention), then every
    # 1-DoF body joint, hands staying in q_nominal.
    probe_model = RobotModel(
        config.urdf_path,
        str(Path(config.urdf_path).resolve().parent),
        set_floating_base=True,
    ).pinocchio_wrapper.model
    names = list(probe_model.names)
    planning_joint_names = [names[1]] + [
        name for name in names[2:] if "hand" not in name
    ]
    print(
        f"Free-flyer model: nq = {probe_model.nq}, nv = {probe_model.nv}"
    )

    start = None  # placate linters; assigned below
    try:
        start = _dataset_frame_to_q_plan(data, config.start_frame)
        goal = _dataset_frame_to_q_plan(data, config.goal_frame)
    except IndexError:
        print(
            f"Frames ({config.start_frame}, {config.goal_frame}) out of "
            f"range for {data['joint_pos'].shape[0]}-frame dataset",
            file=sys.stderr,
        )
        return 1

    planner = ConstrainedOMPLPlanner(
        robot_model=None,  # collision hookup disabled; see validity_checker
        urdf_path=config.urdf_path,
        planning_joint_names=planning_joint_names,
        q_nominal=pin.neutral(probe_model),
        q_reference=start,
        planner=config.planner,
        com_margin=config.com_margin,
    )
    print(
        f"Planning space: {planner.n_dof} DoF; start/goal CoM margins: "
        f"{planner.com_constraint.margin_at(start):.4f} / "
        f"{planner.com_constraint.margin_at(goal):.4f} "
        f"(required: {config.com_margin})"
    )

    waypoints = planner.plan(start, goal, timeout=config.timeout)
    if waypoints.shape[0] <= 1:
        print(
            "No exact solution found: check the CoM margins above (an "
            "unstable endpoint makes the problem infeasible) and the "
            "projection corrections in last_plan_stats "
            f"({planner.last_plan_stats}), or increase --timeout",
            file=sys.stderr,
        )
        return 1

    stats = planner.last_plan_stats
    stable = sum(
        bool(planner.com_constraint.is_stable(q)) for q in waypoints
    )
    print(
        f"Planned {waypoints.shape[0]} waypoints "
        f"({config.start_frame} -> {config.goal_frame}, "
        f"planner {config.planner});\n"
        f"  projection corrections: "
        f"start {stats['start_projection_correction']:.3e}, "
        f"goal {stats['goal_projection_correction']:.3e}\n"
        f"  max feet error: "
        f"{stats['max_feet_error_before_simplify']:.3e} before "
        f"simplification, "
        f"{stats['max_feet_error_after_simplify']:.3e} after\n"
        f"  CoM-stable waypoints: {stable}/{waypoints.shape[0]}"
    )
    return 0


if __name__ == "__main__":
    import tyro

    sys.exit(main(tyro.cli(ConstrainedPlanningDemoConfig)))

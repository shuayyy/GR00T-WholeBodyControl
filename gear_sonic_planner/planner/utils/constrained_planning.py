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
    """Anchored manifold sampler for the ambient space.

    Replaces this build's ``ProjectedStateSpace`` sampler, which returns
    raw ambient samples (measured feet errors of 2-5 on samples that
    should satisfy 1e-3).  Sample targets must be ON the manifold and IN
    bounds for constrained tree search to work here: extensions toward an
    off-manifold or out-of-bounds target make the projected geodesic
    collapse onto the tree node and the extension is discarded, so trees
    only grow through tree-to-tree connects (measured: 6-12 vertices in
    240 s, unsolvable non-trivial problems).

    Uniform-ambient projection cannot produce such targets: Newton
    projection of a uniform 35-D sample succeeds only ~15% of the time
    and the result virtually never respects the joint limits (measured
    0/30 in bounds; clipping it back destroys the projection, feet error
    ~3).  So samples are instead drawn around ANCHORS -- known-feasible
    manifold configurations (the reference posture plus the current
    start/goal, maintained by the planner in the shared ``anchors``
    list) -- perturbed with a per-sample radius, projected, and accepted
    only when in bounds (measured acceptance at radius 0.4: 50/60).  A
    residual uniform-ambient fraction keeps global coverage; when
    everything fails, the projected-then-clipped ambient sample of the
    stock upstream semantics is the fallback.

    The anchors are single feasible configurations, NOT trajectory data
    -- this does not bias the search along any reference motion.
    """

    #: fraction of draws perturbing an anchor (rest are uniform-ambient)
    ANCHOR_FRACTION = 0.8
    #: per-sample perturbation radius range around an anchor [rad | m]
    ANCHOR_RADIUS = (0.2, 1.0)
    #: attempts at an on-manifold in-bounds sample before falling back
    ACCEPT_ATTEMPTS = 20

    def __init__(self, space, constraint, lower, upper, anchors=None, seed=None):
        super().__init__(space)
        self.constraint = constraint
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.n_dof = self.lower.shape[0]
        self.anchors = anchors if anchors is not None else []
        self.rng = np.random.default_rng(seed)

    def _in_bounds(self, x):
        return bool(np.all(x >= self.lower) and np.all(x <= self.upper))

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
        for _ in range(self.ACCEPT_ATTEMPTS):
            if self.anchors and self.rng.random() < self.ANCHOR_FRACTION:
                anchor = self.anchors[self.rng.integers(len(self.anchors))]
                radius = self.rng.uniform(*self.ANCHOR_RADIUS)
                x = np.clip(
                    anchor + self.rng.uniform(-radius, radius, self.n_dof),
                    self.lower,
                    self.upper,
                )
            else:
                x = self.rng.uniform(self.lower, self.upper)
            if self.constraint.project(x) and self._in_bounds(x):
                self._write(state, x)
                return
        # Fallback: stock upstream semantics (project, then enforce
        # bounds), off-manifold but usable as a Voronoi-bias target.
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

    Two defects in this OMPL build are handled and loudly documented:
    ProjectingStateSampler (see its docstring) works around the build's
    non-projecting sampler, and plan() REJECTS corrupted solution paths.
    The corruption is an AORRTC bug, located in the build's source:
    AOXRRTConnect::solve primes its first iteration with a straight-line
    connect (AOXRRTConnect.cpp L369-372) while ``tgi.xmotion`` still
    points at the goal tree's root motion, so when the direct start->goal
    connect succeeds the "solution" is assembled entirely from the goal
    tree -- [goal, goal], length 0, the start state never appears.  It
    fires exactly when the problem is solvable by one direct connect.
    Such a path carries no route information, so plan() raises
    RuntimeError instead of returning anything; use RRTConnect (the
    default) or RRTstar, which do not trigger it.

    A second AORRTC quirk to know when reading last_plan_stats: AORRTC
    resets its inner planner (freeing both trees) every time it finds a
    solution, and getPlannerData() reads the inner planner -- so
    ``planner_vertices`` may legitimately report the in-progress tree of
    the *next* anytime iteration, or 0, for AORRTC.  Tree sizes are only
    faithful for single-shot planners (RRTConnect, RRTstar, ...).
    """

    def __init__(
        self,
        robot_model,
        urdf_path: str,
        planning_joint_names: list[str],
        q_nominal: np.ndarray,
        q_reference: np.ndarray,
        planner: str = "RRTConnect",
        validity_resolution: float = 0.01,
        extend_range: float | None = 0.5,
        com_margin: float = 0.05,
        projection_delta: float = 0.05,
        projection_lambda: float = 10.0,
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
        # Build-bug workaround: make the ambient sampler produce usable
        # manifold samples (see ProjectingStateSampler).  The reference
        # posture seeds the anchor list; plan() adds the projected
        # start/goal.  The allocator must outlive the space, hence the
        # attribute references.
        self._sampler_anchors = [self.q_reference.copy()]
        self._sampler_allocator = lambda sampler_space: ProjectingStateSampler(
            sampler_space,
            self.constraint,
            self._bounds_low,
            self._bounds_high,
            anchors=self._sampler_anchors,
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
        # Feasible seeds for the anchored sampler (list is shared with it).
        self._sampler_anchors[:] = [
            self.q_reference.copy(),
            start.copy(),
            goal.copy(),
        ]
        self.last_plan_stats = {
            "start_projection_correction": start_correction,
            "goal_projection_correction": goal_correction,
            # Kept for downstream readers; corrupted paths raise instead of
            # being recovered, so this is always False when plan() returns.
            "degenerate_path_recovered": False,
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
        # Tree size, read before clear() wipes the planner.  Only faithful
        # for single-shot planners; see the class docstring re AORRTC.
        try:
            planner_data = ob.PlannerData(self.si)
            self.ss.getPlanner().getPlannerData(planner_data)
            self.last_plan_stats["planner_vertices"] = int(
                planner_data.numVertices()
            )
            self.last_plan_stats["planner_edges"] = int(
                planner_data.numEdges()
            )
        except Exception:  # PlannerData is diagnostic; never fail a plan
            pass
        if status.asString() == "Exact solution":
            path = self.ss.getSolutionPath()
            extracted = self._extract_path_states(path)
            path_length = float(
                sum(
                    np.linalg.norm(extracted[i + 1] - extracted[i])
                    for i in range(extracted.shape[0] - 1)
                )
            )
            tolerance = self.constraint.getTolerance()
            # AORRTC's corrupted paths are built from the goal tree only
            # (class docstring): they never begin at the start state.
            # That test subsumes the [goal, goal] zero-length signature
            # and also catches the multi-vertex variant.  Such a "solution"
            # carries no route information, so it is an error, never a
            # result.
            symptoms = []
            start_gap = float(np.linalg.norm(extracted[0] - start))
            if start_gap > tolerance:
                symptoms.append(
                    f"path begins {start_gap:.3e} away from the start state"
                )
            if (
                path_length <= tolerance
                and float(np.linalg.norm(goal - start)) > tolerance
            ):
                symptoms.append(
                    f"path length {path_length:.3e} between endpoints "
                    f"{np.linalg.norm(goal - start):.3e} apart"
                )
            if symptoms:
                self.ss.clear()
                raise RuntimeError(
                    f"{self.planner_name} returned a corrupted solution "
                    f"path: {'; '.join(symptoms)}.  Known AORRTC "
                    f"first-iteration defect in this OMPL build (see class "
                    f"docstring); plan with RRTConnect or RRTstar instead"
                )
            error_before_simplify = self._max_feet_error_over(extracted)
            if smooth_path:
                ps = og.PathSimplifier(self.si)
                if shortcut_path:
                    try:
                        ps.ropeShortcutPath(path)
                    except Exception:
                        ps.shortcutPath(path)
                ps.smoothBSpline(path)
            # Simplification can shortcut off the manifold: measure the
            # constraint drift it introduced rather than trusting it.
            waypoints = self._extract_path_states(path)
            error_after_simplify = self._max_feet_error_over(waypoints)
            self.last_plan_stats["max_feet_error_before_simplify"] = (
                error_before_simplify
            )
            self.last_plan_stats["max_feet_error_after_simplify"] = (
                error_after_simplify
            )
            if self.log:
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
    # RRTConnect: AORRTC in this build corrupts its solution path whenever
    # the first-iteration direct connect succeeds (see ConstrainedOMPLPlanner
    # docstring) and its PlannerData is unreliable.
    planner: str = "RRTConnect"
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

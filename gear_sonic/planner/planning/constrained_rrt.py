from __future__ import annotations

from pathlib import Path

import numpy as np
import ompl.base as ob
import ompl.geometric as og
import ompl.util as ou
import pinocchio as pin

from gear_sonic.data.robot_model.robot_model import RobotModel
from gear_sonic.planner.constraints.com_constraint import (
    CoMConstraint,
)
from gear_sonic.planner.constraints.embedding import PlanningEmbedder
from gear_sonic.planner.constraints.feet_constraint import (
    FeetConstraint,
)

# The URDF gives the free-flyer no position limits: bound the base to a
# workspace box around the reference, and the rotation vector to +-pi.
BASE_POSITION_HALF_RANGE = 1.0


class FeetManifoldConstraint(ob.Constraint):
    """OMPL constraint adapter over FeetConstraint (co-dimension 12).

    The CoM hinge is full-dimensional and would corrupt ProjectedStateSpace's
    bookkeeping, so it lives in the validity checker.  Tolerance is 1e-3, not
    OMPL's 1e-4: each Newton iteration costs a Python FK round trip.
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

    This build's sampler returns raw ambient samples, and off-manifold targets
    collapse the projected geodesic, so extensions get discarded.  Samples are
    instead perturbed around anchors -- known-feasible configurations in the
    shared ``anchors`` list -- projected, and kept only if in bounds.
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
        # Constrained states are bulk-writable via copy(); ambient
        # RealVector states are not, and go elementwise.
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
        # Fallback: off-manifold, but usable as a Voronoi-bias target.
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
    """OMPL constrained geometric planner for whole-body paths.

    Manifold pins both feet; stability is a validity check.  Free-flyer root,
    so the base pose is in the planning space.  Feasibility only.

    Two build defects handled here: ProjectingStateSampler replaces the
    non-projecting sampler, and plan() rejects AORRTC's corrupted paths
    (AOXRRTConnect.cpp L369-372 builds them from the goal tree alone).
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
        goal_threshold: float = 0.0075,
        collision_rig=None,
        collision_tolerance: float = 1e-3,
        log: bool = True,
    ):
        """Initialize Planner"""
        # Kept for parity with OMPLGeometricPlanner; all kinematics below
        # run on the free-flyer planning model.
        self.robot_model = robot_model

        # Free-flyer root; also loads the collision geometry the CoM
        # support polygon needs.
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

        # The rig owns its scene and finger pose; we need only the
        # planning -> MuJoCo permutation, by name.
        self.collision_rig = collision_rig
        self.collision_tolerance = float(collision_tolerance)
        self._plan_to_mj = None
        if collision_rig is not None:
            from gear_sonic.planner.joint_orders import name_permutation

            base_slice = self.embedder.base_plan_slice
            plan_joint_names = [
                name for name in planning_joint_names
                if base_slice is None or name != planning_joint_names[0]
            ]
            self._plan_to_mj = name_permutation(
                plan_joint_names, list(collision_rig.body_joint_names)
            )

        q_reference = np.asarray(q_reference, dtype=float).reshape(-1)
        if q_reference.shape[0] != self.n_dof:
            raise ValueError(
                f"q_reference has {q_reference.shape[0]} DoF, expected "
                f"{self.n_dof}"
            )
        self.q_reference = q_reference.copy()

        # Feet define the manifold; the CoM hinge gates validity.
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
        self.goal_threshold = goal_threshold
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
        # URDF limits, +-2pi fallback for infinite ones.
        revolute_positions = self.embedder.revolute_plan_positions
        revolute_idx_q = self.embedder.revolute_idx_q
        low[revolute_positions] = self.pin_model.lowerPositionLimit[
            revolute_idx_q
        ]
        high[revolute_positions] = self.pin_model.upperPositionLimit[
            revolute_idx_q
        ]
        # Free-flyer slots: workspace box around the reference, +-pi
        # for the rotation vector.
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
        # Build-bug workaround (see ProjectingStateSampler).  The allocator
        # must outlive the space, hence the attribute references.
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
        # delta = geodesic step; lambda caps geodesic/ambient length ratio.
        self.constrained_space.setDelta(self.projection_delta)
        self.constrained_space.setLambda(self.projection_lambda)
        csi = ob.ConstrainedSpaceInformation(self.constrained_space)

        # Simple Setup
        ss = og.SimpleSetup(csi)
        csi.setStateValidityCheckingResolution(self.validity_resolution)
        ss.setStateValidityChecker(self.validity_checker)
        # No optimization objective: pure feasibility planning.

        # Set planner
        planner = getattr(og, self.planner_name)(csi)
        ss.setPlanner(planner)
        return ss, csi

    def validity_checker(self, state: ob.State):
        """Bounds, static stability and self-collision."""
        # nanobind hands over the constrained state type directly; indexing
        # reads the wrapped Eigen map.
        q = np.array([state[i] for i in range(self.n_dof)], dtype=float)

        # Cheapest test first; collision is a full MuJoCo FK + broadphase.
        if not self.si.satisfiesBounds(state):
            return False
        if not self.com_constraint.is_stable(q):
            return False
        if self.collision_rig is not None and self.in_self_collision(q):
            return False
        return True

    def in_self_collision(self, q_plan: np.ndarray) -> bool:
        """Self-collision of a planning vector. The planning space has no
        hand joints, so the rig holds them at the deploy's closure."""
        if self.collision_rig is None:
            return False
        joints_mj, base_pos, base_quat = self.to_collision_frame(q_plan)
        return self.collision_rig.in_contact(
            joints_mj, base_pos, base_quat, tolerance=self.collision_tolerance
        )

    def to_collision_frame(
        self, q_plan: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Planning vector -> (joints in MuJoCo order, base pos, quat wxyz).
        The permutation is built once, by name."""
        q_plan = np.asarray(q_plan, dtype=float).reshape(-1)
        base = self.embedder.base_plan_slice
        if base is None:
            raise RuntimeError(
                "Collision checking needs the free-flyer base in the planning space"
            )
        base_pos = q_plan[base.start : base.start + 3]
        rotation = pin.exp3(q_plan[base.start + 3 : base.stop])
        quaternion = pin.Quaternion(rotation)
        base_quat = np.array(
            [quaternion.w, quaternion.x, quaternion.y, quaternion.z], dtype=float
        )
        joints_plan = np.delete(q_plan, np.arange(base.start, base.stop))
        return joints_plan[self._plan_to_mj], base_pos, base_quat

    def _project_configuration(
        self, q_plan: np.ndarray, label: str
    ) -> tuple[np.ndarray, float]:
        """Project onto the feet manifold, returning (config, correction).
        Raises on failure: an off-manifold root invalidates every geodesic."""
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
        """Plan a constrained path from start to goal. Only "whole_body"
        goals are supported."""
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

        # An unprojected start would root the tree off the manifold.
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
            # Always False on return: corrupted paths raise instead.
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
        # OMPL's setGoalState default is machine epsilon, far below the
        # manifold's 1e-3 tolerance.  Measured against the projected goal.
        self.ss.setGoalState(goal_state, self.goal_threshold)

        # Set up the planner
        self.ss.setup()

        # Solve
        waypoints = np.array([start])
        status = self.ss.solve(float(timeout))
        # Read before clear() wipes the planner.
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
            # AORRTC's corrupted paths never begin at the start state.
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
            # Simplification can shortcut off the manifold; measure the drift.
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


def default_planning_joint_names(urdf_path: str):
    """Free-flyer planning model and its joint names: root first, then the 29
    1-DoF body joints in Pinocchio order. Hands stay in ``q_nominal``."""
    probe_model = RobotModel(
        urdf_path,
        str(Path(urdf_path).resolve().parent),
        set_floating_base=True,
    ).pinocchio_wrapper.model
    names = list(probe_model.names)
    planning_joint_names = [names[1]] + [
        name for name in names[2:] if "hand" not in name
    ]
    return probe_model, planning_joint_names


def load_start_goal_pairs(
    path: str,
    start_key: str = "starts",
    goal_key: str = "goals",
) -> tuple[np.ndarray, np.ndarray]:
    """Load start/goal pairs from an .npz holding two (N, n_plan) arrays in
    planning joint order."""
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

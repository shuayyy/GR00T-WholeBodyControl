"""PhaseRRTstar as a constrained whole-body planner.

Searches ``(q, alpha)``: configuration plus a phase coordinate along a required
reference, advancing alpha monotonically so the path stays ordered along it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import ompl.base as ob
import ompl.geometric as og
import ompl.util as ou
import pinocchio as pin

from decoupled_wbc.control.robot_model.robot_model import RobotModel
from gear_sonic.planner.constraints.com_constraint import CoMConstraint
from gear_sonic.planner.constraints.embedding import PlanningEmbedder
from gear_sonic.planner.constraints.feet_constraint import FeetConstraint
from gear_sonic.planner.joint_orders import mjcf_joint_names, name_permutation
from gear_sonic.planner.planning.constrained_rrt import (
    BASE_POSITION_HALF_RANGE,
    default_planning_joint_names,
)

#: Sample weights that quieten the free-flyer base slots: the projection
#: handles the feet, so samples need not perturb the base aggressively.
QUIET_BASE_WEIGHTS = [0.1] * 6 + [1.0] * 29


def reference_arclength(reference: np.ndarray) -> float:
    """Total joint-space travel of the reference, in planning units."""
    reference = np.asarray(reference, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(reference, axis=0), axis=1)))


def phase_defaults(arclength: float) -> dict:
    """PhaseRRTstar knobs derived from the reference arclength ``L``.

    ``d_alpha_max`` is 0.1, not the source's 0.05: on the constrained
    whole-body problem 0.05 only ever returned "Approximate solution".  The
    tuning assumes obstacle-free problems; with collision on, raise ``sigma``
    and restore some uniform fraction.
    """
    w_alphastate = 0.756 * arclength
    return {
        "w_alphastate": w_alphastate,
        # range must cover L * d_alpha_max with margin (0.0646 * L covers a
        # 0.05 step at 1.3x; it is deliberately NOT rescaled for 0.1 -- the
        # Exact-solution runs used exactly this range with d_alpha_max 0.1).
        "range": 0.0646 * arclength,
        "sigma": 0.0018 * arclength,
        "d_alpha_min": 0.005,
        "d_alpha_max": 0.1,
        "uniform_fraction": 0.0,
        "goal_bias": 0.05,
        "phase_grid": 1000.0,   # 0.001 alpha grid (enables d_alpha_min 0.005)
        # below half a grid level in the alpha term: only alpha = 1.000 can
        # satisfy the goal (the goal door inserts the exact goal state, so a
        # tight threshold costs nothing).
        "goal_threshold": 0.5 * w_alphastate * 0.001,
    }


def load_reference_npz(
    path: str | Path,
    planning_joint_names: list[str],
    lo: np.ndarray | None = None,
    hi: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Trajectory ``.npz`` -> (N, 35) reference in planning coordinates.

    Joints are mapped by name into Pinocchio order and the base quaternion
    becomes an so(3) rotation vector.  With ``lo``/``hi`` the reference is
    clipped into bounds and the clipping reported: a non-zero report means
    the dataset and the URDF limits have drifted apart.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Reference npz not found: {path}")
    data = np.load(path, allow_pickle=True)
    for key in ("joint_pos", "base_pos", "base_quat"):
        if key not in data.files:
            raise KeyError(f"{path} is missing '{key}'")
    order = str(data["joint_order"]) if "joint_order" in data.files else "mujoco"
    if order != "mujoco":
        raise ValueError(f"{path} declares joint_order='{order}', expected 'mujoco'")

    joint_mj = np.asarray(data["joint_pos"], dtype=float)
    base_pos = np.asarray(data["base_pos"], dtype=float)
    base_quat = np.asarray(data["base_quat"], dtype=float)
    perm = name_permutation(mjcf_joint_names(), list(planning_joint_names[1:]))

    n_frames = joint_mj.shape[0]
    reference = np.empty((n_frames, 6 + joint_mj.shape[1]))
    reference[:, 0:3] = base_pos
    for frame in range(n_frames):
        w, x, y, z = base_quat[frame]
        reference[frame, 3:6] = pin.log3(pin.Quaternion(w, x, y, z).matrix())
    reference[:, 6:] = joint_mj[:, perm]

    info = {"frames": n_frames, "clip_max": 0.0, "clip_frames": 0}
    if lo is not None and hi is not None:
        clipped = np.clip(reference, lo, hi)
        info["clip_max"] = float(np.abs(clipped - reference).max())
        info["clip_frames"] = int(
            np.sum(np.any(np.abs(clipped - reference) > 1e-12, axis=1))
        )
        reference = clipped
    info["arclength"] = reference_arclength(reference)
    return reference, info


class ConstrainedPhaseRRTstarPlanner:
    """PhaseRRTstar on the feet manifold, interface-compatible with
    :class:`ConstrainedOMPLPlanner`: same waypoint array, same
    ``last_plan_stats`` keys plus phase-specific ones."""

    def __init__(
        self,
        robot_model,
        urdf_path: str,
        planning_joint_names: list[str],
        q_nominal: np.ndarray,
        q_reference: np.ndarray,
        reference: np.ndarray,
        planner: str = "PhaseRRTstar",
        validity_resolution: float = 0.05,
        com_margin: float = 0.05,
        projection_delta: float = 0.15,
        projection_lambda: float = 10.0,
        rewire_factor: float = 0.5,
        sample_weights: list[float] | None = None,
        phase_params: dict | None = None,
        collision_rig=None,
        collision_tolerance: float = 1e-3,
        seed: int | None = None,
        log: bool = True,
    ):
        """Mirrors ConstrainedOMPLPlanner. ``reference`` is required, and the
        resolution/delta/rewire defaults are coarser because every geodesic
        step costs a Python FK round trip."""
        self.robot_model = robot_model
        self.planner_name = planner
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

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
                f"q_reference has {q_reference.shape[0]} DoF, expected {self.n_dof}"
            )
        self.q_reference = q_reference.copy()

        reference = np.asarray(reference, dtype=float)
        if reference.ndim != 2 or reference.shape[1] != self.n_dof:
            raise ValueError(
                f"reference must be (N, {self.n_dof}), got {reference.shape}"
            )
        if reference.shape[0] < 2:
            raise ValueError("reference needs at least two frames")
        self.reference = reference
        self.arclength = reference_arclength(reference)

        self.params = phase_defaults(self.arclength)
        if phase_params:
            self.params.update(phase_params)
        self.w_alphastate = float(self.params["w_alphastate"])

        # Constraints: feet define the manifold, the CoM hinge gates validity.
        self.feet_constraint = FeetConstraint(
            self.planning_robot_model, self.embedder, self.q_reference
        )
        self.com_constraint = CoMConstraint(
            self.planning_robot_model, self.embedder, margin=com_margin
        )

        self.collision_rig = collision_rig
        self.collision_tolerance = float(collision_tolerance)
        self._plan_to_mj = None
        if collision_rig is not None:
            base_slice = self.embedder.base_plan_slice
            plan_joint_names = [
                name for name in planning_joint_names
                if base_slice is None or name != planning_joint_names[0]
            ]
            self._plan_to_mj = name_permutation(
                plan_joint_names, list(collision_rig.body_joint_names)
            )

        self.validity_resolution = float(validity_resolution)
        self.projection_delta = float(projection_delta)
        self.projection_lambda = float(projection_lambda)
        self.rewire_factor = float(rewire_factor)
        self.sample_weights = sample_weights

        self.ss, self.csi = self.set_up_ompl()
        self.planner = self.ss.getPlanner()

        self.log = log
        if not log:
            ou.setLogLevel(ou.LOG_ERROR)
        self.last_plan_stats: dict = {}


    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """URDF limits for the 1-DoF slots, workspace box for the base,
        +-2pi fallback for infinite limits."""
        n = self.n_dof
        low = np.empty(n)
        high = np.empty(n)
        revolute_positions = self.embedder.revolute_plan_positions
        revolute_idx_q = self.embedder.revolute_idx_q
        low[revolute_positions] = self.pin_model.lowerPositionLimit[revolute_idx_q]
        high[revolute_positions] = self.pin_model.upperPositionLimit[revolute_idx_q]
        base = self.embedder.base_plan_slice
        if base is not None:
            base_position = slice(base.start, base.start + 3)
            base_rotation = slice(base.start + 3, base.stop)
            low[base_position] = self.q_reference[base_position] - BASE_POSITION_HALF_RANGE
            high[base_position] = self.q_reference[base_position] + BASE_POSITION_HALF_RANGE
            low[base_rotation] = -np.pi
            high[base_rotation] = np.pi
        low[~np.isfinite(low)] = -2.0 * np.pi
        high[~np.isfinite(high)] = 2.0 * np.pi
        return low, high

    def set_up_ompl(self):
        n = self.n_dof
        low, high = self._bounds()
        self._bounds_low = low.copy()
        self._bounds_high = high.copy()

        feet = self.feet_constraint

        class FeetManifoldWithAlpha(ob.Constraint):
            """Feet constraint with one extra unconstrained slot: the alpha
            Jacobian column is zero, so projection never touches phase."""

            def __init__(self):
                super().__init__(n + 1, feet.n_rows)
                self.setTolerance(1e-3)

            def function(self, x, out):
                out[:] = feet.error(np.asarray(x, dtype=float)[:n])

            def jacobian(self, x, out):
                J = feet.jacobian(np.asarray(x, dtype=float)[:n])
                for row in range(J.shape[0]):
                    out[row][:n] = J[row]
                    out[row][n] = 0.0

        space = ob.RealVectorStateSpace(n + 1)
        bounds = ob.RealVectorBounds(n + 1)
        for i in range(n):
            bounds.setLow(i, float(low[i]))
            bounds.setHigh(i, float(high[i]))
        bounds.setLow(n, 0.0)
        bounds.setHigh(n, self.w_alphastate)   # slot holds alpha * w_alphastate
        space.setBounds(bounds)

        self.constraint = FeetManifoldWithAlpha()
        constrained_space = ob.ProjectedStateSpace(space, self.constraint)
        constrained_space.setDelta(self.projection_delta)
        constrained_space.setLambda(self.projection_lambda)
        self.constrained_space = constrained_space
        csi = ob.ConstrainedSpaceInformation(constrained_space)
        csi.setStateValidityCheckingResolution(self.validity_resolution)
        csi.setStateValidityChecker(self.validity_checker)

        ss = og.SimpleSetup(csi)

        planner = og.PhaseRRTstar(csi)
        planner.setFlatAlpha(self.w_alphastate)
        planner.setProjectionConstraint(self.constraint)
        planner.setReference([list(map(float, row)) for row in self.reference])
        planner.setRange(float(self.params["range"]))
        planner.setSampleSigma(float(self.params["sigma"]))
        planner.setDAlphaMin(float(self.params["d_alpha_min"]))
        planner.setDAlphaMax(float(self.params["d_alpha_max"]))
        planner.setUniformFraction(float(self.params["uniform_fraction"]))
        planner.setGoalBias(float(self.params["goal_bias"]))
        planner.setPhaseGrid(float(self.params["phase_grid"]))
        planner.setRewireFactor(self.rewire_factor)
        if self.sample_weights is not None:
            planner.setSampleWeights([float(x) for x in self.sample_weights])

        # interpolation False: motionCost as an endpoint trapezoid.  With
        # interpolation every rewire-neighbour cost runs a projected geodesic
        # (a Python FK per step), which is what limits this to ~30 states.
        objective = ob.PhaseSimilarityObjective(csi, False)
        objective.setReference([list(map(float, row)) for row in self.reference])
        objective.setAlphaScale(self.w_alphastate)
        ss.setOptimizationObjective(objective)
        self.objective = objective

        ss.setPlanner(planner)
        return ss, csi


    def validity_checker(self, state) -> bool:
        """Bounds, CoM stability and self-collision.

        Bounds are checked in numpy so the callback never re-enters OMPL, and
        the return is a builtin bool -- ``numpy.bool_`` raises bad_cast.
        """
        q = np.array([state[i] for i in range(self.n_dof)], dtype=float)
        if not (np.all(q >= self._bounds_low) and np.all(q <= self._bounds_high)):
            return False
        if not bool(self.com_constraint.is_stable(q)):
            return False
        if self.collision_rig is not None and self.in_self_collision(q):
            return False
        return True

    def in_self_collision(self, q_plan: np.ndarray) -> bool:
        if self.collision_rig is None:
            return False
        joints_mj, base_pos, base_quat = self.to_collision_frame(q_plan)
        return bool(
            self.collision_rig.in_contact(
                joints_mj, base_pos, base_quat, tolerance=self.collision_tolerance
            )
        )

    def to_collision_frame(self, q_plan: np.ndarray):
        """Planning vector -> (joints MuJoCo order, base pos, base quat wxyz)."""
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


    def _project(self, q_plan: np.ndarray, label: str) -> tuple[np.ndarray, float]:
        """Project a configuration onto the feet manifold (alpha untouched)."""
        x = np.append(np.asarray(q_plan, dtype=float), 0.0)
        if not self.constraint.project(x):
            residual = float(np.linalg.norm(self.feet_constraint.error(x[: self.n_dof])))
            raise RuntimeError(
                f"Failed to project the {label} configuration onto the feet "
                f"manifold (residual {residual:.3e})"
            )
        projected = x[: self.n_dof]
        return projected, float(np.linalg.norm(projected - q_plan))

    def _max_feet_error_over(self, waypoints: np.ndarray) -> float:
        return max(
            float(np.linalg.norm(self.feet_constraint.error(q))) for q in waypoints
        )


    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        goal_type: str = "whole_body",
        timeout: float = 100.0,
        smooth_path: bool = False,
        shortcut_path: bool = False,
    ) -> np.ndarray:
        """Plan a phase-guided constrained path from start to goal.

        ``smooth_path`` / ``shortcut_path`` are ignored with a warning:
        shortcutting reorders states and destroys the monotone alpha
        schedule.  "Approximate solution" is a failure -- alpha never
        reached 1.0.
        """
        if goal_type != "whole_body":
            raise NotImplementedError(
                f"goal_type '{goal_type}' is not implemented for PhaseRRTstar"
            )
        start = np.asarray(start, dtype=float).reshape(-1)
        goal = np.asarray(goal, dtype=float).reshape(-1)
        for label, q in (("start", start), ("goal", goal)):
            if q.shape[0] != self.n_dof:
                raise ValueError(
                    f"{label} has {q.shape[0]} DoF, expected {self.n_dof}"
                )
        if (smooth_path or shortcut_path) and self.log:
            print(
                "[ConstrainedPhaseRRTstarPlanner] smooth_path/shortcut_path "
                "ignored: simplification would break alpha monotonicity"
            )

        start_q, start_correction = self._project(start, "start")
        goal_q, goal_correction = self._project(goal, "goal")

        self.last_plan_stats = {
            "planner": self.planner_name,
            "seed": self.seed,
            "arclength": self.arclength,
            "params": dict(self.params),
            "start_projection_correction": start_correction,
            "goal_projection_correction": goal_correction,
            # kept for parity with ConstrainedOMPLPlanner's readers; this
            # planner raises on corrupted paths rather than recovering.
            "degenerate_path_recovered": False,
        }
        if self.log:
            print(
                f"[ConstrainedPhaseRRTstarPlanner] projection corrections: "
                f"start {start_correction:.3e}, goal {goal_correction:.3e}"
            )

        start_state = self.csi.allocState()
        goal_state = self.csi.allocState()
        start_state.copy(np.append(start_q, 0.0))
        goal_state.copy(np.append(goal_q, self.w_alphastate))
        self.ss.setStartAndGoalStates(
            start_state, goal_state, float(self.params["goal_threshold"])
        )
        self.ss.setup()

        status = self.ss.solve(float(timeout))
        status_str = status.asString()
        self.last_plan_stats["status"] = status_str
        try:
            planner_data = ob.PlannerData(self.csi)
            self.ss.getPlanner().getPlannerData(planner_data)
            self.last_plan_stats["planner_vertices"] = int(planner_data.numVertices())
            self.last_plan_stats["planner_edges"] = int(planner_data.numEdges())
        except Exception:  # diagnostic only; never fail a plan for it
            pass

        if status_str != "Exact solution":
            # "Approximate solution" means alpha never reached 1.0.
            self.ss.clear()
            raise RuntimeError(
                f"{self.planner_name} returned '{status_str}' -- only an exact "
                f"solution reaches the goal phase (alpha = 1.0); stats: "
                f"{self.last_plan_stats}"
            )

        states = self.ss.getSolutionPath().getStates()
        raw = np.array(
            [[s[i] for i in range(self.n_dof + 1)] for s in states], dtype=float
        )
        waypoints = raw[:, : self.n_dof]
        alphas = raw[:, self.n_dof] / self.w_alphastate

        tolerance = self.constraint.getTolerance()
        start_gap = float(np.linalg.norm(waypoints[0] - start_q))
        if start_gap > tolerance:
            self.ss.clear()
            raise RuntimeError(
                f"{self.planner_name} returned a corrupted solution path: it "
                f"begins {start_gap:.3e} away from the start state"
            )

        alpha_steps = np.diff(alphas)
        self.last_plan_stats.update(
            {
                "num_states": int(waypoints.shape[0]),
                "alpha_first": float(alphas[0]),
                "alpha_final": float(alphas[-1]),
                "alpha_monotone": bool(np.all(alpha_steps >= -1e-9)),
                "alpha_step_min": float(alpha_steps.min()) if alpha_steps.size else 0.0,
                "alpha_step_max": float(alpha_steps.max()) if alpha_steps.size else 0.0,
                "max_feet_error_before_simplify": self._max_feet_error_over(waypoints),
                # no simplification is applied (see the docstring), so the
                # "after" value is the same path -- reported for parity.
                "max_feet_error_after_simplify": self._max_feet_error_over(waypoints),
                "com_unstable_waypoints": int(
                    sum(not bool(self.com_constraint.is_stable(q)) for q in waypoints)
                ),
                "joint_limit_violations": int(
                    np.sum(
                        (waypoints < self._bounds_low - 1e-9)
                        | (waypoints > self._bounds_high + 1e-9)
                    )
                ),
                "alphas": alphas,
            }
        )
        if self.log:
            print(
                f"[ConstrainedPhaseRRTstarPlanner] {waypoints.shape[0]} states, "
                f"alpha {alphas[0]:.3f} -> {alphas[-1]:.3f}, max feet error "
                f"{self.last_plan_stats['max_feet_error_after_simplify']:.3e}"
            )

        self.ss.clear()
        return waypoints


def build_default_phase_planner(
    urdf_path: str,
    reference_npz: str | Path,
    com_margin: float = 0.03,
    collision_rig=None,
    sample_weights: list[float] | None = None,
    seed: int | None = None,
    log: bool = True,
) -> tuple[ConstrainedPhaseRRTstarPlanner, np.ndarray, dict]:
    """Load the reference and build the planner around it, returning
    ``(planner, reference, info)``. Feet are pinned at frame 0."""
    probe_model, planning_joint_names = default_planning_joint_names(urdf_path)
    reference, info = load_reference_npz(reference_npz, planning_joint_names)
    planner = ConstrainedPhaseRRTstarPlanner(
        robot_model=None,
        urdf_path=urdf_path,
        planning_joint_names=planning_joint_names,
        q_nominal=pin.neutral(probe_model),
        q_reference=reference[0],
        reference=reference,
        com_margin=com_margin,
        collision_rig=collision_rig,
        sample_weights=sample_weights,
        seed=seed,
        log=log,
    )
    # re-clip the reference against the planner's own bounds and report
    reference, info = load_reference_npz(
        reference_npz,
        planning_joint_names,
        lo=planner._bounds_low,
        hi=planner._bounds_high,
    )
    return planner, reference, info

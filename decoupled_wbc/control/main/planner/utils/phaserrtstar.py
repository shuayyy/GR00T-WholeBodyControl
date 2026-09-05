"""PhaseRRTstar over the upper-body joints: plans in (q, alpha), alpha being the
phase along a reference trajectory."""

import numpy as np
import ompl.base as ob
import ompl.geometric as og
import ompl.util as ou


def phase_defaults(arclength):
    """Planner parameters scaled by the reference arclength."""
    w_alphastate = 0.756 * arclength
    return {
        "w_alphastate": w_alphastate,
        "range": 0.0646 * arclength,
        "sigma": 0.0018 * arclength,
        "d_alpha_min": 0.005,
        "d_alpha_max": 0.1,
        "uniform_fraction": 0.0,
        "goal_bias": 0.05,
        "phase_grid": 1000.0,
        "goal_threshold": 0.5 * w_alphastate * 0.001,
    }


def reference_arclength(reference):
    reference = np.asarray(reference, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(reference, axis=0), axis=1)))


class PhaseRRTstarPlanner:
    """PhaseRRTstar guided by a reference trajectory; same interface as OMPLGeometricPlanner.

    TODO(ablation): CoM stability check in validity_checker (no effect on test1-3).
    """

    def __init__(
        self,
        robot,
        reference,
        validity_resolution=0.01,
        rewire_factor=1.0,
        phase_params=None,
        log=True,
    ):
        self.name = "PhaseRRTstar"
        self.robot = robot
        self.model = robot.model
        self.n_dof = robot.n_joints
        self.joint_limits = robot.joint_limits
        self.validity_resolution = validity_resolution
        self.rewire_factor = rewire_factor

        self.reference = np.asarray(reference, dtype=float)
        if self.reference.ndim != 2 or self.reference.shape[1] != self.n_dof:
            raise ValueError(
                f"reference must be (N, {self.n_dof}), got {self.reference.shape}"
            )
        if self.reference.shape[0] < 2:
            raise ValueError("reference needs at least two waypoints")

        self.params = phase_defaults(reference_arclength(self.reference))
        self.params.update(phase_params or {})
        self.w_alphastate = float(self.params["w_alphastate"])

        if not log:
            ou.setLogLevel(ou.LOG_ERROR)

        self.ss, self.si = self.set_up_ompl()
        self.pdef = self.ss.getProblemDefinition()
        self.planner = self.ss.getPlanner()
        self.last_plan_stats = {}

    def set_up_ompl(self):
        """Compound ``(q, alpha)`` space; alpha is the last dimension."""
        joints = ob.RealVectorStateSpace(self.n_dof)
        bounds = ob.RealVectorBounds(self.n_dof)
        for i in range(self.n_dof):
            low = self.joint_limits[0][i]
            high = self.joint_limits[1][i]
            if low == -np.inf:
                low = -2 * np.pi
            if high == np.inf:
                high = 2 * np.pi
            bounds.setLow(i, low)
            bounds.setHigh(i, high)
        joints.setBounds(bounds)

        alpha = ob.RealVectorStateSpace(1)
        alpha_bounds = ob.RealVectorBounds(1)
        alpha_bounds.setLow(0.0)
        alpha_bounds.setHigh(1.0)
        alpha.setBounds(alpha_bounds)

        space = ob.CompoundStateSpace()
        space.addSubspace(joints, 1.0)
        space.addSubspace(alpha, self.w_alphastate)
        space.lock()

        ss = og.SimpleSetup(space)
        si = ss.getSpaceInformation()
        si.setStateValidityCheckingResolution(self.validity_resolution)
        ss.setStateValidityChecker(self.validity_checker)

        rows = [list(map(float, row)) for row in self.reference]
        planner = og.PhaseRRTstar(si)
        planner.setReference(rows)
        planner.setRange(float(self.params["range"]))
        planner.setSampleSigma(float(self.params["sigma"]))
        planner.setDAlphaMin(float(self.params["d_alpha_min"]))
        planner.setDAlphaMax(float(self.params["d_alpha_max"]))
        planner.setUniformFraction(float(self.params["uniform_fraction"]))
        planner.setGoalBias(float(self.params["goal_bias"]))
        planner.setPhaseGrid(float(self.params["phase_grid"]))
        planner.setRewireFactor(float(self.rewire_factor))

        objective = ob.PhaseSimilarityObjective(si, False)
        objective.setReference(rows)
        objective.setAlphaScale(1.0)
        ss.setOptimizationObjective(objective)
        ss.setPlanner(planner)
        return ss, si

    def validity_checker(self, state):
        """Joint bounds and collision."""
        joints = state[0]
        q = np.array([joints[i] for i in range(self.n_dof)], dtype=float)
        if not self.si.satisfiesBounds(state):
            return False
        self.robot.set_joint_qpos(q)
        return not self.robot.in_contact()

    def plan(
        self,
        start,
        goal,
        goal_type="upper_body",
        timeout=10.0,
        smooth_path=False,
        shortcut_path=False,
    ):
        """Plan from start to goal; returns (N, n_dof) waypoints.
        Path simplification is ignored (it would break the monotone alpha)."""
        if goal_type != "upper_body":
            raise ValueError(
                f"PhaseRRTstar only supports goal_type 'upper_body', got '{goal_type}'"
            )
        start = np.asarray(start, dtype=float).reshape(-1)
        goal = np.asarray(goal, dtype=float).reshape(-1)
        for label, q in (("start", start), ("goal", goal)):
            if q.shape[0] != self.n_dof:
                raise ValueError(f"{label} has {q.shape[0]} DoF, expected {self.n_dof}")

        start_state = self.si.allocState()
        goal_state = self.si.allocState()
        for state, q in ((start_state, start), (goal_state, goal)):
            joints = state[0]
            for i in range(self.n_dof):
                joints[i] = float(q[i])
        start_state[1][0] = 0.0
        goal_state[1][0] = 1.0
        self.ss.setStartAndGoalStates(
            start_state, goal_state, float(self.params["goal_threshold"])
        )

        self.ss.setup()
        status = self.ss.solve(float(timeout))
        status_str = status.asString()

        waypoints = np.array([start])
        alphas = np.zeros(1)
        if status_str == "Exact solution":
            states = self.ss.getSolutionPath().getStates()
            waypoints = np.array(
                [[s[0][i] for i in range(self.n_dof)] for s in states], dtype=float
            )
            alphas = np.array([s[1][0] for s in states], dtype=float)

        self.last_plan_stats = {
            "status": status_str,
            "num_states": int(waypoints.shape[0]),
            "alpha_first": float(alphas[0]),
            "alpha_final": float(alphas[-1]),
            "alpha_monotone": bool(np.all(np.diff(alphas) >= -1e-9)),
            "arclength": reference_arclength(self.reference),
            "params": dict(self.params),
        }
        self.ss.clear()
        if status_str != "Exact solution":
            raise RuntimeError(f"PhaseRRTstar returned '{status_str}', not an exact solution")
        return waypoints

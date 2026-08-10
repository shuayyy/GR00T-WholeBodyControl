"""Foot-placement constraint: keep both feet at reference poses.

Frame and ordering conventions (stated once, relied on everywhere)
------------------------------------------------------------------
* Per-foot error is ``pin.log6(target.actInv(current)).vector`` -- the SE(3)
  logarithm of the pose error, expressed in the LOCAL (body) frame of the
  error transform.  This is the TSR pose-error-as-6-vector formulation
  (Berenson et al., ICRA 2009).
* Pinocchio's ``Motion.vector`` ordering is ``[linear(3), angular(3)]``: for
  each foot, rows 0-2 are translational, rows 3-5 rotational.
* Feet are stacked left-then-right: rows 0-5 left foot, rows 6-11 right foot.
* The analytic Jacobian uses ``pin.getFrameJacobian(..., pin.LOCAL)``.  With
  a constant target, the body twist of the error transform equals the body
  twist of the current foot frame, and Pinocchio's ``Jlog6`` maps LOCAL
  twists to error-vector rates -- so ``Jlog6(T_err) @ J_frame_LOCAL`` is the
  frame-consistent chain rule.  Mixing LOCAL and WORLD here fails silently
  (wrong numbers, no exception); the test suite validates the analytic
  Jacobian against central differences to catch exactly that.

Targets are derived from a reference planning configuration passed to
``__init__`` -- never hardcoded pose literals.  (MCVAMP's digit_test.py
hardcodes its ``eef_transforms`` with no assertion tying them to the start
configuration, a silent-desync trap this module deliberately avoids: targets
here are FK-consistent with the reference config by construction.)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pinocchio as pin

from .base_constraint import Constraint
from .embedding import PlanningEmbedder
from .planner_constraints import G1Constraints

# SE(3) pose error dimension (3 translational + 3 rotational rows).
_SE3_ERROR_DIM = 6


def _copy_se3(pose: pin.SE3) -> pin.SE3:
    """Return a defensive copy of an SE3 (frame placements alias model data)."""
    return pin.SE3(pose.rotation.copy(), pose.translation.copy())


def feet_targets_from_config(
    robot_model,
    embedder: PlanningEmbedder,
    q_plan: Sequence[float],
) -> tuple[pin.SE3, pin.SE3]:
    """Return (left, right) foot target poses at a planning configuration.

    The targets are the forward-kinematics foot poses at ``q_plan`` embedded
    with ``embedder``, copied so they stay valid when the model data buffer
    is overwritten by later FK calls.
    """
    backend = G1Constraints(robot_model)
    left, right = backend.compute_feet_poses(embedder(q_plan))
    return _copy_se3(left), _copy_se3(right)


class FeetConstraint(Constraint):
    """Equality constraint pinning both feet to reference poses.

    ``n_rows == 12``: one 6-D pose error per foot, left then right, each
    ``[linear(3), angular(3)]`` (see module docstring for conventions).
    """

    def __init__(
        self,
        robot_model,
        embedder: PlanningEmbedder,
        q_reference_plan: Sequence[float],
    ):
        self._robot_model = robot_model
        self._embedder = embedder
        self._backend = G1Constraints(robot_model)
        self._target_left, self._target_right = feet_targets_from_config(
            robot_model, embedder, q_reference_plan
        )

        model = robot_model.pinocchio_wrapper.model
        self._frame_ids = (
            model.getFrameId(G1Constraints.LEFT_FOOT_FRAME),
            model.getFrameId(G1Constraints.RIGHT_FOOT_FRAME),
        )

    # ------------------------------------------------------------------ API

    @property
    def n_rows(self) -> int:
        return 2 * _SE3_ERROR_DIM

    @property
    def n_plan(self) -> int:
        return self._embedder.n_plan

    @property
    def targets(self) -> tuple[pin.SE3, pin.SE3]:
        """(left, right) target poses (copies)."""
        return _copy_se3(self._target_left), _copy_se3(self._target_right)

    def error(self, q_plan: Sequence[float]) -> np.ndarray:
        """Stacked 12-D pose error at ``q_plan`` (fresh array)."""
        left, right = self._backend.compute_feet_poses(
            self._embedder(q_plan)
        )
        return np.concatenate(
            [
                pin.log6(self._target_left.actInv(left)).vector,
                pin.log6(self._target_right.actInv(right)).vector,
            ]
        ).astype(float)

    def jacobian(self, q_plan: Sequence[float]) -> np.ndarray:
        """Analytic ``(12, n_plan)`` Jacobian (LOCAL-frame chain rule)."""
        wrapper = self._robot_model.pinocchio_wrapper
        model, data = wrapper.model, wrapper.data

        q_full = self._embedder(q_plan)
        pin.computeJointJacobians(model, data, q_full)
        pin.updateFramePlacements(model, data)

        blocks = []
        for target, frame_id in zip(
            (self._target_left, self._target_right), self._frame_ids
        ):
            error_transform = target.actInv(data.oMf[frame_id])
            frame_jacobian = pin.getFrameJacobian(
                model, data, frame_id, pin.ReferenceFrame.LOCAL
            )
            blocks.append(pin.Jlog6(error_transform) @ frame_jacobian)

        return self._embedder.select_columns(np.vstack(blocks))

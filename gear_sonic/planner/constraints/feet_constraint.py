"""Foot-placement constraint: keep both feet at reference poses.

Per-foot error is the SE(3) log of the pose error in the local frame; 12 rows,
left foot then right, each ``[linear(3), angular(3)]``.
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
    """(left, right) foot poses at ``q_plan``, copied so later FK calls
    cannot overwrite them."""
    backend = G1Constraints(robot_model)
    left, right = backend.compute_feet_poses(embedder(q_plan))
    return _copy_se3(left), _copy_se3(right)


class FeetConstraint(Constraint):
    """Pins both feet to reference poses: 12 rows, left foot then right,
    each ``[linear(3), angular(3)]``."""

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

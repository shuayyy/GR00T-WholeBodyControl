"""CoM stability constraint: keep the CoM ``margin`` inside the support
polygon.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base_constraint import Constraint, numeric_jacobian
from .embedding import PlanningEmbedder
from .planner_constraints import G1Constraints


class CoMConstraint(Constraint):

    def __init__(
        self,
        robot_model,
        embedder: PlanningEmbedder,
        margin: float = 0.05,
    ):
        self._embedder = embedder
        self._backend = G1Constraints(robot_model)
        self._margin = float(margin)


    @property
    def n_rows(self) -> int:
        return 1

    @property
    def n_plan(self) -> int:
        return self._embedder.n_plan

    @property
    def margin(self) -> float:
        """Required CoM clearance from the polygon edge, in metres."""
        return self._margin

    def margin_at(self, q_plan: Sequence[float]) -> float:
        """CoM distance to the nearest polygon edge; negative outside."""
        return float(
            self._backend.compute_stability(self._embedder(q_plan))
        )

    def is_stable(self, q_plan: Sequence[float]) -> bool:
        """True if the pose clears the margin."""
        return self.margin_at(q_plan) >= self._margin

    def error(self, q_plan: Sequence[float]) -> np.ndarray:
        """Constraint error: 0 when stable, else the margin violation."""
        violation = self._margin - self.margin_at(q_plan)
        return np.array([max(0.0, violation)], dtype=float)

    def jacobian(self, q_plan: Sequence[float]) -> np.ndarray:
        """Error gradient. Zero when stable; numeric otherwise, since the
        active polygon edge changes with the pose."""
        if self.is_stable(q_plan):
            return np.zeros((self.n_rows, self.n_plan), dtype=float)
        return numeric_jacobian(self, np.asarray(q_plan, dtype=float))

"""Static-stability constraint: keep the CoM inside the support polygon.

Hinge formulation
-----------------
``error(q) = max(0.0, margin - stability(q))`` where ``stability`` is
``G1Constraints.compute_stability`` -- the signed XY distance from the CoM
projection to the nearest support-polygon edge (positive inside).  The error
is EXACTLY ``0.0`` whenever the CoM is at least ``margin`` inside the polygon
and grows linearly with the violation outside.

This mirrors the hinge gate in MCVAMP's Digit CoM constraint
(``digit.hh``, the ``blend(0., 1., ...)`` at the constraint-error site):
formulated as a one-row equality with a one-sided residual, a violating
sample gets *repaired by projection* (driven back to the boundary) rather
than discarded, so Stage 2 can route this constraint through OMPL's
projection operator unchanged.

Jacobian: numeric by design
---------------------------
The gradient of the stability margin involves the geometry of a convex hull
whose active edge -- and vertex set -- changes with the configuration, which
makes the analytic gradient fiddly and brittle.  The numeric central
difference of the hinged error is the intended default here, not an
oversight.  Inside the satisfied region the hinge is identically zero, so the
Jacobian returned there is the zero matrix -- a subgradient at the
non-differentiable boundary point.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base_constraint import Constraint, numeric_jacobian
from .embedding import PlanningEmbedder
from .planner_constraints import G1Constraints


class CoMConstraint(Constraint):
    """One-row hinge constraint on the static stability margin."""

    def __init__(
        self,
        robot_model,
        embedder: PlanningEmbedder,
        margin: float = 0.05,
    ):
        self._embedder = embedder
        self._backend = G1Constraints(robot_model)
        self._margin = float(margin)

    # ------------------------------------------------------------------ API

    @property
    def n_rows(self) -> int:
        return 1

    @property
    def n_plan(self) -> int:
        return self._embedder.n_plan

    @property
    def margin(self) -> float:
        """Required stability margin in meters."""
        return self._margin

    def margin_at(self, q_plan: Sequence[float]) -> float:
        """Signed stability margin at ``q_plan``.

        Equals ``G1Constraints.compute_stability`` on the embedded full
        configuration: positive inside the support polygon, negative outside.
        """
        return float(
            self._backend.compute_stability(self._embedder(q_plan))
        )

    def is_stable(self, q_plan: Sequence[float]) -> bool:
        """Cheap rejection test: True exactly when ``error(q_plan) == 0.0``."""
        return self.margin_at(q_plan) >= self._margin

    def error(self, q_plan: Sequence[float]) -> np.ndarray:
        """Hinged violation, shape ``(1,)``; exactly 0.0 when stable."""
        violation = self._margin - self.margin_at(q_plan)
        return np.array([max(0.0, violation)], dtype=float)

    def jacobian(self, q_plan: Sequence[float]) -> np.ndarray:
        """``(1, n_plan)`` Jacobian of the hinged error.

        Zero matrix inside the satisfied region (subgradient of the hinge);
        central-difference numeric Jacobian of the hinged error otherwise
        (see module docstring for why numeric is the intended default).
        """
        if self.is_stable(q_plan):
            return np.zeros((self.n_rows, self.n_plan), dtype=float)
        return numeric_jacobian(self, np.asarray(q_plan, dtype=float))

"""Composition of constraints by row stacking.

Mirrors MCVAMP's ``ComposableConstraints`` (composable_constraint.hh):
composition is structural, not a framework -- children are plain
:class:`~gear_sonic_planner.planner.constraints.base_constraint.Constraint`
objects composed by concatenating errors and vertically stacking Jacobians,
in constructor order.

Empty composition is a construction error by choice: a constraint stack with
nothing in it almost always indicates a wiring bug upstream, and without
children ``n_plan`` would be undefined, so ``ComposedConstraint()`` raises
rather than yielding ``n_rows == 0``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base_constraint import Constraint


class ComposedConstraint(Constraint):
    """Stack several constraints into one, preserving constructor order.

    ``n_rows`` is the sum of the children's, ``error`` their concatenation,
    ``jacobian`` their vertical stack.  All children must agree on
    ``n_plan``.
    """

    def __init__(self, *constraints: Constraint):
        if not constraints:
            raise ValueError(
                "ComposedConstraint requires at least one child constraint "
                "(empty composition is a construction error by design; see "
                "module docstring)"
            )
        n_plan = constraints[0].n_plan
        for child in constraints[1:]:
            if child.n_plan != n_plan:
                raise ValueError(
                    f"Constraints disagree on planning dimension: "
                    f"{type(constraints[0]).__name__} has n_plan = {n_plan}, "
                    f"{type(child).__name__} has n_plan = {child.n_plan}"
                )
        self._constraints = tuple(constraints)
        self._n_plan = n_plan
        self._names = self._unique_names(self._constraints)

    @staticmethod
    def _unique_names(constraints: tuple[Constraint, ...]) -> tuple[str, ...]:
        """Per-child keys: the class name, ``[i]``-suffixed on duplicates."""
        class_names = [type(child).__name__ for child in constraints]
        return tuple(
            name if class_names.count(name) == 1 else f"{name}[{i}]"
            for i, name in enumerate(class_names)
        )

    # ------------------------------------------------------------------ API

    @property
    def n_rows(self) -> int:
        return sum(child.n_rows for child in self._constraints)

    @property
    def n_plan(self) -> int:
        return self._n_plan

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        """The child constraints, in composition order."""
        return self._constraints

    def error(self, q_plan: Sequence[float]) -> np.ndarray:
        return np.concatenate(
            [child.error(q_plan) for child in self._constraints]
        )

    def jacobian(self, q_plan: Sequence[float]) -> np.ndarray:
        return np.vstack(
            [child.jacobian(q_plan) for child in self._constraints]
        )

    def per_constraint_error(
        self, q_plan: Sequence[float]
    ) -> dict[str, np.ndarray]:
        """Each child's error under a key identifying that child.

        This is the first diagnostic to reach for when projection stalls:
        it answers *which* constraint is failing to converge, which the
        stacked residual cannot.
        """
        return {
            name: child.error(q_plan)
            for name, child in zip(self._names, self._constraints)
        }

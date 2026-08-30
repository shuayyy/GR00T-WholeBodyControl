"""Abstract constraint interface, mirroring ``ompl::base::Constraint``.

NumPy only -- nothing here imports OMPL, so the adapter stays a thin shim.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class Constraint(ABC):
    """Vector-valued equality constraint ``error(q_plan) == 0``.

    ``q_plan`` is read-only; return fresh arrays.
    """

    @property
    @abstractmethod
    def n_rows(self) -> int:
        """Number of scalar constraint rows (the constraint co-dimension)."""

    @property
    @abstractmethod
    def n_plan(self) -> int:
        """Dimension of the planning space the constraint is defined over."""

    @abstractmethod
    def error(self, q_plan: Sequence[float]) -> np.ndarray:
        """Return the residual at ``q_plan``, shape ``(n_rows,)``."""

    @abstractmethod
    def jacobian(self, q_plan: Sequence[float]) -> np.ndarray:
        """Return ``d error / d q_plan``, shape ``(n_rows, n_plan)``."""

    def is_satisfied(self, q_plan: Sequence[float], tol: float = 1e-3) -> bool:
        """Whether ``||error(q_plan)|| <= tol``, matching
        ``ompl::base::Constraint::isSatisfied``."""
        return float(np.linalg.norm(self.error(q_plan))) <= tol


def numeric_jacobian(
    constraint: Constraint,
    q_plan: Sequence[float],
    eps: float = 1e-6,
) -> np.ndarray:
    """Central-difference Jacobian of ``constraint.error``, used to validate
    the analytic ones. O(eps^2), so eps 1e-6 agrees to ~1e-5."""
    q_plan = np.asarray(q_plan, dtype=float)
    if q_plan.ndim != 1:
        raise ValueError(f"q_plan must be 1-D, got shape {q_plan.shape}")

    J = np.empty((constraint.n_rows, q_plan.shape[0]), dtype=float)
    for i in range(q_plan.shape[0]):
        q_hi = q_plan.copy()
        q_lo = q_plan.copy()
        q_hi[i] += eps
        q_lo[i] -= eps
        J[:, i] = (constraint.error(q_hi) - constraint.error(q_lo)) / (2.0 * eps)
    return J

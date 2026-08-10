"""Constraint layer for constrained whole-body planning (Stage 1).

NumPy/Pinocchio only -- no OMPL, no MuJoCo, no planner imports.  The
:class:`Constraint` interface mirrors ``ompl::base::Constraint`` so the
Stage-2 adapter stays thin.

``stability_visualizer`` is deliberately not imported here: it is a
standalone visualization tool and must not become an import-time dependency
of the constraint layer.
"""

from .base_constraint import Constraint, numeric_jacobian
from .com_constraint import CoMConstraint
from .composable_constraint import ComposedConstraint
from .embedding import PlanningEmbedder, check_embedder
from .feet_constraint import FeetConstraint, feet_targets_from_config
from .planner_constraints import G1Constraints

__all__ = [
    "CoMConstraint",
    "ComposedConstraint",
    "Constraint",
    "FeetConstraint",
    "G1Constraints",
    "PlanningEmbedder",
    "check_embedder",
    "feet_targets_from_config",
    "numeric_jacobian",
]

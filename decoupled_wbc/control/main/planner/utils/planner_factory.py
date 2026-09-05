"""Build a planner (OMPLGeometricPlanner or PhaseRRTstarPlanner) from its name."""

from __future__ import annotations

import numpy as np

from .ompl_planning import OMPLGeometricPlanner
from .phaserrtstar import PhaseRRTstarPlanner

PHASE_PLANNER = "PhaseRRTstar"


def make_planner(
    name: str,
    robot,
    reference: np.ndarray | None = None,
    validity_resolution: float = 0.01,
    log: bool = True,
):
    """name: an ompl.geometric planner or "PhaseRRTstar" (needs reference)."""
    if name == PHASE_PLANNER:
        if reference is None:
            raise ValueError(
                f"{PHASE_PLANNER} needs a reference trajectory: "
                "set use_reference=True and reference_trajectory_path"
            )
        return PhaseRRTstarPlanner(
            robot, reference, validity_resolution=validity_resolution, log=log
        )
    return OMPLGeometricPlanner(
        robot,
        planner=name,
        validity_resolution=validity_resolution,
        log=log,
        reference=reference,
    )

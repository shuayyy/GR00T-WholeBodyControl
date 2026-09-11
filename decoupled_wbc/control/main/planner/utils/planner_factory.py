"""Build a planner (OMPLGeometricPlanner or PhaseRRTstarPlanner) from its name."""

from __future__ import annotations

import numpy as np

from .ompl_planning import OMPLGeometricPlanner
from .phaserrtstar import PhaseRRTstarPlanner, phase_defaults, reference_arclength

PHASE_PLANNER = "PhaseRRTstar"


def make_planner(
    name: str,
    robot,
    reference: np.ndarray | None = None,
    validity_resolution: float = 0.01,
    log: bool = True,
    phase_sigma_scale: float = 1.0,
    validation_robot=None,
):
    """name: an ompl.geometric planner or "PhaseRRTstar" (needs reference).

    ``phase_sigma_scale`` widens PhaseRRTstar's sampling tube (no effect on other
    planners).  ``validation_robot`` is the scene a smoothed path is checked
    against, for every planner."""
    if name == PHASE_PLANNER:
        if reference is None:
            raise ValueError(
                f"{PHASE_PLANNER} needs a reference trajectory: "
                "set use_reference=True and reference_trajectory_path"
            )
        params = None
        if phase_sigma_scale != 1.0:
            base = phase_defaults(reference_arclength(reference))
            params = {"sigma": base["sigma"] * phase_sigma_scale}
        return PhaseRRTstarPlanner(
            robot,
            reference,
            validity_resolution=validity_resolution,
            log=log,
            phase_params=params,
            validation_robot=validation_robot,
        )
    return OMPLGeometricPlanner(
        robot,
        planner=name,
        validity_resolution=validity_resolution,
        log=log,
        reference=reference,
        validation_robot=validation_robot,
    )

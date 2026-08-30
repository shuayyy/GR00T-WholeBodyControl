"""Mapping between the planning vector and the full Pinocchio configuration.

The free-flyer root is the only multi-DoF joint admitted: 7 entries in the
full configuration, 6 in the planning vector (position + so(3) rotvec).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pinocchio as pin

# Free-flyer joint signature: 7 configuration entries (xyz + quaternion),
# 6 tangent entries, 6 planning slots (xyz + rotation vector).
_FREE_FLYER_SHORTNAME = "JointModelFreeFlyer"
_BASE_NQ = 7
_BASE_NV = 6
_BASE_PLAN_WIDTH = 6


class PlanningEmbedder:
    """Embed a planning vector into a full Pinocchio ``q``.

    ``planning_joint_names`` must be 1-DoF joints or the free-flyer root;
    anything else raises.  ``q_nominal`` supplies the non-planning entries.
    """

    def __init__(
        self,
        pin_model,
        planning_joint_names: Sequence[str],
        q_nominal: np.ndarray,
    ):
        self._model = pin_model
        self._planning_joint_names = list(planning_joint_names)

        q_nominal = np.asarray(q_nominal, dtype=float)
        if q_nominal.ndim != 1 or q_nominal.shape[0] != pin_model.nq:
            raise ValueError(
                f"q_nominal has length "
                f"{q_nominal.shape[0] if q_nominal.ndim == 1 else q_nominal.shape}, "
                f"but the model requires nq = {pin_model.nq}"
            )
        self._q_nominal = q_nominal.copy()

        seen: set[str] = set()
        revolute_plan_positions: list[int] = []
        revolute_idx_q: list[int] = []
        revolute_idx_v: list[int] = []
        column_idx: list[int] = []
        self._base_idx_q: int | None = None
        self._base_idx_v: int | None = None
        self._base_plan_offset: int | None = None
        offset = 0
        for name in self._planning_joint_names:
            if name in seen:
                raise ValueError(
                    f"Duplicate joint '{name}' in planning_joint_names"
                )
            seen.add(name)
            if not pin_model.existJointName(name):
                raise KeyError(
                    f"Joint '{name}' does not exist in the Pinocchio model"
                )
            joint = pin_model.joints[pin_model.getJointId(name)]
            if joint.shortname() == _FREE_FLYER_SHORTNAME:
                # The one permitted multi-DoF joint: the floating base.
                if self._base_plan_offset is not None:
                    raise ValueError(
                        f"Free-flyer joint '{name}' listed more than once "
                        f"in planning_joint_names"
                    )
                self._base_idx_q = joint.idx_q
                self._base_idx_v = joint.idx_v
                self._base_plan_offset = offset
                column_idx.extend(
                    range(joint.idx_v, joint.idx_v + _BASE_NV)
                )
                offset += _BASE_PLAN_WIDTH
            elif joint.nq == 1:
                revolute_plan_positions.append(offset)
                revolute_idx_q.append(joint.idx_q)
                revolute_idx_v.append(joint.idx_v)
                column_idx.append(joint.idx_v)
                offset += 1
            else:
                raise ValueError(
                    f"Joint '{name}' has nq = {joint.nq}; only 1-DoF joints "
                    f"and the free-flyer base can enter the planning space "
                    f"(other multi-DoF joints must stay in q_nominal)"
                )

        self._n_plan = offset
        self._revolute_plan_positions = np.asarray(
            revolute_plan_positions, dtype=int
        )
        self._revolute_idx_q = np.asarray(revolute_idx_q, dtype=int)
        self._revolute_idx_v = np.asarray(revolute_idx_v, dtype=int)
        self._column_idx = np.asarray(column_idx, dtype=int)


    @property
    def n_plan(self) -> int:
        """Dimension of the planning space."""
        return self._n_plan

    @property
    def planning_joint_names(self) -> list[str]:
        """Planning joint names, in planning order (copy)."""
        return list(self._planning_joint_names)

    @property
    def has_base(self) -> bool:
        """Whether the free-flyer base is part of the planning space."""
        return self._base_plan_offset is not None

    @property
    def base_plan_slice(self) -> slice | None:
        """Planning-vector slice of the base's 6 slots (None without base)."""
        if self._base_plan_offset is None:
            return None
        return slice(
            self._base_plan_offset,
            self._base_plan_offset + _BASE_PLAN_WIDTH,
        )

    @property
    def revolute_plan_positions(self) -> np.ndarray:
        """Planning-vector positions of the 1-DoF joints (copy)."""
        return self._revolute_plan_positions.copy()

    @property
    def revolute_idx_q(self) -> np.ndarray:
        """Configuration indices of the 1-DoF planning joints (copy)."""
        return self._revolute_idx_q.copy()

    @property
    def idx_q(self) -> np.ndarray:
        """Configuration indices of the planning slots (copy).

        Raises with the base in the planning space: its 3 rotation-vector
        slots have no configuration index, the quaternion having 4.
        """
        if self.has_base:
            raise RuntimeError(
                "idx_q is undefined with the free-flyer base in the "
                "planning space (rotation-vector slots have no "
                "configuration index); use revolute_idx_q / base_plan_slice"
            )
        return self._revolute_idx_q.copy()

    @property
    def idx_v(self) -> np.ndarray:
        """Tangent-space indices of the planning slots (copy).

        See :attr:`idx_q` for why this raises when the base is planned.
        """
        if self.has_base:
            raise RuntimeError(
                "idx_v is undefined with the free-flyer base in the "
                "planning space; use base_plan_slice and select_columns"
            )
        return self._revolute_idx_v.copy()

    def __call__(self, q_plan: Sequence[float]) -> np.ndarray:
        """Full configuration realizing ``q_plan``, as a fresh array."""
        q_plan = np.asarray(q_plan, dtype=float)
        if q_plan.ndim != 1 or q_plan.shape[0] != self._n_plan:
            raise ValueError(
                f"q_plan has shape {q_plan.shape}, expected ({self._n_plan},)"
            )
        q_full = self._q_nominal.copy()
        q_full[self._revolute_idx_q] = q_plan[self._revolute_plan_positions]
        if self._base_plan_offset is not None:
            offset, idx_q = self._base_plan_offset, self._base_idx_q
            q_full[idx_q : idx_q + 3] = q_plan[offset : offset + 3]
            rotation = pin.exp3(q_plan[offset + 3 : offset + _BASE_PLAN_WIDTH])
            # Pinocchio stores the base quaternion as xyzw.
            q_full[idx_q + 3 : idx_q + _BASE_NQ] = pin.Quaternion(
                rotation
            ).coeffs()
        return q_full

    def extract(self, q_full: Sequence[float]) -> np.ndarray:
        """Return the planning vector contained in a full configuration."""
        q_full = np.asarray(q_full, dtype=float)
        if q_full.ndim != 1 or q_full.shape[0] != self._model.nq:
            raise ValueError(
                f"q_full has shape {q_full.shape}, expected ({self._model.nq},)"
            )
        q_plan = np.empty(self._n_plan, dtype=float)
        q_plan[self._revolute_plan_positions] = q_full[self._revolute_idx_q]
        if self._base_plan_offset is not None:
            offset, idx_q = self._base_plan_offset, self._base_idx_q
            q_plan[offset : offset + 3] = q_full[idx_q : idx_q + 3]
            x, y, z, w = q_full[idx_q + 3 : idx_q + _BASE_NQ]
            q_plan[offset + 3 : offset + _BASE_PLAN_WIDTH] = pin.log3(
                pin.Quaternion(w, x, y, z).matrix()
            )
        return q_plan

    def select_columns(self, J_full: np.ndarray) -> np.ndarray:
        """Select the planning columns of a full-model Jacobian.

        Selected with ``idx_v``, not ``idx_q``: on a floating-base model the
        two differ and ``idx_q`` returns wrong columns silently.
        """
        J_full = np.asarray(J_full, dtype=float)
        if J_full.ndim != 2 or J_full.shape[1] != self._model.nv:
            raise ValueError(
                f"J_full has shape {J_full.shape}, expected "
                f"(n_rows, {self._model.nv})"
            )
        return J_full[:, self._column_idx].copy()


def check_embedder(
    embedder: PlanningEmbedder,
    q_plan: Sequence[float],
    atol: float = 1e-12,
) -> None:
    """Assert that ``extract(embed(q_plan))`` reproduces ``q_plan``.

    Exact without the base; with it, the rotation vector round-trips through
    ``exp3``/``log3``, hence the 1e-12 tolerance.
    """
    q_plan = np.asarray(q_plan, dtype=float)
    round_trip = embedder.extract(embedder(q_plan))
    max_diff = float(np.max(np.abs(round_trip - q_plan)))
    if max_diff > atol:
        raise AssertionError(
            f"Embedder round trip failed: max abs diff {max_diff}"
        )

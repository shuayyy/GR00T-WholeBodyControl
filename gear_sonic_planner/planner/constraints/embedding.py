"""Mapping between the planning vector and the full Pinocchio configuration.

Index convention
----------------
Pinocchio distinguishes configuration-space indices (``idx_q``, into ``q`` of
length ``model.nq``) from tangent-space indices (``idx_v``, into velocity /
Jacobian columns of length ``model.nv``).  For 1-DoF joints on a fixed-base
model the two coincide, but on a model with a floating base every joint after
the base has ``idx_q == idx_v + 1`` (the base occupies 7 configuration entries
-- xyz + unit quaternion -- but only 6 tangent entries).  This module
therefore uses:

* ``idx_q`` for reading/writing configuration vectors (:meth:`__call__`,
  :meth:`extract`);
* ``idx_v`` for selecting Jacobian columns (:meth:`select_columns`).

Floating-base representation (Stage 2)
--------------------------------------
The free-flyer root joint is the ONLY multi-DoF joint admitted into the
planning space; every other ``joint.nq != 1`` joint still raises.  In the
full configuration the base occupies 7 entries (``[x, y, z, qx, qy, qz, qw]``,
quaternion stored xyzw); in the planning vector it contributes exactly 6
slots so the planning vector stays a plain element of R^n with no quaternion
(and no unit-norm side constraint) inside it:

    ``[p_x, p_y, p_z, r_x, r_y, r_z]``

world position plus an so(3) rotation vector (exponential coordinates,
``r = log3(R)``).  This representation was chosen because:

* translation round-trips exactly and rotation to machine precision
  (``log3(exp3(r)) == r`` up to float rounding -- exact-equality round-trip
  tests therefore use an atol of 1e-12 when the base is in the planning
  space);
* its only chart singularity is at rotation angle pi, where ``log3`` is
  ambiguous (antipodal rotations) and ``Jexp3`` loses rank.  A standing
  humanoid base operates far from a pi flip.  Roll-pitch-yaw was rejected
  because its gimbal lock sits at pitch pi/2, which a strongly leaning torso
  can approach; a quaternion was rejected because it is 4 parameters plus a
  unit-norm constraint, which is exactly what a plain-R^n planning space
  cannot carry.

Jacobian caveat: :meth:`select_columns` selects the base's 6 tangent-space
columns (``idx_v``) verbatim, so with the base in the planning space the base
block of a selected Jacobian is the LOCAL-twist Jacobian ``df/dnu``, not the
chart derivative ``df/d(p, r)``.  The exact relation is
``J_p = J_lin @ R^T`` and ``J_r = J_ang @ Jexp3(r)``; the two coincide at
base identity and differ by O(|r|) near it, which downstream Newton-style
projection tolerates as a Gauss-Newton approximation.  Making the selection
exact would require passing the current configuration into
``select_columns`` (and through every caller) -- deliberately out of scope
for Stage 2 and flagged for Stage 3.
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
    """Embed a low-dimensional planning vector into a full Pinocchio ``q``.

    Parameters
    ----------
    pin_model:
        The ``pinocchio.Model`` the configurations refer to.
    planning_joint_names:
        Names of the joints that make up the planning space, in planning
        order.  Every name must exist in ``pin_model`` and refer either to a
        1-DoF joint (``joint.nq == 1``, one planning slot) or to the
        free-flyer root joint (6 planning slots -- see module docstring).
        Any other multi-DoF joint raises.
    q_nominal:
        Full configuration of length ``pin_model.nq`` supplying the values of
        every non-planning entry (hands, and any other joint not planned).
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
                f"q_nominal has length {q_nominal.shape[0] if q_nominal.ndim == 1 else q_nominal.shape}, "
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

    # ------------------------------------------------------------------ API

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

        Only defined when the planning space contains 1-DoF joints
        exclusively: the base's 3 rotation-vector slots have no
        configuration index (the quaternion has 4).  Raises with the base
        in the planning space rather than returning something misleading.
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
        """Return the full configuration realizing ``q_plan``.

        Non-planning entries are taken verbatim from ``q_nominal``.  The
        result is a fresh array; the input is never aliased or mutated.
        """
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

        ``J_full`` must have ``model.nv`` columns (tangent space); the
        planning columns are selected with ``idx_v``, NOT ``idx_q`` -- on a
        floating-base model the two differ and using ``idx_q`` would silently
        return wrong columns.  The base contributes its 6 tangent columns
        verbatim; see the module docstring for the chart-derivative caveat.
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

    Exact for 1-DoF-only planning spaces; with the base in the planning
    space the rotation vector passes through ``exp3``/``log3``, so equality
    holds to float rounding (hence the 1e-12 tolerance rather than
    bit-exactness).
    """
    q_plan = np.asarray(q_plan, dtype=float)
    round_trip = embedder.extract(embedder(q_plan))
    max_diff = float(np.max(np.abs(round_trip - q_plan)))
    if max_diff > atol:
        raise AssertionError(
            f"Embedder round trip failed: max abs diff {max_diff}"
        )

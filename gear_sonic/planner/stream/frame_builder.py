"""Planned waypoints -> 50 Hz streamed-motion frames, velocity-capped.

Inserted frames are re-projected onto the feet manifold and the caps
re-enforced afterwards, since projection can stretch a step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from gear_sonic.planner.configs import DEPLOY_FPS, VelocityConfig
from gear_sonic.planner.joint_orders import isaaclab_joint_names, name_permutation

_BASE_POS = slice(0, 3)
_BASE_ROT = slice(3, 6)
_JOINTS = slice(6, None)

#: subdivision passes after projection before giving up; each pass at
#: least halves the worst step.
_MAX_SUBDIVISION_PASSES = 6


@dataclass
class StreamFrames:
    """Frames ready for the pose topic, plus diagnostics."""

    joint_pos: np.ndarray  # (N, 29) f32, IsaacLab order
    joint_vel: np.ndarray  # (N, 29) f32, rad/s
    body_quat: np.ndarray  # (N, 4) f32, wxyz
    fps: float
    peak_joint_velocity: float  # rad/s, max-norm over joints, achieved
    peak_base_velocity: float  # m/s, achieved
    plan_frames: Optional[np.ndarray] = None
    """(N, 35) densified frames in planning order, for per-frame constraint
    checks without undoing the IsaacLab reorder."""

    @property
    def num_frames(self) -> int:
        return self.joint_pos.shape[0]

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps


def _rotations(waypoints: np.ndarray) -> Rotation:
    return Rotation.from_rotvec(waypoints[:, _BASE_ROT])


def _segment_steps(
    a: np.ndarray, b: np.ndarray, rot_a: Rotation, rot_b: Rotation
) -> tuple[float, float, float]:
    """(max joint delta [rad], base translation [m], base rotation [rad])."""
    joint_step = float(np.max(np.abs(b[_JOINTS] - a[_JOINTS]))) if b[_JOINTS].size else 0.0
    base_step = float(np.linalg.norm(b[_BASE_POS] - a[_BASE_POS]))
    rot_step = float((rot_a.inv() * rot_b).magnitude())
    return joint_step, base_step, rot_step


def _interpolate(
    a: np.ndarray, b: np.ndarray, rot_a: Rotation, rot_b: Rotation, t: float
) -> np.ndarray:
    """Linear in position/joints, slerp in base rotation."""
    out = a + (b - a) * t
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([rot_a, rot_b]))
    out[_BASE_ROT] = slerp(t).as_rotvec()
    return out


def densify_plan(
    waypoints: np.ndarray,
    velocity: VelocityConfig,
    fps: float = DEPLOY_FPS,
    project: Optional[Callable[[np.ndarray], bool]] = None,
) -> np.ndarray:
    """Insert frames so per-frame motion respects the velocity caps.

    ``project`` is applied to every inserted frame.  Raises if it fails, or
    if the caps still hold after ``_MAX_SUBDIVISION_PASSES``.
    """
    velocity.validate()
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] < 7:
        raise ValueError(f"Expected (M, 6 + n_joints) waypoints, got {waypoints.shape}")
    if waypoints.shape[0] == 1:
        return waypoints.copy()

    joint_cap = velocity.effective_joint_velocity() / fps
    base_cap = velocity.effective_base_velocity() / fps
    rot_cap = velocity.effective_base_angular_velocity() / fps

    def projected(x: np.ndarray) -> np.ndarray:
        if project is None:
            return x
        y = x.copy()
        if not project(y):
            raise RuntimeError(
                "Projection of an interpolated frame onto the feet manifold failed"
            )
        return y

    frames: list[np.ndarray] = [waypoints[0].copy()]
    rotations = _rotations(waypoints)
    for i in range(waypoints.shape[0] - 1):
        a, b = waypoints[i], waypoints[i + 1]
        rot_a, rot_b = rotations[i], rotations[i + 1]
        joint_step, base_step, rot_step = _segment_steps(a, b, rot_a, rot_b)
        n = max(
            1,
            int(np.ceil(joint_step / joint_cap)),
            int(np.ceil(base_step / base_cap)),
            int(np.ceil(rot_step / rot_cap)),
        )
        for k in range(1, n + 1):
            frames.append(projected(_interpolate(a, b, rot_a, rot_b, k / n)))

    # Projection can stretch steps past the caps: subdivide offenders.
    for _ in range(_MAX_SUBDIVISION_PASSES):
        rebuilt: list[np.ndarray] = [frames[0]]
        violations = 0
        rots = _rotations(np.asarray(frames))
        for i in range(len(frames) - 1):
            a, b = frames[i], frames[i + 1]
            rot_a, rot_b = rots[i], rots[i + 1]
            joint_step, base_step, rot_step = _segment_steps(a, b, rot_a, rot_b)
            if joint_step > joint_cap or base_step > base_cap or rot_step > rot_cap:
                violations += 1
                rebuilt.append(projected(_interpolate(a, b, rot_a, rot_b, 0.5)))
            rebuilt.append(b)
        frames = rebuilt
        if violations == 0:
            break
    else:
        raise RuntimeError(
            f"Velocity caps still violated after {_MAX_SUBDIVISION_PASSES} "
            f"subdivision passes (projection keeps stretching steps); "
            f"lower projection_delta or raise the caps"
        )
    return np.asarray(frames)


def build_stream_frames(
    waypoints: np.ndarray,
    plan_joint_names: list[str],
    velocity: VelocityConfig,
    fps: float = DEPLOY_FPS,
    project: Optional[Callable[[np.ndarray], bool]] = None,
) -> StreamFrames:
    """Densify -> reorder to IsaacLab -> velocities -> quats. ``waypoints``
    are (M, 35) ``[base_pos(3), base_rotvec(3), joints(29)]``."""
    waypoints = np.asarray(waypoints, dtype=float)
    n_joints = len(plan_joint_names)
    if waypoints.ndim != 2 or waypoints.shape[1] != 6 + n_joints:
        raise ValueError(
            f"waypoints shape {waypoints.shape} does not match 6 + "
            f"{n_joints} planning joints"
        )

    frames = densify_plan(waypoints, velocity, fps=fps, project=project)

    to_il = name_permutation(list(plan_joint_names), isaaclab_joint_names())
    joint_pos = frames[:, _JOINTS][:, to_il]

    # Central difference at the stream rate; one-sided at the ends.
    joint_vel = np.gradient(joint_pos, 1.0 / fps, axis=0)

    quats_xyzw = Rotation.from_rotvec(frames[:, _BASE_ROT]).as_quat()
    body_quat = quats_xyzw[:, [3, 0, 1, 2]]  # scipy xyzw -> deploy wxyz

    joint_deltas = np.abs(np.diff(joint_pos, axis=0))
    base_deltas = np.linalg.norm(np.diff(frames[:, _BASE_POS], axis=0), axis=1)
    peak_joint_velocity = float(joint_deltas.max() * fps) if joint_deltas.size else 0.0
    peak_base_velocity = float(base_deltas.max() * fps) if base_deltas.size else 0.0

    return StreamFrames(
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_quat=body_quat.astype(np.float32),
        fps=fps,
        peak_joint_velocity=peak_joint_velocity,
        peak_base_velocity=peak_base_velocity,
        plan_frames=frames,
    )

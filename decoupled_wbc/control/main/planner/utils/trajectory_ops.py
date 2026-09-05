"""Joint-path helpers on (N, D) arrays: arc length, resampling, ramp."""

from __future__ import annotations

import numpy as np


def interp_rows(s_src: np.ndarray, rows: np.ndarray, s_dst: np.ndarray) -> np.ndarray:
    """Linearly interpolate every column of ``rows`` (sampled at ``s_src``) onto ``s_dst``."""
    rows = np.asarray(rows, dtype=float)
    return np.stack(
        [np.interp(s_dst, s_src, rows[:, k]) for k in range(rows.shape[1])], axis=1
    )


def arc_length(path: np.ndarray, per_joint_max: bool = False) -> np.ndarray:
    """Cumulative arc length (N,); ``per_joint_max`` measures each step by its fastest joint."""
    steps = np.diff(np.asarray(path, dtype=float), axis=0)
    seg = np.abs(steps).max(axis=1) if per_joint_max else np.linalg.norm(steps, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def normalised_arc_length(path: np.ndarray) -> np.ndarray:
    """Euclidean arc length rescaled to ``0..1`` (progress along the path)."""
    s = arc_length(path)
    return s / max(s[-1], 1e-12)


def resample_waypoints(waypoints: np.ndarray, max_joint_step: float) -> np.ndarray:
    """Resample uniformly (linear, no smoothing) so the fastest joint moves
    ``max_joint_step`` per row; endpoints are kept."""
    waypoints = np.asarray(waypoints)
    if waypoints.ndim != 2 or waypoints.shape[0] == 0:
        raise ValueError(f"Expected waypoints shape (N, DoF), got {waypoints.shape}")
    if max_joint_step <= 0:
        raise ValueError("max_joint_step must be > 0")
    if waypoints.shape[0] == 1:
        return waypoints.astype(np.float32, copy=True)

    s = arc_length(waypoints, per_joint_max=True)
    if s[-1] < 1e-12:
        return waypoints[[0, -1]].astype(np.float32, copy=True)
    n_steps = max(1, int(np.ceil(s[-1] / max_joint_step)))
    resampled = interp_rows(s, waypoints, np.linspace(0.0, s[-1], n_steps + 1))
    resampled[0], resampled[-1] = waypoints[0], waypoints[-1]
    return resampled.astype(np.float32)


def smoothstep_ramp(start: np.ndarray, end: np.ndarray, steps: int) -> np.ndarray:
    """(steps, D) smoothstep ramp from ``start`` (exclusive) to ``end``."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    a = np.arange(1, steps + 1) / steps
    a = a * a * (3.0 - 2.0 * a)
    return start + a[:, None] * (end - start)

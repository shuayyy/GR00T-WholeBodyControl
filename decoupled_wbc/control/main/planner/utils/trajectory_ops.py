"""Joint-path helpers on (N, D) arrays: arc length, resampling, smoothing, ramp."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import splev, splrep


def interp_rows(s_src: np.ndarray, rows: np.ndarray, s_dst: np.ndarray) -> np.ndarray:
    """Linearly interpolate every column of ``rows`` (sampled at ``s_src``) onto ``s_dst``."""
    rows = np.asarray(rows, dtype=float)
    return np.stack([np.interp(s_dst, s_src, rows[:, k]) for k in range(rows.shape[1])], axis=1)


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


def smooth_spline(
    waypoints: np.ndarray,
    s_factor: float = 0.005,
    n_out: int = 400,
    spacing: float = 0.025,
) -> np.ndarray:
    """Replace a polyline with a cubic smoothing spline, endpoints pinned.

    The path is first resampled to uniform ``spacing`` so the spline parameter is
    arc length, then one cubic ``splrep`` is fitted per joint with tolerance
    ``s_factor * N`` (scipy's ``s``), and sampled at ``n_out`` points.  Unlike
    OMPL's ``smoothBSpline``, which nudges polyline vertices, this returns a curve
    with continuous second derivative: a raw PhaseRRT* plan went from a 96 deg
    worst corner to 12 deg at ``s_factor`` 0.005, beyond which the fit no longer
    changes.  It does no collision checking; the caller must validate the result.
    """
    pts = resample_waypoints(np.asarray(waypoints, dtype=float), spacing)
    if len(pts) < 4:  # a cubic needs four points; nothing to smooth
        return pts
    u = normalised_arc_length(pts)
    weights = np.ones(len(pts))
    weights[0] = weights[-1] = 1e6  # start and goal must not move
    u_out = np.linspace(0.0, 1.0, n_out)
    return np.stack(
        [
            splev(u_out, splrep(u, pts[:, j], w=weights, s=s_factor * len(pts), k=3))
            for j in range(pts.shape[1])
        ],
        axis=1,
    )


def repair_to_polyline(smoothed: np.ndarray, raw: np.ndarray, is_valid, spacing: float = 0.025):
    """``smoothed`` with every invalid sample pulled back toward the raw polyline.

    The raw polyline is interpolated at the same arc-length parameters as the
    smoothed curve, so each sample has a valid counterpart to retreat to; a sample
    is blended toward it in quarter steps until ``is_valid`` accepts it.  Returns
    None if some sample is invalid even at the raw point.
    """
    pts = resample_waypoints(np.asarray(raw, dtype=float), spacing)
    raw_dense = interp_rows(normalised_arc_length(pts), pts, np.linspace(0.0, 1.0, len(smoothed)))
    out = np.array(smoothed, dtype=float, copy=True)
    for i, q in enumerate(out):
        if is_valid(q):
            continue
        for t in (0.25, 0.5, 0.75, 1.0):
            cand = (1.0 - t) * q + t * raw_dense[i]
            if is_valid(cand):
                out[i] = cand
                break
        else:
            return None
    return out


# Smoothing tolerances tried in order, strongest first.  0.005 is where the fit
# stops changing on the humanoid demos; the smaller ones exist for scenes where
# the strongest fit would cut into an obstacle's clearance margin.
SMOOTH_S_FACTORS = (0.005, 0.002, 0.001, 0.0005, 0.0002)


def smooth_valid(waypoints: np.ndarray, is_valid, s_factors=SMOOTH_S_FACTORS):
    """Smoothest ``smooth_spline`` fit whose every sample passes ``is_valid``.

    Strongest tolerance first; each fit is repaired toward the raw polyline before
    being judged.  Returns ``(path, s_factor)``, or ``(None, None)`` if no factor
    yields a valid curve.  The raw plan may graze an obstacle at the validity
    tolerance and the spline moves the path by a few mm, which is why the repair
    step exists rather than a looser tolerance.
    """
    for s_factor in s_factors:
        repaired = repair_to_polyline(
            smooth_spline(waypoints, s_factor=s_factor), waypoints, is_valid
        )
        if repaired is not None:
            return repaired, s_factor
    return None, None

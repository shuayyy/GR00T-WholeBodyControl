"""Metrics of one recorded run: RL tracking error and deviation from the demo
(arc-length matched, so timing does not matter)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import numpy as np

from decoupled_wbc.control.main.planner.simulation.robot import JOINT_NAMES_UP
from decoupled_wbc.control.main.planner.utils.controller_interface import (
    ControllerUpperBodyInterface,
    full_q_to_planning_qpos,
    upper_body_pose_to_planning_qpos,
)
from decoupled_wbc.control.main.planner.utils.ee_trajectory import ee_in_root_frame
from decoupled_wbc.control.main.planner.utils.trajectory_ops import (
    interp_rows,
    normalised_arc_length,
)

GOAL_CHANGE_EPS = 1e-6  # a goal differing from the previous one by more than this "moved"


@dataclass
class TrackingMetrics:
    """|robot joint − commanded joint| while the plan was streamed (rad)."""

    stream_duration: float
    mean: float
    """Mean over the 17 planning joints and all samples in the streamed window."""
    rmse: float
    final: float
    """Mean joint error at the last recorded sample (after the hold)."""
    final_worst_joint: str
    final_worst: float


@dataclass
class EEDeviation:
    pos_mm: float
    rot_rad: float


@dataclass
class PathDeviation:
    """A joint path vs the demo, arc-length matched, mean over the path."""

    joint_mean: float
    joint_rmse: float
    left: EEDeviation
    right: EEDeviation


@dataclass
class RunMetrics:
    tracking: TrackingMetrics
    plan: Optional[PathDeviation]
    """Planner output vs demo; None when the run had no reference."""
    executed: Optional[PathDeviation]
    """Recorded robot motion vs demo; None when the run had no reference."""

    def to_flat(self) -> dict:
        """One level of ``name -> value`` (stored in the npz as ``metric_<name>``)."""
        flat = {"stream_duration": self.tracking.stream_duration}
        for f in fields(TrackingMetrics):
            if f.name != "stream_duration":
                flat[f"tracking_{f.name}"] = getattr(self.tracking, f.name)
        for label, dev in (("plan", self.plan), ("exec", self.executed)):
            if dev is None:
                continue
            flat[f"joint_dev_{label}_mean"] = dev.joint_mean
            flat[f"joint_dev_{label}_rmse"] = dev.joint_rmse
            for side in ("left", "right"):
                ee = getattr(dev, side)
                flat[f"ee_dev_{label}_{side}_pos_mm"] = ee.pos_mm
                flat[f"ee_dev_{label}_{side}_rot_rad"] = ee.rot_rad
        return flat

    def lines(self) -> list[str]:
        t = self.tracking
        out = [
            f"Tracking error over {t.stream_duration:.1f}s of streaming "
            f"(17 upper-body joints, rad): mean {t.mean:.3f}, RMSE {t.rmse:.3f}, "
            f"final-pose {t.final:.3f} (worst joint {t.final_worst_joint} {t.final_worst:.3f})"
        ]
        for label, dev in (("plan", self.plan), ("exec", self.executed)):
            if dev is None:
                continue
            out.append(
                f"Joint deviation from reference ({label}, rad): "
                f"mean {dev.joint_mean:.3f}, RMSE {dev.joint_rmse:.3f}"
            )
            out.append(
                f"EE deviation from reference ({label}): "
                f"left {dev.left.pos_mm:.0f} mm / {dev.left.rot_rad:.3f} rad, "
                f"right {dev.right.pos_mm:.0f} mm / {dev.right.rot_rad:.3f} rad"
            )
        return out


def streamed_window(rec: dict) -> tuple[float, float]:
    """``(t0, t1)``: from the first streamed goal to the last goal that still moved."""
    goals, goal_t = np.asarray(rec["goal_pose"]), np.asarray(rec["goal_t"])
    moved = np.where(np.abs(np.diff(goals, axis=0)).max(axis=1) > GOAL_CHANGE_EPS)[0]
    if moved.size == 0:
        raise ValueError("no goal ever changed; nothing was streamed")
    return float(rec["stream_start_t"]), float(goal_t[moved[-1] + 1])


def tracking_metrics(rec: dict, interface: ControllerUpperBodyInterface) -> TrackingMetrics:
    robot_q = np.stack([full_q_to_planning_qpos(q, interface) for q in rec["q"]])
    goal_q = np.stack([upper_body_pose_to_planning_qpos(g, interface) for g in rec["goal_pose"]])
    t, goal_t = np.asarray(rec["t"]), np.asarray(rec["goal_t"])
    t0, t1 = streamed_window(rec)
    in_window = (t >= t0) & (t <= t1)
    if in_window.sum() < 2:
        raise ValueError("no robot samples inside the streamed window")
    # each robot sample is compared with the goal that was current at that time
    current_goal = np.searchsorted(goal_t, t[in_window], side="right") - 1
    err = np.abs(robot_q[in_window] - goal_q[current_goal])
    final_err = np.abs(robot_q[-1] - goal_q[-1])
    return TrackingMetrics(
        stream_duration=t1 - t0,
        mean=float(err.mean()),
        rmse=float(np.sqrt((err**2).mean())),
        final=float(final_err.mean()),
        final_worst_joint=JOINT_NAMES_UP[int(final_err.argmax())],
        final_worst=float(final_err.max()),
    )


def _quat_angle(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """Geodesic angle between unit quaternions, row-wise (rad)."""
    dot = np.abs((q_a * q_b).sum(axis=1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def path_deviation(path: np.ndarray, reference: np.ndarray) -> PathDeviation:
    """Joint and wrist deviation of ``path`` from ``reference`` (both ``(N, 17)``)."""
    progress = normalised_arc_length(path)
    ref_progress = normalised_arc_length(reference)
    ref_at = interp_rows(ref_progress, reference, progress)
    joint_err = np.abs(path - ref_at)

    ee = ee_in_root_frame(path, JOINT_NAMES_UP)
    ee_ref = ee_in_root_frame(reference, JOINT_NAMES_UP)
    nearest_ref = np.searchsorted(ref_progress, progress).clip(0, len(ref_progress) - 1)
    sides = {}
    for body, poses in ee.items():
        ref_pos = interp_rows(ref_progress, ee_ref[body]["pos"], progress)
        sides["left" if body.startswith("left") else "right"] = EEDeviation(
            pos_mm=float(np.linalg.norm(poses["pos"] - ref_pos, axis=1).mean() * 1000.0),
            rot_rad=float(_quat_angle(poses["quat"], ee_ref[body]["quat"][nearest_ref]).mean()),
        )
    return PathDeviation(
        joint_mean=float(joint_err.mean()),
        joint_rmse=float(np.sqrt((joint_err**2).mean())),
        left=sides["left"],
        right=sides["right"],
    )


def compute_run_metrics(rec: dict, interface: ControllerUpperBodyInterface) -> RunMetrics:
    tracking = tracking_metrics(rec, interface)
    if "reference" not in rec:
        return RunMetrics(tracking=tracking, plan=None, executed=None)
    reference = np.asarray(rec["reference"], dtype=float)
    t = np.asarray(rec["t"])
    t0, t1 = streamed_window(rec)
    robot_q = np.stack(
        [full_q_to_planning_qpos(q, interface) for q, ti in zip(rec["q"], t) if t0 <= ti <= t1]
    )
    return RunMetrics(
        tracking=tracking,
        plan=path_deviation(np.asarray(rec["plan_qpos"], dtype=float), reference),
        executed=path_deviation(robot_q, reference),
    )

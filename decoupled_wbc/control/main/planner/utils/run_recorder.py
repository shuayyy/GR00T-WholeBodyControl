"""Record one executed plan (robot state + every goal sent), save it and compute
its metrics.  ``RECORDING_FORMAT`` documents the npz."""

from __future__ import annotations

from pathlib import Path
import threading
import time
import traceback
from typing import Optional

import numpy as np

from decoupled_wbc.control.main.planner.simulation.robot import JOINT_NAMES_UP
from decoupled_wbc.control.main.planner.utils.controller_interface import (
    ControllerUpperBodyInterface,
)
from decoupled_wbc.control.main.planner.utils.run_metrics import compute_run_metrics

RECORDING_FORMAT = """
recordings/<YYYYmmdd_HHMMSS>_<planner>.npz

t                 (S,)      time.monotonic() of each robot state sample
q                 (S, F)    controller full joint vector at each sample
base_pose         (S, 7)    floating base xyz + wxyz at each sample
goal_t            (G,)      time.monotonic() of each published goal
goal_pose         (G, U)    published target_upper_body_pose (controller order)
goal_target_time  (G,)      target_time sent with each goal
plan_qpos         (N, 17)   the streamed plan, JOINT_NAMES_UP order
reference         (R, 17)   the demo the planner was given (absent if none)
start_t           ()        recording start (ramp begins)
stream_start_t    ()        first plan waypoint published
planner           ()        planner name
full_joint_names / controller_joint_names / planning_joint_names
metric_<name>     ()        RunMetrics.to_flat() entries
"""


class RunRecorder:
    def __init__(
        self,
        record_dir: Path,
        interface: ControllerUpperBodyInterface,
        full_joint_names: list[str],
        reference: Optional[np.ndarray],
        logger,
        settle_s: float = 2.0,
    ):
        self.record_dir = Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.interface = interface
        self.full_joint_names = list(full_joint_names)
        self.reference = None if reference is None else np.asarray(reference, dtype=np.float32)
        self.logger = logger
        self.settle_s = settle_s
        self._lock = threading.Lock()
        self._run: Optional[dict] = None
        self._hold_since: Optional[float] = None

    @property
    def active(self) -> bool:
        return self._run is not None

    def start(self, plan_qpos: np.ndarray, planner_name: str) -> None:
        if self.active:
            self.finish()
        with self._lock:
            self._hold_since = None
            self._run = {
                "planner": planner_name,
                "plan_qpos": np.asarray(plan_qpos, dtype=np.float32),
                "start_t": time.monotonic(),
                "stream_start_t": None,
                "t": [],
                "q": [],
                "base_pose": [],
                "goal_t": [],
                "goal_pose": [],
                "goal_target_time": [],
            }

    def on_state(self, q: np.ndarray, base_pose: np.ndarray) -> None:
        with self._lock:
            if self._run is None:
                return
            self._run["t"].append(time.monotonic())
            self._run["q"].append(np.array(q, dtype=np.float32).reshape(-1))
            self._run["base_pose"].append(np.array(base_pose, dtype=np.float32).reshape(-1))

    def on_goal(self, pose: np.ndarray, target_time: float) -> None:
        with self._lock:
            if self._run is None:
                return
            self._run["goal_t"].append(time.monotonic())
            self._run["goal_pose"].append(np.array(pose, dtype=np.float32).reshape(-1))
            self._run["goal_target_time"].append(float(target_time))

    def mark_stream_start(self) -> None:
        with self._lock:
            if self._run is not None:
                self._run["stream_start_t"] = time.monotonic()

    def mark_hold(self) -> None:
        with self._lock:
            if self._run is not None and self._hold_since is None:
                self._hold_since = time.monotonic()

    def finish_if_settled(self) -> Optional[Path]:
        if self._run is None or self._hold_since is None:
            return None
        if time.monotonic() - self._hold_since < self.settle_s:
            return None
        return self.finish()

    def finish(self) -> Optional[Path]:
        with self._lock:
            run, self._run, self._hold_since = self._run, None, None
        if run is None or not run["t"]:
            self.logger.warn("Recording had no robot state; nothing saved")
            return None

        rec = {
            "t": np.asarray(run["t"]),
            "q": np.asarray(run["q"]),
            "base_pose": np.asarray(run["base_pose"]),
            "goal_t": np.asarray(run["goal_t"]),
            "goal_pose": np.asarray(run["goal_pose"]),
            "goal_target_time": np.asarray(run["goal_target_time"]),
            "plan_qpos": run["plan_qpos"],
            "start_t": np.float64(run["start_t"]),
            "stream_start_t": np.float64(run["stream_start_t"] or run["start_t"]),
            "planner": np.str_(run["planner"]),
            "full_joint_names": np.array(self.full_joint_names),
            "controller_joint_names": np.array(self.interface.joint_names),
            "planning_joint_names": np.array(JOINT_NAMES_UP),
        }
        if self.reference is not None:
            rec["reference"] = self.reference

        metrics = None
        try:
            metrics = compute_run_metrics(rec, self.interface)
        except Exception:  # a metrics bug must not lose the recording itself
            self.logger.warn("Metrics failed:\n" + traceback.format_exc())
        if metrics is not None:
            for name, value in metrics.to_flat().items():
                rec[f"metric_{name}"] = value

        path = self.record_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{run['planner']}.npz"
        np.savez(path, **rec)
        self.logger.info(
            f"Saved recording {path} ({len(run['t'])} states, {len(run['goal_t'])} goals)"
        )
        if metrics is not None:
            for line in metrics.lines():
                self.logger.info(line)
        return path

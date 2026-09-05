from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import threading
import time
from typing import Optional

import numpy as np
import rclpy
import tyro
import mujoco
import msgpack
import msgpack_numpy as mnp
from std_msgs.msg import ByteMultiArray

from decoupled_wbc.control.main.constants import (
    CONTROL_GOAL_TOPIC,
    DEFAULT_BASE_HEIGHT,
    STATE_TOPIC_NAME,
)
from decoupled_wbc.control.main.planner.configs.configs import PlannerConfig
from decoupled_wbc.control.main.planner.utils.ros_utils import (
    ROSDictServiceServer,
)
from decoupled_wbc.control.robot_model.instantiation.g1 import (
    instantiate_g1_robot_model,
)
from decoupled_wbc.control.utils.ros_utils import (
    ROSManager,
    ROSMsgPublisher,
    ROSMsgSubscriber,
)
from decoupled_wbc.control.main.planner.utils.controller_interface import (
    build_controller_interface,
    full_q_to_planning_qpos,
    planning_qpos_to_controller_pose,
)
from decoupled_wbc.control.main.planner.utils.demo_trajectory import load_planning_trajectory
from decoupled_wbc.control.main.planner.utils.planner_factory import make_planner
from decoupled_wbc.control.main.planner.utils.run_recorder import RunRecorder
from decoupled_wbc.control.main.planner.utils.trajectory_ops import (
    resample_waypoints,
    smoothstep_ramp,
)
from decoupled_wbc.control.main.planner.simulation.robot import (
    G1Up,
    JOINT_NAMES_UP,
    JOINT_NAMES_LEFT,
)


PLANNER_DIR = Path(__file__).resolve().parent
PLANNER_NODE_NAME = "PlannerServer"
PLANNER_PLAN_SERVICE = "PlannerServer/plan"


@dataclass
class ActiveTrajectory:
    qpos: np.ndarray
    frame_idx: int
    is_first_publish: bool
    hold_final_pose_printed: bool


def parse_plan_request(
    request: dict, default_goal_type: str, default_execute_immediately: bool
) -> tuple[np.ndarray, Optional[np.ndarray], str, bool]:
    """
    Parse a plan-service request into (goal, start, goal_type, execute_immediately).

    Accepted keys:
      - goal_qpos: 17-DoF planning joints in JOINT_NAMES_UP order
      - start_qpos: optional start overrides
      - goal_type: optional "upper_body" | "bimanual" | "left" | "right"
      - execute_immediately: optional bool; if true, stream path to the control loop
    """
    goal_type = str(request.get("goal_type", default_goal_type))
    execute_immediately = bool(
        request.get("execute_immediately", default_execute_immediately)
    )

    if "goal_qpos" in request:
        goal = np.asarray(request["goal_qpos"], dtype=np.float32).reshape(-1)
        if goal.shape[0] != len(JOINT_NAMES_UP):
            raise ValueError(
                f"goal_qpos length {goal.shape[0]} != {len(JOINT_NAMES_UP)}"
            )
    else:
        raise KeyError("Plan request must include goal_qpos")

    start = None
    if "start_qpos" in request:
        if request["start_qpos"] is not None:
            start = np.asarray(
                request["start_qpos"], dtype=np.float32
            ).reshape(-1)
            if start.shape[0] != len(JOINT_NAMES_UP):
                raise ValueError(
                    f"start_qpos length {start.shape[0]} != {len(JOINT_NAMES_UP)}"
                )
    return goal, start, goal_type, execute_immediately


def interruptible_sleep(duration: float, keep_running) -> None:
    end_time = time.monotonic() + duration
    while keep_running():
        remaining = end_time - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.1, remaining))


class PlannerServer:
    def __init__(
        self,
        config: PlannerConfig,
        logger,
        state_subscriber: ROSMsgSubscriber,
    ):
        if config.planner_frequency <= 0:
            raise ValueError("planner_frequency must be > 0")
        if config.planning_timeout <= 0:
            raise ValueError("planning_timeout must be > 0")

        self.config = config
        self.logger = logger
        self.state_subscriber = state_subscriber
        self.controller_interface, self.full_joint_names = build_controller_interface(
            config.enable_waist, config.high_elbow_pose
        )
        self.ref_traj = self.load_reference_trajectory(config)

        xml_path = self.resolve_planning_xml(config)
        model = self.load_planning_model(xml_path)
        fixed_qpos, base_pose = self.standing_pose(model, config)
        self.robot = G1Up(
            model=model,
            visualize=config.visualize_planning,
            fixed_qpos=fixed_qpos,
            base_pose=base_pose,
        )
        self.planner = make_planner(
            config.ompl_planner,
            self.robot,
            reference=self.ref_traj,
            validity_resolution=config.validity_resolution,
            log=True,
        )

        self._lock = threading.Lock()
        self._active: Optional[ActiveTrajectory] = None
        self._cancel_execution = False
        self._planning = False

        self.recorder: Optional[RunRecorder] = None
        if config.record:
            self.recorder = RunRecorder(
                self.recording_dir(config),
                self.controller_interface,
                self.full_joint_names,
                self.ref_traj,
                self.logger,
            )
            # Own subscription: the shared ROSMsgSubscriber keeps only the latest message.
            self._record_state_sub = state_subscriber.node.create_subscription(
                ByteMultiArray, STATE_TOPIC_NAME, self._on_state_msg, 50
            )
            self.logger.info(f"Recording executed plans to {self.recorder.record_dir}")

        self.logger.info(f"Planning XML: {xml_path}")
        self.logger.info(f"OMPL planner: {config.ompl_planner}")
        self.logger.info(
            "Reference objective: "
            + ("enabled" if self.ref_traj is not None else "disabled")
        )
        self.logger.info(f"Planning joints: {JOINT_NAMES_UP}")
        self.logger.info(
            f"Controller upper-body joints: {self.controller_interface.joint_names}"
        )
        self.logger.info(f"Plan service: {config.plan_service}")
        self.logger.info(f"Trajectory topic: {config.trajectory_topic}")
        self.logger.info(
            f"Default execute_immediately: {config.execute_immediately}"
        )

    def close(self) -> None:
        self.robot.close()

    def read_start_from_state(self) -> np.ndarray:
        state = self.state_subscriber.get_msg()
        if state is None or "q" not in state:
            self.logger.warn(
                "No robot state available; using planning-model qpos as start"
            )
            return self.robot.get_joint_qpos().astype(np.float32)
        return full_q_to_planning_qpos(
            np.asarray(state["q"], dtype=np.float32).reshape(-1),
            self.controller_interface,
        )

    def plan_to_goal(
        self,
        goal: np.ndarray,
        start: Optional[np.ndarray],
        goal_type: str,
    ) -> tuple[np.ndarray, float]:
        if start is None:
            start = self.read_start_from_state()
            self.robot.set_joint_qpos(start)

        start = np.asarray(start, dtype=np.float64).reshape(-1)
        goal = np.asarray(goal, dtype=np.float64).reshape(-1)
        self.logger.info(
            f"Planning Requested \nstart:\n {start} \ngoal:\n {goal}"
        )

        if start.shape[0] != len(JOINT_NAMES_UP) or goal.shape[0] != len(
            JOINT_NAMES_UP
        ):
            raise ValueError(
                "Start/goal must match the 17-DoF planning joint set"
            )

        self.logger.info(
            f"Planning with {self.planner.name} "
            f"(timeout={self.config.planning_timeout}s, goal_type={goal_type})"
        )
        t0 = time.monotonic()
        solution = self.planner.plan(
            start,
            goal,
            goal_type,
            timeout=self.config.planning_timeout,
            smooth_path=self.config.smooth_path,
            shortcut_path=self.config.shortcut_path,
        )
        elapsed = time.monotonic() - t0

        if solution is None or len(solution) < 2:
            raise RuntimeError(f"OMPL failed to find a path in {elapsed:.3f}s")

        solution = np.asarray(solution, dtype=np.float64)
        if solution.ndim != 2 or solution.shape[1] != len(JOINT_NAMES_UP):
            raise RuntimeError(
                f"OMPL returned an invalid path shape: {solution.shape}"
            )
        start_error = float(np.max(np.abs(solution[0] - start)))
        if goal_type == "right":
            goal_indices = [
                idx
                for idx, name in enumerate(JOINT_NAMES_UP)
                if name not in JOINT_NAMES_LEFT
            ]
        else:
            goal_indices = list(range(len(JOINT_NAMES_UP)))
        goal_error = float(
            np.max(
                np.abs(
                    solution[-1, goal_indices] - goal[goal_indices]
                )
            )
        )
        if (
            start_error > self.config.endpoint_tolerance
            or goal_error > self.config.endpoint_tolerance
        ):
            raise RuntimeError(
                "OMPL returned invalid path endpoints: "
                f"start_error={start_error:.6f}, "
                f"goal_error={goal_error:.6f}, "
                f"tolerance={self.config.endpoint_tolerance:.6f}"
            )

        waypoints = resample_waypoints(
            solution, self.config.max_joint_step
        )
        self.logger.info(
            f"Plan ready in {elapsed:.3f}s: "
            f"{len(solution)} raw -> {len(waypoints)} resampled waypoints"
        )
        return waypoints, elapsed

    def set_active_trajectory(self, waypoints: np.ndarray) -> None:
        with self._lock:
            self._cancel_execution = True
            self._active = ActiveTrajectory(
                qpos=waypoints.astype(np.float32, copy=False),
                frame_idx=0,
                is_first_publish=True,
                hold_final_pose_printed=False,
            )
            self._cancel_execution = False

    def handle_plan_request(self, request: dict) -> dict:
        """ROS2 service handler: plan once and optionally begin execution."""
        with self._lock:
            if self._planning:
                raise RuntimeError("Planner is busy with another request")
            self._planning = True

        try:
            goal, start, goal_type, execute_immediately = parse_plan_request(
                request, self.config.goal_type, self.config.execute_immediately
            )
            waypoints, planning_time = self.plan_to_goal(
                goal, start, goal_type
            )

            if execute_immediately:
                self.set_active_trajectory(waypoints)
                self.logger.info(
                    "execute_immediately=True; streaming trajectory to control loop"
                )
            else:
                self.logger.info(
                    "execute_immediately=False; returning path only"
                )

            return {
                "qpos": waypoints.astype(np.float32, copy=False),
                "joint_names": list(JOINT_NAMES_UP),
                "num_waypoints": int(waypoints.shape[0]),
                "planning_time": float(planning_time),
                "executed": bool(execute_immediately),
                "goal_type": goal_type,
            }
        finally:
            with self._lock:
                self._planning = False

    def publish_step(
        self, control_publisher: ROSMsgPublisher, keep_running
    ) -> str:
        """
        Publish one trajectory frame.

        Returns:
            "idle" - nothing to publish
            "initial" - published first waypoint (already waited transition time)
            "streaming" - published a normal waypoint
        """
        if self.recorder is not None:
            self.recorder.finish_if_settled()

        with self._lock:
            active = self._active
            if active is None:
                return "idle"
            frame_idx = active.frame_idx
            is_first = active.is_first_publish
            qpos = active.qpos

        publish_period = 1.0 / self.config.planner_frequency
        if is_first:
            if self.recorder is not None:
                self.recorder.start(qpos, self.planner.name)
            self.publish_start_ramp(control_publisher, qpos[0], keep_running)
            if self.recorder is not None:
                self.recorder.mark_stream_start()
        else:
            self.publish_target(
                control_publisher, qpos[frame_idx], time.monotonic() + publish_period
            )

        with self._lock:
            if self._active is None or self._cancel_execution:
                return "idle"
            active = self._active
            if is_first:
                active.is_first_publish = False
                if active.qpos.shape[0] > 1:
                    active.frame_idx = 1
            elif active.frame_idx < active.qpos.shape[0] - 1:
                active.frame_idx += 1
            else:
                if self.config.hold_final_pose:
                    if not active.hold_final_pose_printed:
                        self.logger.info(
                            "Reached final planned waypoint; holding final pose."
                        )
                        active.hold_final_pose_printed = True
                    active.frame_idx = active.qpos.shape[0] - 1
                else:
                    self._active = None
                if self.recorder is not None:
                    self.recorder.mark_hold()

        return "initial" if is_first else "streaming"

    def publish_target(
        self, control_publisher: ROSMsgPublisher, planning_q: np.ndarray, target_time: float
    ) -> None:
        target_upper_body_pose = planning_qpos_to_controller_pose(
            planning_q, self.controller_interface
        )
        if self.recorder is not None:
            self.recorder.on_goal(target_upper_body_pose, target_time)
        control_publisher.publish(
            {
                "target_upper_body_pose": target_upper_body_pose,
                # needed so the balance policy executes the commanded waist orientation
                "navigate_cmd": [0.0, 0.0, 0.0],
                "timestamp": time.monotonic(),
                "target_time": target_time,
            }
        )

    def publish_start_ramp(
        self, control_publisher: ROSMsgPublisher, start_qpos: np.ndarray, keep_running
    ) -> None:
        """Smoothstep ramp from the measured pose to the plan start over
        initial_transition_time (single target if no robot state)."""
        period = 1.0 / self.config.planner_frequency
        duration = self.config.initial_transition_time
        state = self.state_subscriber.get_msg()
        if state is None or "q" not in state:
            self.logger.warn(
                "No robot state; publishing the plan start as a single target"
            )
            self.publish_target(control_publisher, start_qpos, time.monotonic() + duration)
            interruptible_sleep(duration, keep_running)
            return
        current = full_q_to_planning_qpos(
            np.asarray(state["q"], dtype=np.float32).reshape(-1),
            self.controller_interface,
        )
        start_qpos = np.asarray(start_qpos, dtype=np.float32)
        steps = max(1, int(round(duration * self.config.planner_frequency)))
        self.logger.info(
            f"Ramping to plan start over {duration:.1f}s ({steps} steps, "
            f"largest joint gap {np.abs(start_qpos - current).max():.3f} rad)"
        )
        t_next = time.monotonic()
        for pose in smoothstep_ramp(current, start_qpos, steps):
            if not keep_running() or self._cancel_execution:
                return
            t_next += period
            self.publish_target(control_publisher, pose, t_next)
            remaining = t_next - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

    def _on_state_msg(self, msg: ByteMultiArray) -> None:
        """Robot state topic -> recorder (runs on the ROS executor thread)."""
        state = msgpack.unpackb(bytes([b for a in msg.data for b in a]), object_hook=mnp.decode)
        if "q" in state:
            self.recorder.on_state(state["q"], state.get("floating_base_pose", np.zeros(7)))

    # Helper functions
    def load_reference_trajectory(
        self, config: PlannerConfig
    ) -> Optional[np.ndarray]:
        if not config.use_reference:
            return None
        reference = load_planning_trajectory(config.reference_trajectory_path, PLANNER_DIR)
        self.logger.info(
            f"Reference trajectory: {config.reference_trajectory_path} "
            f"({reference.shape[0]} frames)"
        )
        return reference

    @staticmethod
    def recording_dir(config: PlannerConfig) -> Path:
        """``recordings/`` next to the reference trajectory, or under the planner package."""
        base = PLANNER_DIR
        if config.use_reference and config.reference_trajectory_path:
            ref = Path(config.reference_trajectory_path)
            base = (ref if ref.is_absolute() else PLANNER_DIR / ref).resolve().parent
        return base / "recordings"

    def standing_pose(self, model, config: PlannerConfig):
        """WBC default pose for the non-planned joints and the base of a full-body
        planning model; (None, None) for a model without them (g1_up.xml)."""
        robot_model = instantiate_g1_robot_model(
            waist_location="lower_and_upper_body" if config.enable_waist else "lower_body",
            high_elbow_pose=config.high_elbow_pose,
        )
        model_joints = {model.joint(i).name for i in range(model.njnt)}
        fixed_qpos = {
            name: float(value)
            for name, value in zip(robot_model.joint_names, robot_model.default_body_pose)
            if name not in JOINT_NAMES_UP and name in model_joints
        }
        has_free = any(model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE for i in range(model.njnt))
        base_pose = (np.array([0.0, 0.0, DEFAULT_BASE_HEIGHT]), np.array([1.0, 0.0, 0.0, 0.0])) if has_free else None
        return (fixed_qpos or None), base_pose

    def resolve_planning_xml(self, config: PlannerConfig) -> Path:
        xml_path = Path(config.planning_xml)
        if not xml_path.is_absolute():
            xml_path = PLANNER_DIR / xml_path
        xml_path = xml_path.resolve()
        if not xml_path.exists():
            raise FileNotFoundError(f"Planning XML not found: {xml_path}")
        return xml_path

    def load_planning_model(self, xml_path: Path):
        """Load MuJoCo model with includes resolved from the planner package root."""
        prev_cwd = os.getcwd()
        os.chdir(PLANNER_DIR)
        try:
            return mujoco.MjModel.from_xml_path(str(xml_path))
        finally:
            os.chdir(prev_cwd)


def main(config: PlannerConfig):
    if not config.plan_service:
        config.plan_service = PLANNER_PLAN_SERVICE
    if not config.trajectory_topic:
        config.trajectory_topic = CONTROL_GOAL_TOPIC

    ros_manager = ROSManager(node_name=PLANNER_NODE_NAME)
    node = ros_manager.node
    logger = node.get_logger()
    server = None

    try:
        state_subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)
        control_publisher = ROSMsgPublisher(config.trajectory_topic)
        server = PlannerServer(config, logger, state_subscriber)
        ROSDictServiceServer(config.plan_service, server.handle_plan_request)
        rate = node.create_rate(config.planner_frequency)

        logger.info(
            f"Planner service ready at '{config.plan_service}'. "
            f"Set execute_immediately=true to stream to '{config.trajectory_topic}'."
        )

        while rclpy.ok():
            status = server.publish_step(control_publisher, rclpy.ok)
            if status != "initial":
                rate.sleep()

    except ros_manager.exceptions() as e:
        logger.info(f"ROSManager interrupted by user: {e}")
    finally:
        logger.info("Cleaning up...")
        if server is not None:
            server.close()
        ros_manager.shutdown()


if __name__ == "__main__":
    config = tyro.cli(PlannerConfig)
    main(config)

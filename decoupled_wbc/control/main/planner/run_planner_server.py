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
STATE_WAIT_TIMEOUT = 5.0  # give up on a plan if no robot state arrives before its ramp


@dataclass
class ActiveTrajectory:
    """One execution, advanced stage by stage: ramp_up -> execute -> settle -> ramp_down."""

    qpos: np.ndarray
    frame_idx: int
    stage: str = "ramp_up"
    home_qpos: Optional[np.ndarray] = None  # measured pose before ramp_up, target of ramp_down
    waiting_since: Optional[float] = None


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
        Advance the active execution by one step.

        Returns:
            "idle" - nothing to publish
            "initial" - ran a ramp (it already took its own time)
            "streaming" - published a normal waypoint
        """
        if self.recorder is not None:
            self.recorder.finish_if_settled()

        with self._lock:
            active = self._active
            if active is None:
                return "idle"
            frame_idx = active.frame_idx
            stage = active.stage
            qpos = active.qpos
            home_qpos = active.home_qpos

        publish_period = 1.0 / self.config.planner_frequency

        if stage == "ramp_up":
            # Capture where the robot is now: this is where ramp_down will bring it back.
            home = self.measured_qpos()
            if home is None:
                with self._lock:
                    if self._active is None:
                        return "idle"
                    if self._active.waiting_since is None:
                        self._active.waiting_since = time.monotonic()
                        self.logger.warn("Waiting for robot state before the start ramp")
                    waited = time.monotonic() - self._active.waiting_since
                if waited > STATE_WAIT_TIMEOUT:
                    return self.abandon(
                        f"no robot state after {STATE_WAIT_TIMEOUT:.0f}s; refusing to move"
                    )
                return "idle"
            if not self.confirm(
                f"ramp to the plan start over {self.config.initial_transition_time:.1f}s "
                f"(largest joint gap {np.abs(qpos[0] - home).max():.3f} rad)",
                control_publisher,
                home,
                keep_running,
            ):
                return self.abandon("cancelled before the start ramp")
            if self.recorder is not None:
                self.recorder.start(qpos, self.planner.name)
            self.publish_ramp(
                control_publisher, home, qpos[0], keep_running, "plan start"
            )
            with self._lock:
                if self._active is None or self._cancel_execution:
                    return "idle"
                self._active.home_qpos = home
                self._active.stage = "execute"
            return "initial"

        if stage == "execute":
            if frame_idx == 0:
                if not self.confirm(
                    f"execute the plan ({qpos.shape[0]} waypoints, "
                    f"{qpos.shape[0] * publish_period:.1f}s)",
                    control_publisher,
                    qpos[0],
                    keep_running,
                ):
                    return self.abandon("cancelled before execution")
                if self.recorder is not None:
                    self.recorder.mark_stream_start()
            self.publish_target(
                control_publisher, qpos[frame_idx], time.monotonic() + publish_period
            )
            with self._lock:
                if self._active is None or self._cancel_execution:
                    return "idle"
                active = self._active
                if active.frame_idx < active.qpos.shape[0] - 1:
                    active.frame_idx += 1
                else:
                    self.logger.info("Reached final planned waypoint; holding final pose.")
                    active.stage = "settle"
                    if self.recorder is not None:
                        self.recorder.mark_hold()
            return "streaming"

        if stage == "settle":
            # Hold the final pose until the recording has been saved, then offer ramp_down.
            self.publish_target(
                control_publisher, qpos[-1], time.monotonic() + publish_period
            )
            if self.recorder is None or not self.recorder.active:
                with self._lock:
                    if self._active is not None and not self._cancel_execution:
                        self._active.stage = "ramp_down"
            return "streaming"

        if stage == "ramp_down":
            if home_qpos is None or not self.config.ramp_down:
                return self.hold_or_stop(control_publisher, qpos[-1], publish_period)
            if not self.confirm(
                f"ramp back to the pose you started from over "
                f"{self.config.initial_transition_time:.1f}s "
                f"(largest joint gap {np.abs(np.asarray(home_qpos) - qpos[-1]).max():.3f} rad)",
                control_publisher,
                qpos[-1],
                keep_running,
            ):
                return self.abandon("cancelled before the return ramp")
            self.publish_ramp(
                control_publisher, qpos[-1], np.asarray(home_qpos), keep_running, "return"
            )
            with self._lock:
                if self._active is not None and not self._cancel_execution:
                    self._active.stage = "done"
            self.logger.info("Run complete; holding the pose you started from.")
            return "initial"

        return self.hold_or_stop(control_publisher, home_qpos, publish_period)

    def hold_or_stop(
        self, control_publisher: ROSMsgPublisher, pose, publish_period: float
    ) -> str:
        """Keep the last pose commanded, or drop the trajectory if holding is disabled."""
        if not self.config.hold_final_pose or pose is None:
            with self._lock:
                self._active = None
            return "idle"
        self.publish_target(
            control_publisher, np.asarray(pose), time.monotonic() + publish_period
        )
        return "streaming"

    def abandon(self, reason: str) -> str:
        """Operator declined a stage: stop commanding and keep whatever was recorded."""
        self.logger.info(f"{reason}; the robot holds its current pose.")
        if self.recorder is not None and self.recorder.active:
            self.recorder.finish()
        with self._lock:
            self._active = None
        return "idle"

    def confirm(
        self, what: str, control_publisher: ROSMsgPublisher, hold_qpos, keep_running
    ) -> bool:
        """Wait for the operator's Enter, holding ``hold_qpos`` at the publish rate meanwhile.

        The stream has to keep running: the controller injects a safe goal after 1 s
        without one, which would move the robot while it waits.
        """
        if not self.config.step:
            return True
        print(f"\n>>> press Enter to {what} (Ctrl-C to stop here)", flush=True)
        answer: list[bool] = []

        def read_line() -> None:
            try:
                input()
                answer.append(True)
            except (EOFError, KeyboardInterrupt):
                answer.append(False)

        waiter = threading.Thread(target=read_line, daemon=True)
        waiter.start()
        period = 1.0 / self.config.planner_frequency
        while not answer:
            if not keep_running() or self._cancel_execution:
                return False
            self.publish_target(
                control_publisher, np.asarray(hold_qpos), time.monotonic() + period
            )
            time.sleep(period)
        return bool(answer[0])

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

    def measured_qpos(self) -> Optional[np.ndarray]:
        """Planning-joint pose from the latest robot state, or None if none has arrived."""
        state = self.state_subscriber.get_msg()
        if state is None or "q" not in state:
            return None
        return full_q_to_planning_qpos(
            np.asarray(state["q"], dtype=np.float32).reshape(-1),
            self.controller_interface,
        )

    def publish_ramp(
        self,
        control_publisher: ROSMsgPublisher,
        from_qpos: np.ndarray,
        to_qpos: np.ndarray,
        keep_running,
        label: str,
    ) -> None:
        """Smoothstep ramp between two poses over initial_transition_time."""
        period = 1.0 / self.config.planner_frequency
        duration = self.config.initial_transition_time
        from_qpos = np.asarray(from_qpos, dtype=np.float32)
        to_qpos = np.asarray(to_qpos, dtype=np.float32)
        steps = max(1, int(round(duration * self.config.planner_frequency)))
        self.logger.info(
            f"Ramping to {label} over {duration:.1f}s ({steps} steps, "
            f"largest joint gap {np.abs(to_qpos - from_qpos).max():.3f} rad)"
        )
        t_next = time.monotonic()
        for pose in smoothstep_ramp(from_qpos, to_qpos, steps):
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

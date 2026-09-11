"""Constrained whole-body planning server for SONIC.

Plans on request and streams the result to the C++ deploy's ``pose`` topic
(29 joints at 50 Hz) over ZMQ; requests are msgpack dicts on ``service_port``.
"""

from __future__ import annotations

from pathlib import Path
import threading
import time
import traceback
from typing import Optional

import msgpack
import msgpack_numpy as mnp
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import tyro
import zmq

from gear_sonic.planner.configs import DEPLOY_FPS, ServerConfig
from gear_sonic.planner.joint_orders import name_permutation
from gear_sonic.planner.planning.constrained_rrt import (
    ConstrainedOMPLPlanner,
    default_planning_joint_names,
)
from gear_sonic.planner.stream.frame_builder import StreamFrames, build_stream_frames
from gear_sonic.planner.stream.pose_stream import PoseStreamPublisher

mnp.patch()


def quat_wxyz_to_rotvec(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=float).reshape(4)
    return Rotation.from_quat([x, y, z, w]).as_rotvec()


def load_endpoint_npz(path: str, plan_joint_names: list[str]) -> np.ndarray:
    """``goal/*.npz`` endpoint -> planning vector, joints mapped by name."""
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Endpoint npz not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    required = {"base_pos", "base_quat", "joint_pos_full", "joint_names_full"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"{npz_path} is missing keys: {missing}")
    names = [str(n) for n in data["joint_names_full"].tolist()]
    perm = name_permutation(names, list(plan_joint_names[1:]))
    joints = np.asarray(data["joint_pos_full"], dtype=float)[perm]
    return np.concatenate(
        [
            np.asarray(data["base_pos"], dtype=float).reshape(3),
            quat_wxyz_to_rotvec(data["base_quat"]),
            joints,
        ]
    )


class PlannerService:
    """Owns the planner, the publisher, and the single streaming thread."""

    def __init__(self, config: ServerConfig):
        config.velocity.validate()
        self.config = config

        if not Path(config.planning.urdf_path).exists():
            raise FileNotFoundError(
                f"Planning URDF not found: {config.planning.urdf_path}"
            )
        _, self.plan_joint_names = default_planning_joint_names(
            config.planning.urdf_path
        )

        self.publisher = PoseStreamPublisher(config.stream)
        # Switch the deploy to streamed-motion mode right away.  Besides
        # selecting the pose topic, this makes the deploy forward keyboard
        # input to its motion interface again -- so, as in decoupled_wbc,
        # a human can press ] in the deploy terminal to start control and
        # the planner only ever sends goals.
        self.publisher.select_streamed_motion()
        self._log(
            "Streamed-motion mode requested; press ] in the deploy "
            "terminal to start control (O still stops)"
        )

        self.state_reader = None
        if config.read_robot_state:
            from gear_sonic.planner.stream.state_reader import RobotStateReader

            self.state_reader = RobotStateReader(
                config.sim_scene_xml,
                domain_id=config.dds_domain_id,
                interface=config.dds_interface,
            )

        self.planner: Optional[ConstrainedOMPLPlanner] = None

        self._lock = threading.Lock()
        self._planning = False
        # Streaming state.  _generation is bumped for every new trajectory
        # (and on cancel); the stream thread captures its generation and
        # stops as soon as it is stale -- the race in the decoupled_wbc
        # publish_step (re-reading self._active after dropping the lock)
        # cannot happen because nothing mutates another generation's state.
        self._generation = 0
        self._stream_thread: Optional[threading.Thread] = None
        self._frames_sent = 0
        # Phased execution: plan/load stores frames here; ramp_to_start,
        # track, verify_goal and go_home all act on the stored trajectory.
        self._stored: Optional[StreamFrames] = None
        self._home: Optional[tuple] = None  # (joints_il, quat_wxyz) at first ramp
        self._last_ref_end: Optional[tuple] = None  # (joints_il, quat_wxyz) of the last stream
        self._log(f"Planning joints: {self.plan_joint_names[1:]}")
        self._log(
            f"Velocity caps: joints "
            f"{config.velocity.effective_joint_velocity():.3f} rad/s, base "
            f"{config.velocity.effective_base_velocity():.3f} m/s "
            f"(time_scale {config.velocity.time_scale})"
        )

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f"[PlannerService] {message}")


    def _resolve_endpoint(self, spec, role: str) -> np.ndarray:
        n_plan = 6 + len(self.plan_joint_names) - 1
        if spec is None:
            if role != "start":
                raise ValueError("A goal endpoint is required")
            if self.state_reader is not None:
                q = self.state_reader.read_q_plan(self.plan_joint_names)
                if q is not None:
                    self._log("Start read from robot state (DDS)")
                    return q
                self._log("Robot state unavailable; using reference start")
            return load_endpoint_npz(
                self.config.reference_start_npz, self.plan_joint_names
            )
        if "q_plan" in spec:
            q = np.asarray(spec["q_plan"], dtype=float).reshape(-1)
            if q.shape[0] != n_plan:
                raise ValueError(
                    f"{role} q_plan has {q.shape[0]} DoF, expected {n_plan}"
                )
            return q
        if "npz" in spec:
            return load_endpoint_npz(str(spec["npz"]), self.plan_joint_names)
        required = {"base_pos", "base_quat", "joint_pos", "joint_names"}
        if required.issubset(spec):
            names = [str(n) for n in spec["joint_names"]]
            perm = name_permutation(names, list(self.plan_joint_names[1:]))
            joints = np.asarray(spec["joint_pos"], dtype=float)[perm]
            return np.concatenate(
                [
                    np.asarray(spec["base_pos"], dtype=float).reshape(3),
                    quat_wxyz_to_rotvec(spec["base_quat"]),
                    joints,
                ]
            )
        raise ValueError(
            f"Unrecognized {role} endpoint; expected q_plan / npz / "
            f"(base_pos, base_quat, joint_pos, joint_names), got "
            f"{sorted(spec)}"
        )


    def _ensure_planner(self, start: np.ndarray) -> None:
        """(Re)build the planner when the pinned feet no longer match: a
        start standing elsewhere would be projected onto the old feet
        placement, silently teleporting them."""
        cfg = self.config.planning
        if self.planner is not None:
            feet_error = float(
                np.linalg.norm(self.planner.feet_constraint.error(start))
            )
            if feet_error <= cfg.feet_retarget_tolerance:
                return
            self._log(
                f"Start's feet deviate {feet_error:.4f} from the pinned "
                f"targets (> {cfg.feet_retarget_tolerance}); rebuilding the "
                f"planner around the new start"
            )
        import pinocchio as pin

        collision_rig = None
        if cfg.check_collisions:
            from gear_sonic.planner.collision import (
                DEFAULT_COLLISION_SCENE,
                CollisionRig,
            )

            collision_rig = CollisionRig(
                cfg.collision_scene or DEFAULT_COLLISION_SCENE,
                finger_closure=cfg.finger_closure,
            )
            self._log(
                f"Self-collision checking ON (fingers at "
                f"{cfg.finger_closure:.0%} closure)"
            )

        probe_model, _ = default_planning_joint_names(cfg.urdf_path)

        if cfg.planner_kind == "phaserrtstar":
            # PhaseRRTstar searches (q, alpha) along a reference trajectory,
            # so it needs one; the plain planner does not.
            from gear_sonic.planner.planning.constrained_phaserrtstar import (
                QUIET_BASE_WEIGHTS,
                ConstrainedPhaseRRTstarPlanner,
                load_reference_npz,
            )

            phase = cfg.phase
            reference, ref_info = load_reference_npz(
                phase.reference_npz, self.plan_joint_names
            )
            self._log(
                f"PhaseRRTstar reference: {ref_info['frames']} frames, "
                f"arclength {ref_info['arclength']:.3f}"
            )
            self.planner = ConstrainedPhaseRRTstarPlanner(
                robot_model=None,
                urdf_path=cfg.urdf_path,
                planning_joint_names=self.plan_joint_names,
                q_nominal=pin.neutral(probe_model),
                q_reference=start,
                reference=reference,
                validity_resolution=phase.validity_resolution,
                com_margin=cfg.com_margin,
                projection_delta=phase.projection_delta,
                projection_lambda=phase.projection_lambda,
                rewire_factor=phase.rewire_factor,
                sample_weights=(
                    QUIET_BASE_WEIGHTS if phase.quiet_base_sample_weights else None
                ),
                phase_params=phase.overrides(),
                collision_rig=collision_rig,
                collision_tolerance=cfg.collision_tolerance,
                seed=phase.seed,
                log=self.config.verbose,
            )
            return

        self.planner = ConstrainedOMPLPlanner(
            robot_model=None,  # collision hookup disabled; see validity_checker
            urdf_path=cfg.urdf_path,
            planning_joint_names=self.plan_joint_names,
            q_nominal=pin.neutral(probe_model),
            q_reference=start,
            planner=cfg.planner,
            validity_resolution=cfg.validity_resolution,
            extend_range=cfg.extend_range,
            com_margin=cfg.com_margin,
            projection_delta=cfg.projection_delta,
            projection_lambda=cfg.projection_lambda,
            goal_threshold=cfg.goal_threshold,
            collision_rig=collision_rig,
            collision_tolerance=cfg.collision_tolerance,
            log=self.config.verbose,
        )


    def _start_stream(self, frames: StreamFrames) -> None:
        with self._lock:
            self._generation += 1
            generation = self._generation
        previous = self._stream_thread
        if previous is not None and previous.is_alive():
            previous.join(timeout=3.0)

        def run() -> None:
            def keep_going() -> bool:
                with self._lock:
                    return self._generation == generation

            sent = self.publisher.stream(frames, should_continue=keep_going)
            with self._lock:
                self._frames_sent = sent
            self._log(f"Stream finished: {sent}/{frames.num_frames} frames")

        thread = threading.Thread(target=run, name="pose-stream", daemon=True)
        self._stream_thread = thread
        thread.start()

    def cancel(self) -> dict:
        with self._lock:
            self._generation += 1
        return {"ok": True, "cancelled": True}


    def _measure_pose(self) -> tuple:
        """Robot's current (joints IL-order, base quat wxyz) from DDS."""
        if self.state_reader is None:
            raise RuntimeError(
                "ramping needs the robot state (start the server with "
                "read_robot_state enabled and the sim running)"
            )
        q_plan = self.state_reader.read_q_plan(self.plan_joint_names)
        if q_plan is None:
            raise RuntimeError("robot state unavailable or stale over DDS")
        from gear_sonic.planner.joint_orders import isaaclab_joint_names

        to_il = name_permutation(
            list(self.plan_joint_names[1:]), isaaclab_joint_names()
        )
        joints_il = q_plan[6:][to_il]
        rotation = Rotation.from_rotvec(q_plan[3:6])
        x, y, z, w = rotation.as_quat()
        return joints_il, np.array([w, x, y, z])

    _LEG_KEYWORDS = ("hip", "knee", "ankle")

    def _leg_indices_il(self) -> np.ndarray:
        from gear_sonic.planner.joint_orders import isaaclab_joint_names

        return np.array(
            [i for i, n in enumerate(isaaclab_joint_names())
             if any(k in n for k in self._LEG_KEYWORDS)]
        )

    def _ramp_frames(self, joints_from, quat_from, joints_to, quat_to) -> StreamFrames:
        """Smoothstep joint ramp + quaternion slerp at the deploy rate."""
        n = max(2, int(round(self.config.ramp_duration * DEPLOY_FPS)))
        alpha = np.linspace(0.0, 1.0, n)
        ease = alpha * alpha * (3.0 - 2.0 * alpha)          # smoothstep
        joints = joints_from[None, :] + ease[:, None] * (joints_to - joints_from)[None, :]
        r = Rotation.from_quat(np.array([quat_from, quat_to])[:, [1, 2, 3, 0]])
        quats = Slerp([0.0, 1.0], r)(ease).as_quat()[:, [3, 0, 1, 2]]
        vel = np.gradient(joints, 1.0 / DEPLOY_FPS, axis=0)
        return StreamFrames(
            joint_pos=joints.astype(np.float32),
            joint_vel=vel.astype(np.float32),
            body_quat=quats.astype(np.float32),
            fps=DEPLOY_FPS,
            peak_joint_velocity=float(np.abs(np.diff(joints, axis=0)).max() * DEPLOY_FPS),
            peak_base_velocity=0.0,
        )

    def _stream_and_wait(self, frames: StreamFrames, margin: float = 5.0) -> None:
        self._start_stream(frames)
        thread = self._stream_thread
        if thread is not None:
            thread.join(timeout=frames.duration + margin)

    def _ramp_to(self, joints_to, quat_to, label: str) -> dict:
        joints_now, quat_now = self._measure_pose()
        legs = self._leg_indices_il()
        leg_dev = float(np.abs(joints_now[legs] - joints_to[legs]).max())
        if leg_dev > self.config.ramp_leg_threshold:
            raise RuntimeError(
                f"refusing to ramp to {label}: leg joints deviate "
                f"{leg_dev:.3f} rad (> {self.config.ramp_leg_threshold}); "
                f"ramps are unconstrained interpolation and unsafe for leg "
                f"motion -- reach this pose by tracking a planned segment"
            )
        if self._home is None:
            self._home = (joints_now.copy(), quat_now.copy())
            self._log("home pose captured")
        frames = self._ramp_frames(joints_now, quat_now, joints_to, quat_to)
        self._log(
            f"ramping to {label}: {frames.num_frames} frames "
            f"({frames.duration:.1f}s), initial max dev "
            f"{np.abs(joints_now - joints_to).max():.3f} rad"
        )
        self._stream_and_wait(frames)
        self._last_ref_end = (np.asarray(joints_to, dtype=float),
                              np.asarray(quat_to, dtype=float))
        time.sleep(0.5)  # settle
        joints_after, _ = self._measure_pose()
        achieved = float(np.abs(joints_after - joints_to).max())
        return {
            "ok": True,
            "ramped": True,
            "achieved_error": achieved,
            "within_tolerance": achieved <= self.config.start_tolerance,
            "tolerance": self.config.start_tolerance,
            "duration_s": frames.duration,
        }

    #: reference-continuity tolerance for chained segments [rad]
    _CHAIN_TOLERANCE = 0.1

    def ramp_to_start(self) -> dict:
        if self._stored is None:
            raise RuntimeError("no stored trajectory: run 'plan' or 'load_frames' first")
        # Chained segments: if the PREVIOUS reference ended where this one
        # begins the stream is continuous and no ramp is needed.  Ramping on
        # robot pose instead would always refuse, since the robot never stands
        # exactly on a bend start -- that offset is ordinary tracking error,
        # present throughout tracking anyway.
        last = self._last_ref_end
        if last is not None:
            gap = float(np.abs(
                last[0] - np.asarray(self._stored.joint_pos[0], dtype=float)
            ).max())
            if gap <= self._CHAIN_TOLERANCE:
                self._log(
                    f"ramp skipped: reference continuous with previous "
                    f"segment (gap {gap:.3f} rad)"
                )
                return {
                    "ok": True,
                    "ramped": False,
                    "skipped": "reference continuous",
                    "achieved_error": gap,
                    "within_tolerance": True,
                    "tolerance": self._CHAIN_TOLERANCE,
                    "duration_s": 0.0,
                }
        return self._ramp_to(
            np.asarray(self._stored.joint_pos[0], dtype=float),
            np.asarray(self._stored.body_quat[0], dtype=float),
            "trajectory start",
        )

    def track(self) -> dict:
        if self._stored is None:
            raise RuntimeError("no stored trajectory: run 'plan' or 'load_frames' first")
        self._start_stream(self._stored)
        self._last_ref_end = (
            np.asarray(self._stored.joint_pos[-1], dtype=float),
            np.asarray(self._stored.body_quat[-1], dtype=float),
        )
        return {
            "ok": True,
            "tracking": True,
            "num_frames": self._stored.num_frames,
            "duration_s": self._stored.duration,
        }

    def verify_goal(self) -> dict:
        if self._stored is None:
            raise RuntimeError("no stored trajectory")
        joints_now, _ = self._measure_pose()
        goal = np.asarray(self._stored.joint_pos[-1], dtype=float)
        err = np.abs(joints_now - goal)
        return {
            "ok": True,
            "goal_max_error": float(err.max()),
            "goal_rmse": float(np.sqrt((err ** 2).mean())),
            "within_tolerance": float(err.max()) <= self.config.start_tolerance,
        }

    def go_home(self) -> dict:
        if self._home is None:
            raise RuntimeError("no home pose captured yet (home is set on the first ramp)")
        return self._ramp_to(self._home[0], self._home[1], "home")


    def handle_plan(self, request: dict) -> dict:
        with self._lock:
            if self._planning:
                raise RuntimeError("Planner is busy with another request")
            self._planning = True
        try:
            start = self._resolve_endpoint(request.get("start"), "start")
            goal = self._resolve_endpoint(request.get("goal"), "goal")
            timeout = float(request.get("timeout") or self.config.planning.timeout)

            self._ensure_planner(start)
            t0 = time.monotonic()
            waypoints = self.planner.plan(
                start,
                goal,
                timeout=timeout,
                smooth_path=self.config.planning.smooth_path,
                shortcut_path=self.config.planning.shortcut_path,
            )
            planning_time = time.monotonic() - t0
            if waypoints.shape[0] < 2:
                raise RuntimeError(
                    f"No exact solution within {timeout}s "
                    f"(stats: {self.planner.last_plan_stats})"
                )

            frames = build_stream_frames(
                waypoints,
                self.plan_joint_names[1:],
                self.config.velocity,
                fps=DEPLOY_FPS,
                project=self.planner.constraint.project,
            )
            # Feet drift over the frames actually streamed (plan order),
            # subsampled to ~50 evaluations.
            stride = max(1, frames.num_frames // 50)
            feet_error_frames = max(
                float(np.linalg.norm(self.planner.feet_constraint.error(q)))
                for q in frames.plan_frames[::stride]
            )

            # Phased execution: the plan is STORED, never auto-streamed.
            # Drive it with ramp_to_start -> track -> verify_goal (the old
            # execute:true fire-and-move behavior caused a step command
            # whenever the robot was not at the plan's start).
            self._stored = frames

            response = {
                "ok": True,
                "num_waypoints": int(waypoints.shape[0]),
                "num_frames": frames.num_frames,
                "duration_s": frames.duration,
                "planning_time_s": planning_time,
                "peak_joint_velocity": frames.peak_joint_velocity,
                "peak_base_velocity": frames.peak_base_velocity,
                "joint_velocity_cap": self.config.velocity.effective_joint_velocity(),
                "base_velocity_cap": self.config.velocity.effective_base_velocity(),
                "start_error": float(np.max(np.abs(waypoints[0] - start))),
                "goal_error": float(np.max(np.abs(waypoints[-1] - goal))),
                "max_feet_error_frames": feet_error_frames,
                "plan_stats": dict(self.planner.last_plan_stats),
                "stored": True,
                "executed": False,
            }
            if request.get("return_frames"):
                response["frames"] = {
                    "joint_pos": frames.joint_pos,
                    "joint_vel": frames.joint_vel,
                    "body_quat": frames.body_quat,
                    "fps": frames.fps,
                }
            self._log(
                f"Planned {waypoints.shape[0]} waypoints -> "
                f"{frames.num_frames} frames ({frames.duration:.2f} s) in "
                f"{planning_time:.2f} s; peak joint velocity "
                f"{frames.peak_joint_velocity:.3f} rad/s"
            )
            return response
        finally:
            with self._lock:
                self._planning = False

    def status(self) -> dict:
        with self._lock:
            streaming = (
                self._stream_thread is not None and self._stream_thread.is_alive()
            )
            return {
                "ok": True,
                "planner_ready": self.planner is not None,
                "streaming": streaming,
                "frames_sent": self._frames_sent,
                "next_frame_index": self.publisher.next_frame_index,
            }

    def handle(self, request: dict) -> dict:
        action = str(request.get("action", "plan"))
        if action == "plan":
            return self.handle_plan(request)
        if action == "cancel":
            return self.cancel()
        if action == "status":
            return self.status()
        if action == "stream_frames":
            # Replay caller-supplied frames without planning.  The server
            # owns the ZMQ publisher (it binds :5556), so a separate test
            # process cannot stream on its own -- it hands the frames here.
            frames = request["frames"]
            stream = StreamFrames(
                joint_pos=np.asarray(frames["joint_pos"], dtype=np.float32),
                joint_vel=np.asarray(frames["joint_vel"], dtype=np.float32),
                body_quat=np.asarray(frames["body_quat"], dtype=np.float32),
                fps=float(frames.get("fps", DEPLOY_FPS)),
                peak_joint_velocity=float(frames.get("peak_joint_velocity", 0.0)),
                peak_base_velocity=0.0,
            )
            if request.get("store_only"):
                self._stored = stream
                return {
                    "ok": True,
                    "num_frames": stream.num_frames,
                    "duration_s": stream.duration,
                    "stored": True,
                }
            self._start_stream(stream)
            return {
                "ok": True,
                "num_frames": stream.num_frames,
                "duration_s": stream.duration,
                "streaming": True,
            }
        if action == "ramp_to_start":
            return self.ramp_to_start()
        if action == "track":
            return self.track()
        if action == "verify_goal":
            return self.verify_goal()
        if action == "go_home":
            return self.go_home()
        if action == "start_control":
            # Remote equivalent of pressing ] in the deploy terminal.
            self.publisher.send_command(planner=False, start=True)
            return {"ok": True, "start_sent": True}
        if action == "stop_control":
            self.publisher.send_command(planner=False, stop=True)
            return {"ok": True, "stop_sent": True}
        raise ValueError(f"Unknown action '{action}'")

    def close(self) -> None:
        with self._lock:
            self._generation += 1
        if self._stream_thread is not None and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=3.0)
        self.publisher.close()


def main(config: ServerConfig) -> int:
    service = PlannerService(config)
    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{config.service_port}")
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    print(
        f"[PlannerService] Plan service on tcp://*:{config.service_port}; "
        f"streaming to {service.publisher.endpoint} "
        f"(topic '{config.stream.pose_topic}')"
    )
    try:
        while True:
            if not poller.poll(timeout=200):
                continue
            request = msgpack.unpackb(socket.recv(), object_hook=mnp.decode)
            try:
                response = service.handle(request)
            except Exception as exc:
                traceback.print_exc()
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            socket.send(msgpack.packb(response, default=mnp.encode))
    except KeyboardInterrupt:
        print("\n[PlannerService] Interrupted; cleaning up...")
    finally:
        service.close()
        socket.close(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(ServerConfig)))

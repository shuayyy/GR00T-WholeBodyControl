"""ZMQ streamed-motion publisher for the SONIC C++ deploy.

Packed-message protocol on its input port: ``[topic][1280-byte JSON
header][little-endian fields]``.  The deploy connects, so this binds.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np
import zmq

from gear_sonic.planner.configs import StreamConfig
from gear_sonic.planner.stream.frame_builder import StreamFrames
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    HEADER_SIZE,
    build_command_message,
    pack_pose_message,
)

#: Stream lead over deploy playback [s].  Must exceed the policy's 0.92 s
#: reference lookahead (10 samples at stride 5, 50 Hz) or the deploy clamps
#: the future samples to the last frame received.
_REALTIME_LEAD = 2.0

#: PUB/SUB slow-joiner guard after bind before the first message.
_BIND_SETTLE_S = 0.5


class PoseStreamPublisher:
    """Publishes command + pose messages; the deploy connects to us."""

    def __init__(self, config: StreamConfig):
        self.config = config
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.endpoint = f"tcp://{config.zmq_bind}:{config.zmq_port}"
        self.socket.bind(self.endpoint)
        # ZMQ sockets are not thread-safe; streaming and commands race.
        self._send_lock = threading.Lock()
        #: session-global: successive trajectories continue one timeline,
        #: so the merger appends instead of re-aligning.
        self.next_frame_index = 0
        time.sleep(_BIND_SETTLE_S)

    def close(self) -> None:
        self.socket.close(0)

    def _send(self, message: bytes) -> None:
        with self._send_lock:
            self.socket.send(message)

    def send_command(
        self, planner: bool, start: bool = False, stop: bool = False
    ) -> None:
        self._send(build_command_message(start=start, stop=stop, planner=planner))

    def select_streamed_motion(self) -> None:
        """Switch the deploy's input manager to the pose topic."""
        self.send_command(planner=False)

    def _chunk_message(
        self, frames: StreamFrames, i0: int, i1: int, index_offset: int
    ) -> bytes:
        pose_data = {
            "joint_pos": np.ascontiguousarray(frames.joint_pos[i0:i1]),
            "joint_vel": np.ascontiguousarray(frames.joint_vel[i0:i1]),
            "body_quat": np.ascontiguousarray(frames.body_quat[i0:i1]),
            "frame_index": np.arange(
                index_offset + i0, index_offset + i1, dtype=np.int64
            ),
            "catch_up": np.array([self.config.catch_up], dtype=bool),
        }
        return pack_pose_message(
            pose_data, topic=self.config.pose_topic, version=1
        )

    def stream(
        self,
        frames: StreamFrames,
        index_offset: Optional[int] = None,
        should_continue: Optional[Callable[[], bool]] = None,
    ) -> int:
        """Send all frames in chunks; returns the number sent.

        With ``config.realtime`` each chunk goes out ``_REALTIME_LEAD`` ahead
        of playback, so a cancel stops the robot within the lead.
        """
        self.select_streamed_motion()
        if index_offset is None:
            index_offset = self.next_frame_index
        chunk = max(1, int(self.config.chunk_frames))
        t0 = time.monotonic()
        sent = 0
        for i0 in range(0, frames.num_frames, chunk):
            if should_continue is not None and not should_continue():
                break
            if self.config.realtime and i0 > 0:
                target = t0 + i0 / frames.fps - _REALTIME_LEAD
                delay = target - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            i1 = min(i0 + chunk, frames.num_frames)
            self._send(self._chunk_message(frames, i0, i1, index_offset))
            sent = i1
        self.next_frame_index = index_offset + sent
        return sent


def decode_packed_message(message: bytes, topic: str) -> dict[str, np.ndarray]:
    """Independent decoder mirroring ``ZMQPackedMessageSubscriber``.

    Verifies that what we pack is what the C++ side will parse.
    """
    import json

    prefix = topic.encode()
    if not message.startswith(prefix):
        raise ValueError(f"Message does not start with topic {topic!r}")
    body = message[len(prefix):]
    if len(body) < HEADER_SIZE:
        raise ValueError("Message shorter than the header")
    header = json.loads(body[:HEADER_SIZE].rstrip(b"\x00").decode())
    if header.get("endian", "le") != "le":
        raise ValueError("Only little-endian messages are produced")
    dtype_map = {
        "f32": np.float32,
        "f64": np.float64,
        "i32": np.int32,
        "i64": np.int64,
        "u8": np.uint8,
        "bool": np.bool_,
    }
    out: dict[str, np.ndarray] = {}
    offset = HEADER_SIZE
    for field in header["fields"]:
        dtype = np.dtype(dtype_map[field["dtype"]]).newbyteorder("<")
        count = int(np.prod(field["shape"]))
        nbytes = count * dtype.itemsize
        out[field["name"]] = np.frombuffer(
            body, dtype=dtype, count=count, offset=offset
        ).reshape(field["shape"])
        offset += nbytes
    if offset != len(body):
        raise ValueError(
            f"Payload size mismatch: consumed {offset - HEADER_SIZE}, "
            f"got {len(body) - HEADER_SIZE}"
        )
    return out

"""Read the robot's current configuration from the sim's DDS topics.

Sim-only: real hardware has no world base position, and this refuses rather
than guessing.  Joints map to planning order by name.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.planner.joint_orders import mjcf_joint_names, name_permutation


class RobotStateReader:
    """Latest-value cache over rt/lowstate + rt/odostate."""

    def __init__(
        self,
        sim_scene_xml: str,
        domain_id: int = 0,
        interface: str = "lo",
    ):
        # Deferred: keeps the planner package independent of unitree_sdk2py.
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, OdoState_

        try:
            if interface:
                ChannelFactoryInitialize(domain_id, interface)
            else:
                ChannelFactoryInitialize(domain_id)
        except Exception as exc:  # already initialized in this process
            print(f"[RobotStateReader] ChannelFactory init note: {exc}")

        #: motor order = scene XML body joints in document order
        self.motor_joint_names = mjcf_joint_names(sim_scene_xml)

        self._lock = threading.Lock()
        self._low_state = None
        self._odo_state = None
        self._low_state_time = 0.0
        self._odo_state_time = 0.0

        self._low_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._low_sub.Init(self._on_low_state, 1)
        self._odo_sub = ChannelSubscriber("rt/odostate", OdoState_)
        self._odo_sub.Init(self._on_odo_state, 1)

    def _on_low_state(self, msg) -> None:
        with self._lock:
            self._low_state = msg
            self._low_state_time = time.monotonic()

    def _on_odo_state(self, msg) -> None:
        with self._lock:
            self._odo_state = msg
            self._odo_state_time = time.monotonic()

    def wait_for_state(self, timeout: float = 2.0) -> bool:
        """True once both topics have delivered at least one message."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._low_state is not None and self._odo_state is not None:
                    return True
            time.sleep(0.02)
        return False

    def read_q_plan(
        self,
        plan_joint_names: list[str],
        timeout: float = 2.0,
        max_staleness: float = 0.5,
    ) -> Optional[np.ndarray]:
        """Current configuration as ``[base_pos(3), base_rotvec(3),
        joints(29)]``, or None.  ``plan_joint_names`` is free-flyer first.
        """
        if not self.wait_for_state(timeout):
            return None
        with self._lock:
            low, odo = self._low_state, self._odo_state
            age = time.monotonic() - min(self._low_state_time, self._odo_state_time)
        if age > max_staleness:
            print(f"[RobotStateReader] state is {age:.2f}s stale; refusing")
            return None

        n_motors = len(self.motor_joint_names)
        joints_motor = np.array(
            [low.motor_state[i].q for i in range(n_motors)], dtype=float
        )
        to_plan = name_permutation(self.motor_joint_names, list(plan_joint_names[1:]))
        joints_plan = joints_motor[to_plan]

        base_pos = np.array(odo.position[:3], dtype=float)
        w, x, y, z = np.array(odo.orientation[:4], dtype=float)  # wxyz
        rotvec = Rotation.from_quat([x, y, z, w]).as_rotvec()

        return np.concatenate([base_pos, rotvec, joints_plan])

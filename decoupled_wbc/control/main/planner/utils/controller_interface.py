"""Map between the 17 planning joints and the controller's upper-body vector (by name)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from decoupled_wbc.control.main.planner.simulation.robot import JOINT_NAMES_UP
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model


@dataclass
class ControllerUpperBodyInterface:
    indices: list[int]
    """Positions of the upper-body joints inside the controller's full joint vector."""
    joint_names: list[str]
    default_qpos: np.ndarray
    name_to_index: dict[str, int]


def build_controller_interface(
    enable_waist: bool, high_elbow_pose: bool
) -> tuple[ControllerUpperBodyInterface, list[str]]:
    """Return the upper-body interface and the controller's full joint-name list."""
    robot_model = instantiate_g1_robot_model(
        waist_location="lower_and_upper_body" if enable_waist else "lower_body",
        high_elbow_pose=high_elbow_pose,
    )
    indices = robot_model.get_joint_group_indices("upper_body")
    joint_names = [robot_model.joint_names[i] for i in indices]
    interface = ControllerUpperBodyInterface(
        indices=indices,
        joint_names=joint_names,
        default_qpos=robot_model.default_body_pose[indices].astype(np.float32, copy=True),
        name_to_index={name: i for i, name in enumerate(joint_names)},
    )
    missing = [name for name in JOINT_NAMES_UP if name not in interface.name_to_index]
    if missing:
        raise ValueError(f"Controller upper_body missing planning joints: {missing}")
    return interface, list(robot_model.joint_names)


def upper_body_pose_to_planning_qpos(
    upper_body_pose: np.ndarray, interface: ControllerUpperBodyInterface
) -> np.ndarray:
    """Controller-ordered upper-body vector -> 17-DoF planning configuration."""
    upper_body_pose = np.asarray(upper_body_pose, dtype=np.float32).reshape(-1)
    if upper_body_pose.shape[0] != len(interface.joint_names):
        raise ValueError(
            f"Upper-body pose length {upper_body_pose.shape[0]} does not match "
            f"controller upper_body size {len(interface.joint_names)}"
        )
    return np.array(
        [upper_body_pose[interface.name_to_index[name]] for name in JOINT_NAMES_UP],
        dtype=np.float32,
    )


def full_q_to_planning_qpos(
    full_q: np.ndarray, interface: ControllerUpperBodyInterface
) -> np.ndarray:
    """Controller full joint vector (robot state ``q``) -> 17-DoF planning configuration."""
    full_q = np.asarray(full_q, dtype=np.float32).reshape(-1)
    if full_q.shape[0] < max(interface.indices) + 1:
        raise ValueError(
            f"Robot state q length {full_q.shape[0]} is too short for upper-body indices"
        )
    return upper_body_pose_to_planning_qpos(full_q[interface.indices], interface)


def planning_qpos_to_controller_pose(
    planning_q: np.ndarray, interface: ControllerUpperBodyInterface
) -> np.ndarray:
    """17-DoF planning configuration -> controller upper-body target (hands at default)."""
    planning_q = np.asarray(planning_q, dtype=np.float32).reshape(-1)
    if planning_q.shape[0] != len(JOINT_NAMES_UP):
        raise ValueError(
            f"Expected planning qpos length {len(JOINT_NAMES_UP)}, got {planning_q.shape[0]}"
        )
    target = interface.default_qpos.copy()
    for value, name in zip(planning_q, JOINT_NAMES_UP):
        target[interface.name_to_index[name]] = value
    return target

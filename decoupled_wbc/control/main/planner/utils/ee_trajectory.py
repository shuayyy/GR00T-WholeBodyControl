"""Wrist trajectories in the pelvis frame, from joint angles."""

import os

import mujoco
import numpy as np

_PLANNER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_XML = os.path.join(_PLANNER_DIR, "assets", "unitree_g1", "g1_with_hands.xml")

ROOT_BODY = "pelvis"
EE_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def ee_in_root_frame(joint_pos, joint_names, xml_path=DEFAULT_XML, ee_bodies=EE_BODIES):
    """Wrist poses in the pelvis frame: {body: {"pos": (N, 3), "quat": (N, 4) wxyz}}.
    Joints are matched by name."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    adr = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if jid < 0:
            raise ValueError(f"{name} is not a joint of {xml_path}")
        adr.append(int(model.jnt_qposadr[jid]))
    adr = np.asarray(adr, dtype=int)

    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
    ids = {b: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in ee_bodies}
    for b, i in ids.items():
        if i < 0:
            raise ValueError(f"{b} is not a body of {xml_path}")

    joint_pos = np.asarray(joint_pos, dtype=float)
    n = joint_pos.shape[0]
    out = {b: {"pos": np.zeros((n, 3)), "quat": np.zeros((n, 4))} for b in ee_bodies}
    for f in range(n):
        data.qpos[adr] = joint_pos[f]
        mujoco.mj_forward(model, data)
        root_pos = data.xpos[root]
        root_quat_inv = _quat_conj(data.xquat[root])
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, root_quat_inv)
        rot = rot.reshape(3, 3)
        for b, i in ids.items():
            out[b]["pos"][f] = rot @ (data.xpos[i] - root_pos)
            out[b]["quat"][f] = _quat_mul(root_quat_inv, data.xquat[i])
    return out

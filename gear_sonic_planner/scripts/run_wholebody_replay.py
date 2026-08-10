#!/usr/bin/env python3
"""Replay a motionbricks G1 motion NPZ through the SONIC whole-body policy.

Behaves like planner_wbc's trajectory replay, but for the whole body: pass the
NPZ directly and this script converts it internally (to a reference folder
under ``gear_sonic_planner/reference/``) and then launches the SONIC deploy on
it. The policy tracks all 29 joints, legs included.

    # Terminal 1 — MuJoCo simulator
    python gear_sonic/scripts/run_sim_loop.py

    # Terminal 2 — replay the NPZ (converts internally, then starts SONIC)
    python gear_sonic_planner/scripts/run_wholebody_replay.py \\
        dataset/wholebody_box.npz

Then press ``]`` to start control and ``T`` to play the motion.

Pipeline
--------
1. NPZ (34-joint ``G1Skeleton34`` arrays, y-up/z-forward) -> MuJoCo qpos
   (3 root translation + 4 root quaternion wxyz + 29 hinge joints) per frame.
   Mirrors ``motionbricks.helper.mujoco_helper.mujoco_qpos_converter`` without
   needing torch or a MotionRepBase.
2. Resample from the source frame rate to the 50 Hz the deploy C++ consumes
   (``localmotion_kplanner.hpp`` steps references at 50 Hz).
3. MuJoCo forward kinematics on ``g1_29dof_rev_1_0.xml`` for the world pose of
   the 14 reference bodies (same list as ``commands/terms/motion.yaml``).
4. Central-difference velocities (matches ``ComputeGlobalVelocities`` in
   ``motion_data_reader.hpp``).
5. Write the CSV set + ``metadata.txt`` that ``MotionDataReader`` expects.
   Joint columns are in IsaacLab order. (policy_parameters.hpp claims reference
   motions are MuJoCo order, but FK on the shipped example proves otherwise:
   posing g1_29dof_rev_1_0.xml with the example's joint_pos reproduces its
   body_pos to 2 mm under the IsaacLab interpretation vs 23 cm under MuJoCo.)
6. Launch ``gear_sonic_deploy/deploy.sh --motion-data <reference>/ sim`` (or
   ``real`` with ``--real``). Use ``--convert-only`` to skip the launch.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REPO_ROOT = Path(__file__).resolve().parents[2]
MOTIONBRICKS_ROOT = REPO_ROOT / "motionbricks"
MOTIONBRICKS_XML = MOTIONBRICKS_ROOT / "assets" / "skeletons" / "g1" / "g1.xml"
FK_XML = REPO_ROOT / "gear_sonic" / "data" / "assets" / "robot_description" / "mjcf" / "g1_29dof_rev_1_0.xml"
DEFAULT_OUTPUT_BASE = REPO_ROOT / "gear_sonic_planner" / "reference"
DEPLOY_FPS = 50.0

# motionbricks features are y-up / z-forward; MuJoCo is z-up / x-forward.
MUJOCO_TO_MOTION = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
MOTION_TO_MUJOCO = MUJOCO_TO_MOTION.T

# A MuJoCo hinge axis, expressed in the motion frame, as a selector over the
# (x, y, z) Euler terms decomposed from the local rotation.
_AXIS_IN_MOTION_SPACE = {
    0: np.array([0.0, 0.0, 1.0]),  # mujoco x -> motion z
    1: np.array([1.0, 0.0, 0.0]),  # mujoco y -> motion x
    2: np.array([0.0, 1.0, 0.0]),  # mujoco z -> motion y
}

# The 14 reference bodies, in the order used by training
# (gear_sonic/config/manager_env/commands/terms/motion.yaml: body_names) and by
# every reference in gear_sonic_deploy/reference/example/.
REFERENCE_BODIES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

# IsaacLab-convention indexes of those 14 bodies in the full skeleton, parsed
# by MotionDataReader::ReadMetadata from the "Body part indexes:" line.
# Copied verbatim from the shipped example references.
BODY_PART_INDEXES = [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]

# MJ_TO_IL[mj] = il, copied from gear_sonic/data_process/convert_soma_csv_to_motion_lib.py.
# Verified against the shipped example by FK (2 mm body-position agreement).
MJ_TO_IL = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
)


# ---------------------------------------------------------------------------
# NPZ -> qpos (motionbricks skeleton conversion)
# ---------------------------------------------------------------------------


def import_g1_skeleton():
    """Import ``G1Skeleton34`` without requiring motionbricks to be installed."""
    if str(MOTIONBRICKS_ROOT) not in sys.path:
        sys.path.insert(0, str(MOTIONBRICKS_ROOT))
    from motionbricks.motionlib.core.skeletons.g1 import G1Skeleton34

    return G1Skeleton34


class G1MotionToQpos:
    """Convert motionbricks G1 skeleton rotations into MuJoCo ``qpos``."""

    def __init__(self, xml_path: Path):
        skeleton = import_g1_skeleton()
        self.motion_joint_names = [name for name, _ in skeleton.bone_order_names_with_parents]
        motion_index = {name: i for i, name in enumerate(self.motion_joint_names)}
        self.n_motion_joints = len(self.motion_joint_names)

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Hinge-joint axes may be declared inline or inherited from a default class.
        class_axes = {}
        for element in tree.findall(".//default"):
            if "class" not in element.attrib:
                continue
            joints = element.findall("joint")
            if joints:
                class_axes[element.get("class")] = joints[0].get("axis")

        # <freejoint> is a distinct tag, so this finds the 29 hinge joints only.
        hinge_joints = root.find("worldbody").findall(".//joint")
        parent_of = {child: parent for parent in root.iter() for child in parent}

        self.mujoco_joint_names = []
        self.mujoco_to_motion_index = np.zeros(len(hinge_joints), dtype=np.int64)
        self.joint_axis_motion_space = np.zeros((len(hinge_joints), 3))
        # Per-motion-joint frame correction for the body <quat> mounting offsets.
        self.rot_offsets = np.tile(np.eye(3), (self.n_motion_joints, 1, 1))

        for mujoco_index, joint in enumerate(hinge_joints):
            name = joint.get("name")
            self.mujoco_joint_names.append(name)
            skeleton_name = name.replace("_joint", "_skel")
            if skeleton_name not in motion_index:
                raise KeyError(f"MuJoCo joint {name!r} has no counterpart {skeleton_name!r} in {skeleton.name}")
            joint_motion_index = motion_index[skeleton_name]
            self.mujoco_to_motion_index[mujoco_index] = joint_motion_index

            axis_text = joint.get("axis") or class_axes[joint.get("class")]
            axis = np.array([float(v) for v in axis_text.split()])
            self.joint_axis_motion_space[mujoco_index] = _AXIS_IN_MOTION_SPACE[int(np.argmax(axis))]

            body = parent_of[joint]
            if "quat" in body.attrib:
                quat = [float(v) for v in body.get("quat").split()]
                rotation = Rotation.from_quat(quat, scalar_first=True)
                rotation = (
                    Rotation.from_matrix(MUJOCO_TO_MOTION) * rotation * Rotation.from_matrix(MUJOCO_TO_MOTION).inv()
                )
                self.rot_offsets[joint_motion_index] = rotation.as_matrix().T

        self.n_qpos = 7 + len(hinge_joints)

    def convert(self, local_rot_mats: np.ndarray, root_positions: np.ndarray) -> np.ndarray:
        """Return ``(n_frames, 36)`` qpos from local rotations and root positions."""
        n_frames = local_rot_mats.shape[0]
        local_rot = self.rot_offsets[None] @ local_rot_mats

        qpos = np.zeros((n_frames, self.n_qpos))
        qpos[:, :3] = root_positions @ MOTION_TO_MUJOCO.T

        root_rot = MOTION_TO_MUJOCO @ local_rot[:, 0] @ MUJOCO_TO_MOTION
        # scipy returns xyzw; MuJoCo free joints expect wxyz.
        qpos[:, 3:7] = Rotation.from_matrix(root_rot).as_quat()[:, [3, 0, 1, 2]]

        joint_rot = local_rot[:, self.mujoco_to_motion_index]
        euler_terms = np.stack(
            [
                np.arctan2(joint_rot[..., 2, 1], joint_rot[..., 2, 2]),
                np.arctan2(joint_rot[..., 0, 2], joint_rot[..., 0, 0]),
                np.arctan2(joint_rot[..., 1, 0], joint_rot[..., 1, 1]),
            ],
            axis=-1,
        )
        qpos[:, 7:] = (euler_terms * self.joint_axis_motion_space[None]).sum(axis=-1)
        return qpos


def load_motion(path: Path) -> dict[str, np.ndarray]:
    """Load and validate a motionbricks G1 skeleton NPZ."""
    with np.load(path, allow_pickle=False) as data:
        available = set(data.files)
        if "root_positions" not in available:
            raise KeyError(f"{path} has no 'root_positions'; found {sorted(available)}")
        if "local_rot_mats" not in available and "global_rot_mats" not in available:
            raise KeyError(f"{path} has neither 'local_rot_mats' nor 'global_rot_mats'; found {sorted(available)}")
        motion = {
            key: np.asarray(data[key], dtype=np.float64)
            for key in ("local_rot_mats", "global_rot_mats")
            if key in available
        }
        motion["root_positions"] = np.asarray(data["root_positions"], dtype=np.float64)

    root_positions = motion["root_positions"]
    if root_positions.ndim != 2 or root_positions.shape[1] != 3:
        raise ValueError(f"Expected root_positions (N, 3), got {root_positions.shape}")
    rotations = motion.get("local_rot_mats", motion.get("global_rot_mats"))
    if rotations.ndim != 4 or rotations.shape[2:] != (3, 3):
        raise ValueError(f"Expected rotations (N, J, 3, 3), got {rotations.shape}")
    if rotations.shape[0] != root_positions.shape[0]:
        raise ValueError(
            f"Frame count mismatch: rotations {rotations.shape[0]} vs root_positions {root_positions.shape[0]}"
        )
    for key, array in motion.items():
        if not np.isfinite(array).all():
            raise ValueError(f"{key} contains non-finite values")
    return motion


def local_rotations(motion: dict[str, np.ndarray], parents: np.ndarray) -> np.ndarray:
    """Return per-joint local rotations, deriving them from globals if needed."""
    if "local_rot_mats" in motion:
        return motion["local_rot_mats"]
    global_rot = motion["global_rot_mats"]
    local_rot = np.einsum("fjnm,fjno->fjmo", global_rot[:, parents], global_rot)
    local_rot[:, 0] = global_rot[:, 0]
    return local_rot


# ---------------------------------------------------------------------------
# Resampling, FK, velocities
# ---------------------------------------------------------------------------


def resample_qpos(qpos: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Resample (N, 36) qpos: linear for translation/joints, slerp for the root quat."""
    n_frames = qpos.shape[0]
    if n_frames < 2 or abs(source_fps - target_fps) < 1e-9:
        return qpos.copy()

    t_src = np.arange(n_frames) / source_fps
    n_out = int(np.floor(t_src[-1] * target_fps)) + 1
    t_out = np.arange(n_out) / target_fps

    out = np.zeros((n_out, qpos.shape[1]))
    for col in [0, 1, 2] + list(range(7, qpos.shape[1])):
        out[:, col] = np.interp(t_out, t_src, qpos[:, col])

    # Root quaternion: qpos stores wxyz, scipy wants xyzw.
    slerp = Slerp(t_src, Rotation.from_quat(qpos[:, [4, 5, 6, 3]]))
    out[:, 3:7] = slerp(np.clip(t_out, t_src[0], t_src[-1])).as_quat()[:, [3, 0, 1, 2]]
    return out


def run_fk(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return world positions (N, 14, 3), quaternions wxyz (N, 14, 4), and joint names."""
    model = mujoco.MjModel.from_xml_path(str(FK_XML))
    if model.nq != qpos.shape[1]:
        raise ValueError(f"FK model nq={model.nq}, qpos has {qpos.shape[1]}")
    data = mujoco.MjData(model)

    body_ids = []
    for name in REFERENCE_BODIES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise KeyError(f"Body {name!r} not found in {FK_XML.name}")
        body_ids.append(body_id)

    # Hinge joints in model order == MuJoCo joint order (skip the freejoint).
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]

    n_frames = qpos.shape[0]
    body_pos = np.zeros((n_frames, len(body_ids), 3))
    body_quat = np.zeros((n_frames, len(body_ids), 4))
    for f in range(n_frames):
        data.qpos[:] = qpos[f]
        mujoco.mj_kinematics(model, data)
        body_pos[f] = data.xpos[body_ids]
        body_quat[f] = data.xquat[body_ids]  # wxyz
    return body_pos, body_quat, joint_names


def central_diff(values: np.ndarray, fps: float) -> np.ndarray:
    """Central finite differences, one-sided at the ends (mirrors the C++ reader)."""
    out = np.zeros_like(values)
    for f in range(values.shape[0]):
        f0 = max(f - 1, 0)
        f1 = min(values.shape[0] - 1, f + 1)
        out[f] = (values[f1] - values[f0]) * fps / max(1, f1 - f0)
    return out


def angular_velocities(body_quat: np.ndarray, fps: float) -> np.ndarray:
    """World-frame angular velocity (N, B, 3) from wxyz quaternions."""
    n_frames, n_bodies = body_quat.shape[:2]
    out = np.zeros((n_frames, n_bodies, 3))
    rot = Rotation.from_quat(body_quat.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix()
    rot = rot.reshape(n_frames, n_bodies, 3, 3)
    for f in range(n_frames):
        f0 = max(f - 1, 0)
        f1 = min(n_frames - 1, f + 1)
        dt = (f1 - f0) / fps
        if dt == 0:
            continue
        # R1 = dR @ R0  ->  dR = R1 @ R0^T ; log(dR)/dt is the world angular velocity
        delta = np.einsum("bij,bkj->bik", rot[f1], rot[f0])
        out[f] = Rotation.from_matrix(delta).as_rotvec() / dt
    return out


def mujoco_to_isaaclab(values_mj: np.ndarray) -> np.ndarray:
    """Reorder (N, 29) joint columns from MuJoCo order to IsaacLab order."""
    values_il = np.empty_like(values_mj)
    values_il[:, MJ_TO_IL] = values_mj
    return values_il


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_csv(path: Path, array_2d: np.ndarray, headers: list[str]) -> None:
    with open(path, "w") as f:
        f.write(",".join(headers) + "\n")
        np.savetxt(f, array_2d, delimiter=",", fmt="%.6f")


def write_reference(
    output_dir: Path,
    name: str,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_lin_vel: np.ndarray,
    body_ang_vel: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_frames, n_bodies = body_pos.shape[:2]

    save_csv(output_dir / "joint_pos.csv", joint_pos, [f"joint_{i}" for i in range(joint_pos.shape[1])])
    save_csv(output_dir / "joint_vel.csv", joint_vel, [f"joint_vel_{i}" for i in range(joint_vel.shape[1])])
    save_csv(
        output_dir / "body_pos.csv",
        body_pos.reshape(n_frames, -1),
        [f"body_{i // 3}_{'xyz'[i % 3]}" for i in range(n_bodies * 3)],
    )
    save_csv(
        output_dir / "body_quat.csv",
        body_quat.reshape(n_frames, -1),
        [f"body_{i // 4}_{'wxyz'[i % 4]}" for i in range(n_bodies * 4)],
    )
    save_csv(
        output_dir / "body_lin_vel.csv",
        body_lin_vel.reshape(n_frames, -1),
        [f"body_{i // 3}_vel_{'xyz'[i % 3]}" for i in range(n_bodies * 3)],
    )
    save_csv(
        output_dir / "body_ang_vel.csv",
        body_ang_vel.reshape(n_frames, -1),
        [f"body_{i // 3}_angvel_{'xyz'[i % 3]}" for i in range(n_bodies * 3)],
    )

    indexes = " ".join(str(i) for i in BODY_PART_INDEXES)
    with open(output_dir / "metadata.txt", "w") as f:
        f.write(f"Metadata for: {name}\n")
        f.write("=" * 30 + "\n\n")
        f.write("Body part indexes:\n")
        f.write(f"[ {indexes}]\n\n")
        f.write(f"Total timesteps: {n_frames}\n\n")
        f.write("Data arrays summary:\n")
        f.write(f"  joint_pos: ({n_frames}, {joint_pos.shape[1]}) (float32)\n")
        f.write(f"  joint_vel: ({n_frames}, {joint_vel.shape[1]}) (float32)\n")
        f.write(f"  body_pos_w: ({n_frames}, {n_bodies}, 3) (float32)\n")
        f.write(f"  body_quat_w: ({n_frames}, {n_bodies}, 4) (float32)\n")
        f.write(f"  body_lin_vel_w: ({n_frames}, {n_bodies}, 3) (float32)\n")
        f.write(f"  body_ang_vel_w: ({n_frames}, {n_bodies}, 3) (float32)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("motion", type=Path, help="motionbricks G1 motion .npz")
    parser.add_argument("--source-fps", type=float, default=30.0, help="Frame rate the NPZ was recorded at")
    parser.add_argument("--name", type=str, default=None, help="Motion name (default: npz stem)")
    parser.add_argument("--real", action="store_true", help="Launch on the real robot instead of simulation")
    parser.add_argument("--convert-only", action="store_true", help="Only build the reference; do not launch deploy")
    args = parser.parse_args()
    if args.source_fps <= 0:
        parser.error("--source-fps must be positive")

    name = args.name or args.motion.stem
    output_base = DEFAULT_OUTPUT_BASE / name

    # 1. NPZ -> qpos.
    motion = load_motion(args.motion.resolve())
    skeleton = import_g1_skeleton()
    name_to_index = {n: i for i, (n, _) in enumerate(skeleton.bone_order_names_with_parents)}
    parents = np.array(
        [
            index if parent is None else name_to_index[parent]
            for index, (_, parent) in enumerate(skeleton.bone_order_names_with_parents)
        ]
    )
    converter = G1MotionToQpos(MOTIONBRICKS_XML)
    qpos = converter.convert(local_rotations(motion, parents), motion["root_positions"])
    print(f"{args.motion.name}: {qpos.shape[0]} frames @ {args.source_fps:g} fps")

    # 2. Resample to the 50 Hz deploy playback rate.
    qpos = resample_qpos(qpos, args.source_fps, DEPLOY_FPS)
    print(f"Resampled to {qpos.shape[0]} frames @ {DEPLOY_FPS:g} fps ({qpos.shape[0] / DEPLOY_FPS:.2f} s)")

    # 3. FK for the 14 reference bodies.
    body_pos, body_quat, fk_joint_names = run_fk(qpos)
    if converter.mujoco_joint_names != fk_joint_names:
        raise ValueError(
            "Joint order mismatch between motionbricks g1.xml and g1_29dof_rev_1_0.xml:\n"
            f"  {converter.mujoco_joint_names}\n  {fk_joint_names}"
        )

    # 4. Velocities; joint columns reordered to IsaacLab convention for the CSVs.
    joint_pos = mujoco_to_isaaclab(qpos[:, 7:])
    joint_vel = central_diff(joint_pos, DEPLOY_FPS)
    body_lin_vel = central_diff(body_pos, DEPLOY_FPS)
    body_ang_vel = angular_velocities(body_quat, DEPLOY_FPS)

    # 5. Write the reference folder (base dir contains one subfolder per motion).
    motion_dir = output_base / name
    write_reference(motion_dir, name, joint_pos, joint_vel, body_pos, body_quat, body_lin_vel, body_ang_vel)

    ankle_z = body_pos[:, [3, 6], 2]
    print(f"Reference ready: {motion_dir}")
    print(f"Pelvis height: {body_pos[:, 0, 2].min():.3f} .. {body_pos[:, 0, 2].max():.3f} m")
    print(f"Ankle height:  {ankle_z.min():.3f} .. {ankle_z.max():.3f} m")
    print(f"Peak joint speed: {np.abs(joint_vel).max():.2f} rad/s")

    if args.convert_only:
        return

    # 6. Launch the SONIC deploy on the generated reference.
    deploy_dir = REPO_ROOT / "gear_sonic_deploy"
    mode = "real" if args.real else "sim"
    command = ["./deploy.sh", "--motion-data", f"{output_base}/", mode]
    print(f"\nLaunching: {' '.join(command)}  (cwd: {deploy_dir})")
    print("Keys once running: ] start control, T play motion, O stop, R emergency stop\n")
    os.chdir(deploy_dir)
    os.execv("./deploy.sh", command)


if __name__ == "__main__":
    main()

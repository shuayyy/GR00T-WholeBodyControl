#!/usr/bin/env python3
"""Kinematically replay a motionbricks G1 motion NPZ in the MuJoCo viewer.

The viewer is driven by ``mj_forward`` only, so the robot is posed exactly as
recorded: no gravity, no contacts, no integration. Nothing in this script ever
calls ``mj_step``.

The expected NPZ is the motionbricks 34-joint G1 skeleton format
(``G1Skeleton34``): ``root_positions`` + ``local_rot_mats`` (or
``global_rot_mats``), y-up / z-forward, converted here to MuJoCo's z-up /
x-forward ``qpos``. The conversion is shared with ``run_wholebody_replay.py``.

Examples
--------
    python gear_sonic_planner/scripts/play_motion_npz.py dataset/wholebody_box.npz
    python gear_sonic_planner/scripts/play_motion_npz.py dataset/wholebody_box.npz --fps 60 --loop
    python gear_sonic_planner/scripts/play_motion_npz.py dataset/wholebody_box.npz --show-skeleton
    python gear_sonic_planner/scripts/play_motion_npz.py dataset/wholebody_box.npz --frame 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_wholebody_replay import (  # noqa: E402
    MOTION_TO_MUJOCO,
    MOTIONBRICKS_XML,
    REPO_ROOT,
    G1MotionToQpos,
    import_g1_skeleton,
    load_motion,
    local_rotations,
)

# The motionbricks g1.xml declares meshdir="../meshes/g1", which does not
# exist; its meshes sit in g1/meshes/ and three of them (the *_rev_1_0 links)
# only ship under decoupled_wbc. Meshes are resolved by searching these
# directories in order.
EXTRA_MESH_DIRS = (
    REPO_ROOT / "decoupled_wbc" / "sim2mujoco" / "resources" / "robots" / "g1" / "meshes",
    REPO_ROOT / "decoupled_wbc" / "control" / "robot_model" / "model_data" / "g1" / "meshes",
)


def load_model(xml_path: Path) -> mujoco.MjModel:
    """Load an MJCF, resolving mesh files across several candidate directories."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    compiler = root.find("compiler")

    search_dirs = []
    if compiler is not None and compiler.get("meshdir"):
        search_dirs.append((xml_path.parent / compiler.get("meshdir")).resolve())
    search_dirs.append(xml_path.parent / "meshes")
    search_dirs.append(xml_path.parent / "assets")
    search_dirs.extend(EXTRA_MESH_DIRS)

    assets = {}
    missing = []
    for mesh in root.findall(".//mesh"):
        file_name = mesh.get("file")
        if file_name is None or file_name in assets:
            continue
        for directory in search_dirs:
            candidate = directory / file_name
            if candidate.is_file():
                assets[file_name] = candidate.read_bytes()
                break
        else:
            missing.append(file_name)
    if missing:
        raise FileNotFoundError(
            f"Could not resolve meshes {missing} for {xml_path}. Searched: "
            + ", ".join(str(d) for d in search_dirs)
        )

    # Assets are keyed by bare filename, so the (broken) meshdir must not prefix them.
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets)


def joint_limits(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    """Return ``(n_joints, 2)`` limits, with unlimited joints as +/-inf."""
    limits = np.zeros((len(joint_names), 2))
    for i, name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or not model.jnt_limited[joint_id]:
            limits[i] = (-np.inf, np.inf)
        else:
            limits[i] = model.jnt_range[joint_id]
    return limits


def report_limit_violations(qpos: np.ndarray, model: mujoco.MjModel, joint_names: list[str]) -> None:
    """Warn about joints the recorded motion drives past their MuJoCo range."""
    limits = joint_limits(model, joint_names)
    angles = qpos[:, 7:]
    overshoot = np.maximum(limits[:, 0] - angles, angles - limits[:, 1])
    worst = np.nanmax(np.where(np.isfinite(overshoot), overshoot, -np.inf), axis=0)
    offenders = [(name, value) for name, value in zip(joint_names, worst) if np.isfinite(value) and value > 1e-3]
    if not offenders:
        return
    print("Warning: motion exceeds MuJoCo joint limits (kinematic replay ignores them):")
    for name, value in sorted(offenders, key=lambda item: -item[1]):
        print(f"  {name:<28} by {value:.4f} rad")


def draw_skeleton(viewer, joints_mujoco: np.ndarray, parents: np.ndarray) -> None:
    """Overlay the raw recorded joint positions as spheres in the viewer scene."""
    scene = viewer.user_scn
    scene.ngeom = 0
    for index, position in enumerate(joints_mujoco):
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.015, 0.0, 0.0]),
            position,
            np.eye(3).flatten(),
            np.array([1.0, 0.35, 0.0, 0.9], dtype=np.float32),
        )
        scene.ngeom += 1
        parent = parents[index]
        if parent == index or scene.ngeom >= scene.maxgeom:
            continue
        connector = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            connector,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).flatten(),
            np.array([1.0, 0.75, 0.2, 0.6], dtype=np.float32),
        )
        mujoco.mjv_connector(
            connector, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006, position, joints_mujoco[parent]
        )
        scene.ngeom += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("motion", type=Path, help="Path to the motion .npz")
    parser.add_argument("--xml", type=Path, default=MOTIONBRICKS_XML, help="G1 MJCF to pose")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback rate")
    parser.add_argument("--loop", action="store_true", help="Repeat until the viewer closes")
    parser.add_argument("--start", type=int, default=0, help="First frame (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="Last frame (exclusive)")
    parser.add_argument("--frame", type=int, default=None, help="Hold a single frame instead of playing back")
    parser.add_argument(
        "--show-skeleton",
        action="store_true",
        help="Overlay the recorded joint positions on the posed robot",
    )
    parser.add_argument("--dry-run", action="store_true", help="Convert and report without opening a viewer")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    return args


def main() -> None:
    args = parse_args()

    motion = load_motion(args.motion.resolve())
    converter = G1MotionToQpos(args.xml.resolve())

    skeleton = import_g1_skeleton()
    name_to_index = {name: i for i, (name, _) in enumerate(skeleton.bone_order_names_with_parents)}
    parents = np.array(
        [
            index if parent is None else name_to_index[parent]
            for index, (_, parent) in enumerate(skeleton.bone_order_names_with_parents)
        ]
    )

    local_rot = local_rotations(motion, parents)
    if local_rot.shape[1] != converter.n_motion_joints:
        raise ValueError(
            f"Expected {converter.n_motion_joints} joints for {skeleton.name}, got {local_rot.shape[1]}"
        )

    qpos = converter.convert(local_rot, motion["root_positions"])

    model = load_model(args.xml.resolve())
    if model.nq != qpos.shape[1]:
        raise ValueError(f"Model expects nq={model.nq}, converter produced {qpos.shape[1]}")
    data = mujoco.MjData(model)

    n_frames = qpos.shape[0]
    if args.frame is not None:
        if not -n_frames <= args.frame < n_frames:
            raise ValueError(f"--frame {args.frame} is out of range for {n_frames} frames")
        frames = np.array([args.frame % n_frames])
    else:
        end = n_frames if args.end is None else args.end
        frames = np.arange(n_frames)[args.start : end]
    if frames.size == 0:
        raise ValueError(f"Selected frame range [{args.start}, {args.end}) is empty for {n_frames} frames")

    print(f"{args.motion.name}: {n_frames} frames, playing {frames.size}")
    print(f"Model: {args.xml} (nq={model.nq})")
    print(f"Root height: {qpos[:, 2].min():.3f} .. {qpos[:, 2].max():.3f} m")
    report_limit_violations(qpos[frames], model, converter.mujoco_joint_names)

    if args.dry_run:
        return

    joints_mujoco = None
    if args.show_skeleton:
        with np.load(args.motion.resolve(), allow_pickle=False) as raw:
            if "posed_joints" not in raw.files:
                raise KeyError("--show-skeleton needs 'posed_joints' in the NPZ")
            joints_mujoco = np.asarray(raw["posed_joints"], dtype=np.float64) @ MOTION_TO_MUJOCO.T

    period = 1.0 / args.fps
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        while viewer.is_running():
            for frame in frames:
                if not viewer.is_running():
                    return
                started = time.perf_counter()
                data.qpos[:] = qpos[frame]
                # Kinematics only: mj_forward never integrates, so no physics runs.
                mujoco.mj_forward(model, data)
                if joints_mujoco is not None:
                    draw_skeleton(viewer, joints_mujoco[frame], parents)
                viewer.sync()
                time.sleep(max(0.0, period - (time.perf_counter() - started)))
            if not args.loop or args.frame is not None:
                break
        while viewer.is_running():
            time.sleep(0.02)


if __name__ == "__main__":
    main()

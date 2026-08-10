"""Kinematic split-trajectory viewer with a static box obstacle.

Visualization only -- no simulation, no planning, no controller.  Loads
``test_traj/original.npz`` (299 frames @ 50 Hz, MuJoCo joint order, see
``test_traj/traj.md``), splits it at the bottom pose (lowest pelvis), and
replays the two halves kinematically:

    traj1 = start pose  -> bottom pose
    traj2 = bottom pose -> start pose (the lift back up)

Press ``T`` in the viewer to play the next trajectory; presses cycle
traj1, traj2, traj1, ...  Between presses the robot holds its last pose.

Static boxes (``BOXES`` below -- edit by hand freely) sit in the swept
volume.  Collision against each is checked at every frame on startup and
the per-box report printed before the viewer opens.

Run (viewer):
    .venv_sim/bin/python gear_sonic_planner/planner/demo/view_split_box.py

Collision report only, no window:
    ... view_split_box.py --check

Re-search box placements (prints the best candidates; edit the constants):
    ... view_split_box.py --sweep
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import NamedTuple

import mujoco
import mujoco.viewer
import numpy as np

# --------------------------------------------------------------------------
# Box placement -- hand-editable.  One entry per box.
#   pos  = center in world coordinates: x forward, y left, z up   [m]
#   size = HALF-extents (MuJoCo convention), so the box measures
#          2*size in each direction                               [m]
# Startup re-verifies whatever is written here and prints the per-box
# collision report.
# --------------------------------------------------------------------------
class Box(NamedTuple):
    pos: tuple[float, float, float]
    size: tuple[float, float, float]
    rgba: tuple[float, float, float, float] = (0.9, 0.15, 0.15, 0.55)


BOXES = [
    Box(pos=(0.4, 0.0, 0.425), size=(0.04, 0.1, 0.04),
        rgba=(0.9, 0.15, 0.15, 0.55)),   # red+
]

PLANNER_DIR = Path(__file__).resolve().parents[1]
SCENE_XML = PLANNER_DIR.parents[0] / "data" / "robots" / "g1" / "scene_29dof_freebase.xml"
TRAJECTORY_NPZ = PLANNER_DIR / "test_traj" / "original.npz"
BOX_GEOM_NAME = "demo_box"


def expand_short_joint_name(short: str) -> str:
    """Mocap short name -> model joint name (``L_sh_pitch`` -> ``left_shoulder_pitch_joint``)."""
    name = short.replace("L_", "left_").replace("R_", "right_")
    name = name.replace("_sh_", "_shoulder_").replace("_wr_", "_wrist_")
    name = name.replace("ankle_p", "ankle_pitch").replace("ankle_r", "ankle_roll")
    return name + "_joint"


def build_model(boxes=None) -> tuple[mujoco.MjModel, list[int]]:
    """Scene model plus the demo boxes; returns (model, box geom ids).

    Box positions must be compiled in: worldbody geom positions are baked
    into MuJoCo's static broadphase BVH at compile time, so editing
    ``model.geom_pos`` afterwards silently misses contacts.  Every sweep
    candidate therefore gets its own compile.
    """
    if boxes is None:
        boxes = BOXES
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    names = []
    for index, box in enumerate(boxes):
        geom = spec.worldbody.add_geom()
        geom.name = f"{BOX_GEOM_NAME}_{index}"
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.pos = list(box.pos)
        geom.size = list(box.size)
        geom.rgba = list(box.rgba)
        names.append(geom.name)
    model = spec.compile()
    box_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in names
    ]
    return model, box_ids


def load_trajectory(model: mujoco.MjModel) -> tuple[np.ndarray, float]:
    """original.npz -> (N, nq) qpos array, mapped to model joints BY NAME.

    The NPZ is MuJoCo joint order (traj.md) with short mocap names; each
    column is matched to the model joint it names -- positional slicing is
    never trusted.
    """
    with np.load(TRAJECTORY_NPZ, allow_pickle=False) as data:
        joint_pos = np.asarray(data["joint_pos"], dtype=float)
        base_pos = np.asarray(data["base_pos"], dtype=float)
        base_quat = np.asarray(data["base_quat"], dtype=float)  # wxyz
        names = [str(n) for n in data["joint_names"].tolist()]
        fps = float(data["fps"])

    n_frames = joint_pos.shape[0]
    qpos = np.tile(np.array(model.qpos0, dtype=float), (n_frames, 1))
    # Free joint: [x y z, qw qx qy qz] -- both sides are wxyz.
    qpos[:, 0:3] = base_pos
    qpos[:, 3:7] = base_quat
    for column, short in enumerate(names):
        joint = expand_short_joint_name(short)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if joint_id < 0:
            raise ValueError(f"NPZ joint '{short}' -> '{joint}' not in the model")
        qpos[:, model.jnt_qposadr[joint_id]] = joint_pos[:, column]
    return qpos, fps


def find_bottom_frame(qpos: np.ndarray) -> int:
    """Frame with the lowest pelvis (minimum base z) -- the bottom pose."""
    return int(np.argmin(qpos[:, 2]))


_HAND_BODY_KEYS = ("rubber_hand", "wrist")


def hand_contact_profile(model, data, qpos, box_id):
    """Box contacts over all frames.

    Returns (colliding frames, True iff every contact was a hand/wrist
    geom, max hand penetration depth [m], names of contacted bodies).
    """
    frames, bodies = [], set()
    hand_only, penetration = True, 0.0
    for frame in range(qpos.shape[0]):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        hit = False
        for contact in data.contact[: data.ncon]:
            geoms = (contact.geom1, contact.geom2)
            if box_id not in geoms:
                continue
            other = geoms[1] if geoms[0] == box_id else geoms[0]
            body = model.geom_bodyid[other]
            # Robot geoms live on non-world bodies; the floor (and the box
            # itself) are on the world body.
            if body == 0:
                continue
            hit = True
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
            bodies.add(name)
            if any(key in name for key in _HAND_BODY_KEYS):
                penetration = max(penetration, -min(0.0, float(contact.dist)))
            else:
                hand_only = False
        if hit:
            frames.append(frame)
    return frames, hand_only, penetration, bodies


def as_ranges(frames: list[int]) -> str:
    if not frames:
        return "none"
    ranges, lo, prev = [], frames[0], frames[0]
    for f in frames[1:]:
        if f != prev + 1:
            ranges.append((lo, prev))
            lo = f
        prev = f
    ranges.append((lo, prev))
    return ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in ranges)


def verify_boxes(model, data, qpos, box_ids, bottom: int) -> bool:
    """Print the per-box collision report; True iff endpoints are clear."""
    last = qpos.shape[0] - 1
    all_clear = True
    for box, box_id in zip(BOXES, box_ids):
        colliding, hand_only, penetration, bodies = hand_contact_profile(
            model, data, qpos, box_id
        )
        traj1 = [f for f in colliding if f <= bottom]
        traj2 = [f for f in colliding if f >= bottom]
        print(f"box pos {box.pos}, half-extents {box.size} "
              f"(full size {tuple(round(2 * s, 3) for s in box.size)}):")
        print(f"  colliding frames (traj1, 0->{bottom}): {as_ranges(traj1)}")
        print(f"  colliding frames (traj2, {bottom}->{last}): "
              f"{as_ranges(traj2)}")
        print(f"  contacted bodies: {sorted(bodies) if bodies else 'none'} "
              f"(hand-only: {hand_only}, max hand penetration "
              f"{penetration * 1000:.1f} mm)")
        endpoints_clear = not {0, bottom} & set(colliding)
        print(f"  frame 0 / bottom {bottom} / final {last} collision-free: "
              f"{0 not in colliding} / {bottom not in colliding} / "
              f"{last not in colliding}")
        if not colliding:
            print("  WARNING: this box never collides")
        if not endpoints_clear:
            print("  WARNING: this box collides at an endpoint pose -- "
                  "edit BOXES")
        all_clear = all_clear and endpoints_clear
    return all_clear


def sweep(qpos, bottom: int) -> None:
    """Grid-search box centers; print placements that satisfy the demo.

    Recompiles the model per candidate (see build_model) and reports only
    placements whose contacts are hand/wrist geoms -- the demo wants the
    box to graze the hands, nothing else.
    """
    last = qpos.shape[0] - 1
    size = BOXES[0].size
    print(f"sweeping a single box (y = 0, half-extents {size}, "
          f"recompiling per candidate) ...")
    results = []
    for x in np.round(np.arange(0.30, 0.70, 0.02), 2):
        for z in np.round(np.arange(0.30, 0.90, 0.05), 2):
            model, (box_id,) = build_model(
                [Box(pos=(float(x), 0.0, float(z)), size=size)]
            )
            data = mujoco.MjData(model)
            colliding, hand_only, penetration, _ = hand_contact_profile(
                model, data, qpos, box_id
            )
            if not colliding or not hand_only:
                continue
            if {0, bottom, last} & set(colliding):
                continue
            margin = min(
                min(f, abs(f - bottom), last - f) for f in colliding
            )
            results.append(
                (margin, len(colliding), float(x), float(z), colliding,
                 penetration)
            )
    results.sort(reverse=True)
    if not results:
        print("no placement found: every candidate misses the motion, "
              "collides at an endpoint, or touches more than the hands")
        return
    print("top hand-only candidates (frame margin, #frames, max pen):")
    for margin, count, x, z, colliding, pen in results[:10]:
        print(f"  pos ({x:.2f}, 0.00, {z:.2f})  margin {margin:>3}  "
              f"frames {count:>3}  pen {pen * 1000:4.1f} mm  "
              f"({as_ranges(colliding)})")


def play(viewer, model, data, qpos_segment, fps: float) -> None:
    period = 1.0 / fps
    for state in qpos_segment:
        if not viewer.is_running():
            return
        started = time.monotonic()
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(max(0.0, period - (time.monotonic() - started)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="print the collision report and exit (no viewer)")
    parser.add_argument("--sweep", action="store_true",
                        help="grid-search box placements and exit (no viewer)")
    args = parser.parse_args()

    model, box_ids = build_model()
    data = mujoco.MjData(model)
    qpos, fps = load_trajectory(model)
    bottom = find_bottom_frame(qpos)
    print(f"bottom pose: frame {bottom} (criterion: minimum pelvis height, "
          f"{qpos[bottom, 2]:.3f} m vs {qpos[0, 2]:.3f} m at frame 0)")

    if args.sweep:
        sweep(qpos, bottom)
        return
    verify_boxes(model, data, qpos, box_ids, bottom)
    if args.check:
        return

    segments = [
        (f"traj1 (start -> bottom, frames 0-{bottom})", qpos[: bottom + 1]),
        (f"traj2 (bottom -> start, frames {bottom}-{qpos.shape[0] - 1})",
         qpos[bottom:]),
    ]
    state = {"play": False}

    def key_callback(keycode):
        if keycode in (ord("T"), ord("t")):
            state["play"] = True

    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model, data)
    next_segment = 0
    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        viewer.sync()
        print("viewer ready -- press T to play the next trajectory")
        while viewer.is_running():
            if state["play"]:
                state["play"] = False
                label, segment = segments[next_segment]
                print(f"playing {label}")
                play(viewer, model, data, segment, fps)
                next_segment = (next_segment + 1) % len(segments)
                print("holding -- press T for the next trajectory")
            time.sleep(0.02)


if __name__ == "__main__":
    main()

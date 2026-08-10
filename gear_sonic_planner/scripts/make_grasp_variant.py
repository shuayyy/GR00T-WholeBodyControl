#!/usr/bin/env python3
"""Build a bimanual-grasp variant of a SONIC deploy reference.

Takes an existing reference folder and re-solves the two arms so that, for the
whole motion:

  * both end-effectors sit at the **same height** (z_L == z_R);
  * the two hands are **opposite-parallel** — palms facing each other, i.e. both
    wrists share the target rotation [forward | u | forward x u], which puts the
    left palm normal along -u and the right along +u;
  * the **EE-to-EE** surface clearance is 10 cm (a 10 cm object between them);
  * the **EE-to-body** surface clearance is at least 10 cm;
  * the fingers are posed 90 % closed for all clearance checks.

Legs, waist and the root trajectory are copied through untouched; only the 14
arm joints are re-solved. Nothing is deleted — output goes to a new folder.

Method
------
Per frame: build the target hand poses from the pelvis frame, then solve the 7
joints of each arm with least-squares IK (warm-started from the previous frame,
regularised toward the input pose). An outer loop calibrates the wrist
separation so the measured geom clearance lands on 10 cm, and pushes the hand
pair forward along the body's facing direction until the EE-to-body clearance
target is met.

Usage
-----
    python gear_sonic_planner/scripts/make_grasp_variant.py
    python gear_sonic_planner/scripts/make_grasp_variant.py --name my_variant
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_wholebody_replay import (  # noqa: E402
    DEPLOY_FPS,
    MJ_TO_IL,
    REPO_ROOT,
    angular_velocities,
    central_diff,
    mujoco_to_isaaclab,
    write_reference,
)

SCENE = REPO_ROOT / "gear_sonic" / "data" / "robot_model" / "model_data" / "g1" / "scene_43dof.xml"
MJ29 = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
LEFT_ARM = list(range(15, 22))   # indices into MJ29
RIGHT_ARM = list(range(22, 29))
REFERENCE_BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]
FINGER_CLOSE_RATIO = 0.90


def load_csv(path: Path) -> np.ndarray:
    with open(path) as f:
        rows = list(csv.reader(f))
    return np.array(rows[1:], dtype=float)


class Rig:
    """MuJoCo wrapper holding the pose and answering FK / clearance queries."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in MJ29]

        def bname(g):
            return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""

        self.bname = bname
        collidable = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        self.ee_geoms = [g for g in collidable if "hand" in bname(g) or "wrist" in bname(g)]
        self.left_ee = [g for g in self.ee_geoms if bname(g).startswith("left")]
        self.right_ee = [g for g in self.ee_geoms if bname(g).startswith("right")]
        self.floor = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
        self.body_geoms = [
            g for g in collidable
            if g not in self.ee_geoms and g not in self.floor and "elbow" not in bname(g)
        ]
        # Finger joints, posed at FINGER_CLOSE_RATIO of their closing range.
        self.finger_j = [
            j for j in range(m.njnt)
            if "hand" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "")
        ]
        self.finger_q = {}
        for j in self.finger_j:
            lo, hi = m.jnt_range[j]
            # "closed" is whichever limit has the larger magnitude flexion.
            closed = hi if abs(hi) > abs(lo) else lo
            self.finger_q[m.jnt_qposadr[j]] = closed * FINGER_CLOSE_RATIO
        self.wrist_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
        self.wrist_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        self.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.body_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in REFERENCE_BODIES]

    def set_pose(self, root_pos, root_quat, q29):
        d = self.data
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0
        d.qpos[0:3] = root_pos
        d.qpos[3:7] = root_quat
        for k, a in enumerate(self.adr):
            d.qpos[a] = q29[k]
        for a, v in self.finger_q.items():
            d.qpos[a] = v

    def kinematics(self):
        mujoco.mj_kinematics(self.model, self.data)

    def wrist_pose(self, left: bool):
        b = self.wrist_l if left else self.wrist_r
        return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    def clearances(self, margin=0.30):
        """Return (min EE-to-body gap, min EE-to-EE gap) using contact margins."""
        m = self.model
        saved = m.geom_margin.copy()
        m.geom_margin[:] = 0.0
        m.geom_margin[self.ee_geoms] = margin
        mujoco.mj_forward(m, self.data)
        ee_body, ee_ee = 9e9, 9e9
        ee_set = set(self.ee_geoms)
        body_set = set(self.body_geoms)
        for c in range(self.data.ncon):
            g1, g2 = self.data.contact[c].geom1, self.data.contact[c].geom2
            if g1 in self.floor or g2 in self.floor:
                continue
            gap = self.data.contact[c].dist
            in1, in2 = g1 in ee_set, g2 in ee_set
            if in1 and in2:
                if self.bname(g1).startswith("left") != self.bname(g2).startswith("left"):
                    ee_ee = min(ee_ee, gap)
            elif (in1 and g2 in body_set) or (in2 and g1 in body_set):
                ee_body = min(ee_body, gap)
        m.geom_margin[:] = saved
        return ee_body, ee_ee


def arm_residual(rig, q29, arm_idx, x, target_pos, target_rot, q_ref, w_reg):
    q = q29.copy()
    q[arm_idx] = x
    rig.set_pose(rig._root_pos, rig._root_quat, q)
    rig.kinematics()
    pos, rot = rig.wrist_pose(left=(arm_idx is LEFT_ARM))
    r_err = Rotation.from_matrix(target_rot.T @ rot).as_rotvec()
    return np.concatenate([(pos - target_pos) * 10.0, r_err * 2.0, (x - q_ref) * w_reg])


def solve_frame(rig, q29, root_pos, root_quat, tgt_l, tgt_r, rot_t, warm):
    rig._root_pos, rig._root_quat = root_pos, root_quat
    out = q29.copy()
    for arm_idx, tgt, w in ((LEFT_ARM, tgt_l, warm[0]), (RIGHT_ARM, tgt_r, warm[1])):
        res = least_squares(
            lambda x, a=arm_idx, t=tgt: arm_residual(rig, out, a, x, t, rot_t, q29[a], 0.15),
            w, method="lm", max_nfev=180, xtol=1e-8, ftol=1e-8,
        )
        out[arm_idx] = res.x
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path,
                   default=REPO_ROOT / "gear_sonic_planner/reference/wholebody_box_fixed/wholebody_box_fixed")
    p.add_argument("--name", type=str, default="wholebody_box_grasp")
    p.add_argument("--ee-ee", type=float, default=0.10, help="target EE-to-EE clearance (m)")
    p.add_argument("--ee-body", type=float, default=0.10, help="minimum EE-to-body clearance (m)")
    args = p.parse_args()

    src = args.source
    jp_in = load_csv(src / "joint_pos.csv")[:, MJ_TO_IL]          # (N,29) MuJoCo order
    bp_in = load_csv(src / "body_pos.csv").reshape(-1, 14, 3)
    bq_in = load_csv(src / "body_quat.csv").reshape(-1, 14, 4)
    N = len(jp_in)
    rig = Rig()
    print(f"source: {src.name}  ({N} frames)")
    print(f"targets: EE-EE = {args.ee_ee*100:.0f} cm, EE-body >= {args.ee_body*100:.0f} cm, "
          f"fingers {FINGER_CLOSE_RATIO*100:.0f}% closed")

    # Pelvis frame per frame -> forward f and lateral u (horizontal, orthonormal).
    fwd = np.zeros((N, 3))
    lat = np.zeros((N, 3))
    for f in range(N):
        R = Rotation.from_quat(bq_in[f, 0][[1, 2, 3, 0]]).as_matrix()
        x = R[:, 0].copy(); x[2] = 0.0; x /= np.linalg.norm(x)
        y = np.cross([0, 0, 1.0], x)
        fwd[f], lat[f] = x, y

    mid_in = 0.5 * (bp_in[:, 10] + bp_in[:, 13])   # midpoint of the two wrists

    sep, push = 0.22, 0.06                         # wrist separation, forward offset
    q_out = jp_in.copy()
    for it in range(6):
        warm = [jp_in[0][LEFT_ARM].copy(), jp_in[0][RIGHT_ARM].copy()]
        ee_ee_all, ee_body_all = [], []
        for f in range(N):
            M = mid_in[f] + push * fwd[f]
            M = np.array([M[0], M[1], mid_in[f][2]])              # level: shared z
            tgt_l = M + 0.5 * sep * lat[f]
            tgt_r = M - 0.5 * sep * lat[f]
            rot_t = np.column_stack([fwd[f], lat[f], np.cross(fwd[f], lat[f])])
            q_out[f] = solve_frame(rig, jp_in[f], bp_in[f, 0], bq_in[f, 0], tgt_l, tgt_r, rot_t, warm)
            warm = [q_out[f][LEFT_ARM].copy(), q_out[f][RIGHT_ARM].copy()]
            rig.set_pose(bp_in[f, 0], bq_in[f, 0], q_out[f])
            eb, ee = rig.clearances()
            ee_body_all.append(eb); ee_ee_all.append(ee)
        ee_ee_all = np.array(ee_ee_all); ee_body_all = np.array(ee_body_all)
        med_ee = np.median(ee_ee_all[ee_ee_all < 9e8])
        min_body = ee_body_all[ee_body_all < 9e8].min() if (ee_body_all < 9e8).any() else 9e9
        print(f"  iter {it}: sep={sep*100:5.1f}cm push={push*100:5.1f}cm -> "
              f"EE-EE median {med_ee*100:5.1f}cm (min {ee_ee_all.min()*100:.1f}), "
              f"EE-body min {min_body*100:5.1f}cm")
        ok_ee = abs(med_ee - args.ee_ee) < 0.01
        ok_body = min_body >= args.ee_body - 0.005
        if ok_ee and ok_body:
            break
        if not ok_ee:
            sep += (args.ee_ee - med_ee)
        if not ok_body:
            push += min(0.06, max(0.01, args.ee_body - min_body))

    # Outputs -------------------------------------------------------------
    body_pos = np.zeros((N, 14, 3)); body_quat = np.zeros((N, 14, 4))
    for f in range(N):
        rig.set_pose(bp_in[f, 0], bq_in[f, 0], q_out[f]); rig.kinematics()
        body_pos[f] = rig.data.xpos[rig.body_ids]
        body_quat[f] = rig.data.xquat[rig.body_ids]
    joint_pos = mujoco_to_isaaclab(q_out)
    joint_vel = central_diff(joint_pos, DEPLOY_FPS)
    out_dir = REPO_ROOT / "gear_sonic_planner" / "reference" / args.name / args.name
    write_reference(out_dir, args.name, joint_pos, joint_vel, body_pos, body_quat,
                    central_diff(body_pos, DEPLOY_FPS), angular_velocities(body_quat, DEPLOY_FPS))

    dz = body_pos[:, 10, 2] - body_pos[:, 13, 2]
    dq = q_out - jp_in
    print(f"\nwrote {out_dir}")
    print(f"  EE z difference: max {np.abs(dz).max()*1000:.2f} mm")
    print(f"  arm joints changed: {np.count_nonzero(np.abs(dq).max(0) > 1e-4)}/29, "
          f"max delta {np.degrees(np.abs(dq).max()):.1f} deg")
    np.savez(REPO_ROOT / "test_dataset" / f"reference_trajectory_{args.name}.npz",
             joint_pos=q_out.astype(np.float32), base_pos=bp_in[:, 0].astype(np.float32),
             base_quat=bq_in[:, 0].astype(np.float32), joint_names=np.array(MJ29),
             fps=DEPLOY_FPS, joint_order="mujoco", finger_close_ratio=FINGER_CLOSE_RATIO)
    print(f"  wrote test_dataset/reference_trajectory_{args.name}.npz")


if __name__ == "__main__":
    main()

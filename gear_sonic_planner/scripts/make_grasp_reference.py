#!/usr/bin/env python3
"""Retarget the collision-free reference into a symmetric two-hand grasp pose.

Starts from ``wholebody_box_fixed`` and enforces, on every frame:

* both end-effectors at the **same z**;
* the two hands **mirror-symmetric** about the body's sagittal plane, so their
  palm normals are anti-parallel ("opposite parallel");
* **EE -> body** clearance >= ``--body-clearance`` (default 100 mm);
* **EE -> EE** clearance >= ``--ee-clearance`` (default 100 mm);
* fingers held at **90 % closure** (0.9 x the deploy's full-close pose).

Only the 14 arm joints move. Legs and waist are copied through untouched.

Method: mirror-average the two wrist poses in the pelvis frame to get one
symmetric target pair, offset it laterally / forward by a searched amount, then
solve damped-least-squares IK on each 7-DOF arm chain. Clearances are measured
in ``scene_43dof.xml`` with the fingers at 90 % closure, and the offsets are
increased until both thresholds hold.

Nothing existing is modified: output goes to a new reference folder and a new
NPZ alongside the originals.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_wholebody_replay import (  # noqa: E402
    DEPLOY_FPS,
    MJ_TO_IL,
    angular_velocities,
    central_diff,
    mujoco_to_isaaclab,
    write_reference,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE = REPO_ROOT / "gear_sonic" / "data" / "robot_model" / "model_data" / "g1" / "scene_43dof.xml"
SRC = REPO_ROOT / "gear_sonic_planner" / "reference" / "wholebody_box_fixed" / "wholebody_box_fixed"

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

# 90 % of the deploy's hardcoded full-close pose (input_interface.hpp GetHandPose).
FULL_CLOSE_LEFT = np.array([0.0, 0.0, 1.75, -1.57, -1.75, -1.57, -1.75])
FINGER_ORDER = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]

MIRROR = np.diag([1.0, -1.0, 1.0])


def load_csv(path: Path) -> np.ndarray:
    with open(path) as f:
        rows = list(csv.reader(f))
    return np.array(rows[1:], dtype=float)


def quat_to_mat(q_wxyz: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(q_wxyz[[1, 2, 3, 0]]).as_matrix()


class Rig:
    """Model handles plus finger posing and clearance measurement."""

    def __init__(self, closure: float):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.jid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in MJ29}
        self.qadr = np.array([m.jnt_qposadr[self.jid[n]] for n in MJ29])
        self.dadr = np.array([m.jnt_dofadr[self.jid[n]] for n in MJ29])
        self.lim = np.array([m.jnt_range[self.jid[n]] for n in MJ29])
        self.wrist = {
            "L": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"),
            "R": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"),
        }
        # Fingers at the requested closure fraction.
        self.finger_q = {}
        for side, sign in (("left", 1.0), ("right", -1.0)):
            for name, full in zip(FINGER_ORDER, FULL_CLOSE_LEFT):
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hand_{name}_joint")
                if j < 0:
                    continue
                target = closure * full * (1.0 if side == "left" else sign)
                lo, hi = m.jnt_range[j]
                self.finger_q[m.jnt_qposadr[j]] = float(np.clip(target, lo, hi))

        def bname(g):
            return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""

        self.bname = bname
        collidable = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        self.ee = [g for g in collidable if "hand" in bname(g) or "wrist" in bname(g)]
        self.floor = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
        # "Body" for clearance purposes = trunk + legs + head. The arm's own
        # elbow/shoulder are excluded: elbow<->wrist_pitch sits at a fixed 9.5 mm
        # by construction and no joint motion can open it.
        TRUNK_LEG = ("pelvis", "waist", "torso", "head", "hip", "knee", "ankle", "foot")
        self.trunk_leg = {g for g in collidable if any(k in bname(g) for k in TRUNK_LEG)}
        # Detection halo on the hands only, so contact counts stay bounded.
        m.geom_margin[:] = 0.0
        m.geom_margin[self.ee] = 0.28

    def pose(self, q29: np.ndarray, root_pos: np.ndarray, root_quat: np.ndarray) -> None:
        d = self.d
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0
        d.qpos[0:3] = root_pos
        d.qpos[3:7] = root_quat
        d.qpos[self.qadr] = q29
        for adr, val in self.finger_q.items():
            d.qpos[adr] = val

    def clearances(self, q29, root_pos, root_quat):
        """Return (min EE-body gap, min EE-EE gap) in metres; +inf if out of range."""
        self.pose(q29, root_pos, root_quat)
        mujoco.mj_forward(self.m, self.d)
        d, ee = self.d, set(self.ee)
        body_gap, ee_gap = np.inf, np.inf
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if g1 in self.floor or g2 in self.floor:
                continue
            a, b = g1 in ee, g2 in ee
            gap = d.contact[c].dist
            if a and b:
                if self.bname(g1).startswith("left") != self.bname(g2).startswith("left"):
                    ee_gap = min(ee_gap, gap)
            elif (a and g2 in self.trunk_leg) or (b and g1 in self.trunk_leg):
                body_gap = min(body_gap, gap)
        return body_gap, ee_gap


def symmetric_targets(rig: Rig, q29: np.ndarray, bp0: np.ndarray, bq0: np.ndarray):
    """Mirror-average the two wrist poses; return targets in the pelvis frame."""
    rig.pose(q29, bp0, bq0)
    mujoco.mj_kinematics(rig.m, rig.d)
    R_pel = quat_to_mat(bq0)
    out = {}
    for side in ("L", "R"):
        b = rig.wrist[side]
        out[side] = (
            R_pel.T @ (rig.d.xpos[b] - bp0),
            R_pel.T @ rig.d.xmat[b].reshape(3, 3),
        )
    pL, RL = out["L"]
    pR, RR = out["R"]
    p_sym = 0.5 * (pR + MIRROR @ pL)                     # right-side representative
    R_sym = Rotation.from_matrix(np.stack([RR, MIRROR @ RL @ MIRROR])).mean().as_matrix()
    return p_sym, R_sym, R_pel


def ik_arm(rig: Rig, arm_idx, body_id, q29, bp0, bq0, p_target_w, R_target_w, iters=200):
    """Damped least-squares IK on one 7-DOF arm chain. Modifies q29 in place."""
    m, d = rig.m, rig.d
    dofs = rig.dadr[arm_idx]
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    for _ in range(iters):
        rig.pose(q29, bp0, bq0)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p = d.xpos[body_id]
        R = d.xmat[body_id].reshape(3, 3)
        err = np.concatenate([p_target_w - p, Rotation.from_matrix(R_target_w @ R.T).as_rotvec()])
        if np.linalg.norm(err) < 1e-4:
            break
        mujoco.mj_jacBody(m, d, jacp, jacr, body_id)
        J = np.vstack([jacp[:, dofs], jacr[:, dofs]])
        lam = 1e-2
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), err)
        dq = np.clip(dq, -0.2, 0.2)
        q29[arm_idx] = np.clip(q29[arm_idx] + dq, rig.lim[arm_idx, 0], rig.lim[arm_idx, 1])
    return q29


def solve(rig: Rig, jp_src, bp, bq, y_half, fwd, up):
    """Build the symmetric trajectory.

    ``y_half``, ``fwd`` and ``up`` are per-frame offsets (m); scalars are
    broadcast.
    """
    n = len(jp_src)
    y_half = np.broadcast_to(np.asarray(y_half, dtype=float), (n,))
    q_out = jp_src.copy()
    for f in range(n):
        q = q_out[f]
        if f > 0:
            q[LEFT_ARM] = q_out[f - 1][LEFT_ARM]     # warm start for continuity
            q[RIGHT_ARM] = q_out[f - 1][RIGHT_ARM]
        p_sym, R_sym, R_pel = symmetric_targets(rig, jp_src[f], bp[f, 0], bq[f, 0])
        p_sym = p_sym.copy()
        p_sym[0] += fwd[f]
        p_sym[1] = -abs(y_half[f])
        p_sym[2] += up[f]
        for side, arm, mirror in (("R", RIGHT_ARM, False), ("L", LEFT_ARM, True)):
            p_pel = MIRROR @ p_sym if mirror else p_sym
            R_pel_t = MIRROR @ R_sym @ MIRROR if mirror else R_sym
            ik_arm(
                rig, arm, rig.wrist[side], q, bp[f, 0], bq[f, 0],
                bp[f, 0] + R_pel @ p_pel, R_pel @ R_pel_t,
            )
        q_out[f] = q
    return q_out


def per_frame_gaps(rig: Rig, q_all, bp, bq):
    body = np.zeros(len(q_all))
    ee = np.zeros(len(q_all))
    for f in range(len(q_all)):
        b, e = rig.clearances(q_all[f], bp[f, 0], bq[f, 0])
        body[f] = 10.0 if not np.isfinite(b) else b
        ee[f] = 10.0 if not np.isfinite(e) else e
    return body, ee


def measure(rig: Rig, q_all, bp, bq):
    body = np.inf
    ee = np.inf
    argb = arge = -1
    for f in range(len(q_all)):
        b, e = rig.clearances(q_all[f], bp[f, 0], bq[f, 0])
        if b < body:
            body, argb = b, f
        if e < ee:
            ee, arge = e, f
    return body, ee, argb, arge


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body-clearance", type=float, default=0.10, help="min EE->body gap (m)")
    ap.add_argument("--ee-clearance", type=float, default=0.10, help="min EE->EE gap (m)")
    ap.add_argument("--ee-max", type=float, default=0.21, help="max EE->EE gap (m)")
    ap.add_argument("--closure", type=float, default=0.9, help="finger closure fraction")
    ap.add_argument("--name", type=str, default="wholebody_box_grasp")
    args = ap.parse_args()

    jp_src = load_csv(SRC / "joint_pos.csv")[:, MJ_TO_IL]      # -> MuJoCo order
    bp = load_csv(SRC / "body_pos.csv").reshape(-1, 14, 3)
    bq = load_csv(SRC / "body_quat.csv").reshape(-1, 14, 4)
    rig = Rig(args.closure)
    print(f"source: {len(jp_src)} frames; fingers at {args.closure * 100:.0f}% closure")

    from scipy.ndimage import gaussian_filter1d, maximum_filter1d

    n = len(jp_src)
    # Per-frame knobs: y_half sets the hand-to-hand gap (band-limited between
    # --ee-clearance and --ee-max), fwd pushes the pair away from trunk+legs.
    y_half = np.full(n, 0.18)
    fwd = np.full(n, 0.17)
    up = np.zeros(n)
    q = None
    for it in range(14):
        q = solve(rig, jp_src, bp, bq, y_half, fwd, up)
        body, ee = per_frame_gaps(rig, q, bp, bq)
        print(f"  iter {it:2d}: EE-body min {body.min() * 1000:6.1f} | EE-EE {ee.min() * 1000:6.1f}"
              f"..{ee.max() * 1000:6.1f} mm | y_half {y_half.min():.3f}..{y_half.max():.3f}"
              f" | fwd max {fwd.max():.3f}")
        ok = (body.min() >= args.body_clearance
              and ee.min() >= args.ee_clearance
              and ee.max() <= args.ee_max)
        if ok:
            break
        # Body clearance: push the pair forward where it is short.
        need_body = np.maximum(args.body_clearance + 0.004 - body, 0.0)
        if need_body.max() > 0:
            fwd = gaussian_filter1d(maximum_filter1d(fwd + 0.9 * need_body, 11), 4)
        # EE-EE band: widen where too close, narrow where too far apart.
        too_close = np.maximum(args.ee_clearance + 0.004 - ee, 0.0)
        too_far = np.maximum(ee - (args.ee_max - 0.004), 0.0)
        y_half = y_half + 0.45 * too_close - 0.45 * too_far
        y_half = gaussian_filter1d(np.clip(y_half, 0.05, 0.30), 4)

    body, ee, fb, fe = measure(rig, q, bp, bq)
    _, ee_series = per_frame_gaps(rig, q, bp, bq)
    print(f"\ny_half {y_half.min():.3f}..{y_half.max():.3f} m, forward push {fwd.min():.3f}..{fwd.max():.3f} m")
    print(f"EE->body min = {body * 1000:.1f} mm at frame {fb}   (target >= {args.body_clearance * 1000:.0f})")
    print(f"EE->EE   min = {ee * 1000:.1f} mm at frame {fe}   (target >= {args.ee_clearance * 1000:.0f})")
    print(f"EE->EE   max = {ee_series.max() * 1000:.1f} mm   (target <= {args.ee_max * 1000:.0f})")

    # Symmetry report.
    # A mirrored pair satisfies p_L = M p_R and R_L = M R_R M (in the pelvis
    # frame). The left hand's own frame is the mirror of the right's, so its
    # local +y is flipped -- comparing raw palm normals would be meaningless.
    zL, zR, mirr_p, mirr_R = [], [], [], []
    for f in range(len(q)):
        rig.pose(q[f], bp[f, 0], bq[f, 0])
        mujoco.mj_kinematics(rig.m, rig.d)
        R_pel = quat_to_mat(bq[f, 0])
        pL = R_pel.T @ (rig.d.xpos[rig.wrist["L"]] - bp[f, 0])
        pR = R_pel.T @ (rig.d.xpos[rig.wrist["R"]] - bp[f, 0])
        RL = R_pel.T @ rig.d.xmat[rig.wrist["L"]].reshape(3, 3)
        RR = R_pel.T @ rig.d.xmat[rig.wrist["R"]].reshape(3, 3)
        zL.append(rig.d.xpos[rig.wrist["L"]][2])
        zR.append(rig.d.xpos[rig.wrist["R"]][2])
        mirr_p.append(np.linalg.norm(pL - MIRROR @ pR))
        mirr_R.append(np.degrees(np.linalg.norm(
            Rotation.from_matrix((MIRROR @ RR @ MIRROR) @ RL.T).as_rotvec())))
    dz = np.abs(np.array(zL) - np.array(zR))
    mirr_p = np.array(mirr_p)
    mirr_R = np.array(mirr_R)
    print(f"EE z difference:      max {dz.max() * 1000:6.2f} mm, mean {dz.mean() * 1000:.2f} mm")
    print(f"mirror pos residual:  max {mirr_p.max() * 1000:6.2f} mm  (0 = perfectly mirrored)")
    print(f"mirror rot residual:  max {mirr_R.max():6.2f} deg  (0 = palms exactly opposing)")
    dq = q - jp_src
    print(f"joints changed: {np.count_nonzero(np.abs(dq).max(0) > 1e-4)}/29, "
          f"max delta {np.abs(dq).max():.3f} rad")

    # Write outputs (new files only)
    qpos = np.zeros((len(q), 36))
    qpos[:, 0:3] = bp[:, 0]
    qpos[:, 3:7] = bq[:, 0]
    qpos[:, 7:] = q
    body_pos = np.zeros((len(q), 14, 3))
    body_quat = np.zeros((len(q), 14, 4))
    ref_bodies = ["pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
                  "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
                  "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
                  "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link"]
    bids = [mujoco.mj_name2id(rig.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ref_bodies]
    for f in range(len(q)):
        rig.pose(q[f], bp[f, 0], bq[f, 0])
        mujoco.mj_kinematics(rig.m, rig.d)
        body_pos[f] = rig.d.xpos[bids]
        body_quat[f] = rig.d.xquat[bids]

    joint_pos = mujoco_to_isaaclab(q)
    out = REPO_ROOT / "gear_sonic_planner" / "reference" / args.name / args.name
    write_reference(out, args.name, joint_pos, central_diff(joint_pos, DEPLOY_FPS), body_pos,
                    body_quat, central_diff(body_pos, DEPLOY_FPS),
                    angular_velocities(body_quat, DEPLOY_FPS))
    print(f"\nwrote {out}")

    fingers = {f"left_hand_{n}_joint": float(args.closure * v) for n, v in zip(FINGER_ORDER, FULL_CLOSE_LEFT)}
    np.savez(
        REPO_ROOT / "test_dataset" / f"reference_trajectory_{args.name.replace('wholebody_box_', '')}.npz",
        joint_pos=q.astype(np.float32), base_pos=bp[:, 0].astype(np.float32),
        base_quat=bq[:, 0].astype(np.float32), joint_names=np.array(MJ29),
        fps=DEPLOY_FPS, joint_order="mujoco",
        finger_closure=args.closure,
        finger_targets_left=np.array(list(fingers.values()), dtype=np.float32),
        ee_body_clearance_m=body, ee_ee_clearance_m=ee,
    )
    print(f"wrote test_dataset/reference_trajectory_{args.name.replace('wholebody_box_', '')}.npz")


if __name__ == "__main__":
    main()

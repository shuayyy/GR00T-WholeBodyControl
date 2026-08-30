"""Self-collision checking against the simulation scene, fingers included.

The planning URDF has no hand joints; leaving the deployed robot's 14 finger
DoF at zero folds them into the hips and reports collisions that do not exist.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.planner.joint_orders import mjcf_joint_names

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Simulation scene the deploy stack runs (wbc_configs/*.yaml ROBOT_SCENE).
#: 43 DoF: free base + 29 body joints + 14 fingers.
DEFAULT_COLLISION_SCENE = (
    _REPO_ROOT / "gear_sonic" / "data" / "robot_model" / "model_data" / "g1" / "scene_43dof.xml"
)

#: The deploy's hardcoded full-close pose for one hand, in FINGER_ORDER
#: (gear_sonic_deploy .../input_interface.hpp, GetHandPose defaults).
FULL_CLOSE_LEFT = np.array([0.0, 0.0, 1.75, -1.57, -1.75, -1.57, -1.75])
FINGER_ORDER = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]

#: Bodies counted as "body" for end-effector clearance: trunk, legs, head.
#: The arm's own shoulder/elbow are excluded -- elbow<->wrist_pitch sits at a
#: fixed ~9.5 mm by construction and no joint motion can open it.
_TRUNK_LEG_KEYWORDS = ("pelvis", "waist", "torso", "head", "hip", "knee", "ankle", "foot")

#: Contact-detection halo on the hands, so clearance queries see near misses
#: without the contact count exploding. Only used when measuring clearances.
_EE_MARGIN = 0.28


class CollisionRig:
    """MuJoCo scene posed from a 29-joint trajectory frame, fingers included."""

    def __init__(
        self,
        scene_xml: str | Path = DEFAULT_COLLISION_SCENE,
        finger_closure: float = 0.9,
        ee_margin: float = 0.0,
    ):
        """``finger_closure`` is the fraction of the deploy's full-close
        pose; ``ee_margin`` is a contact halo, 0 for boolean checks."""
        scene_xml = Path(scene_xml)
        if not scene_xml.exists():
            raise FileNotFoundError(f"Collision scene not found: {scene_xml}")
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)
        model = self.model

        # Body joints, addressed by name (the scene interleaves fingers).
        self.body_joint_names = mjcf_joint_names(scene_xml)
        joint_ids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.body_joint_names
        }
        missing = [n for n, i in joint_ids.items() if i < 0]
        if missing:
            raise ValueError(f"Joints missing from {scene_xml}: {missing}")
        self.qadr = np.array([model.jnt_qposadr[joint_ids[n]] for n in self.body_joint_names])

        # Fingers at the requested closure, mirrored on the right, clipped.
        self.finger_closure = float(finger_closure)
        self.finger_qpos: dict[int, float] = {}
        for side, sign in (("left", 1.0), ("right", -1.0)):
            for name, full in zip(FINGER_ORDER, FULL_CLOSE_LEFT):
                jid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hand_{name}_joint"
                )
                if jid < 0:
                    continue
                target = self.finger_closure * full * sign
                low, high = model.jnt_range[jid]
                self.finger_qpos[int(model.jnt_qposadr[jid])] = float(
                    np.clip(target, low, high)
                )

        collidable = [
            g for g in range(model.ngeom)
            if model.geom_contype[g] or model.geom_conaffinity[g]
        ]
        self.ee_geoms = {
            g for g in collidable
            if "hand" in self.body_name(g) or "wrist" in self.body_name(g)
        }
        self.trunk_leg_geoms = {
            g for g in collidable
            if any(k in self.body_name(g) for k in _TRUNK_LEG_KEYWORDS)
        }
        #: geoms belonging to the world body (floor); ground contact is normal
        self.world_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == 0}

        model.geom_margin[:] = 0.0
        if ee_margin > 0.0 and self.ee_geoms:
            model.geom_margin[list(self.ee_geoms)] = ee_margin

    def body_name(self, geom_id: int) -> str:
        return (
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[geom_id])
            or ""
        )

    def pose(
        self, joints_mj: np.ndarray, base_pos: np.ndarray, base_quat_wxyz: np.ndarray
    ) -> None:
        """Write one frame into ``qpos`` (joints in MuJoCo order) and run FK."""
        joints_mj = np.asarray(joints_mj, dtype=float).reshape(-1)
        if joints_mj.shape[0] != self.qadr.shape[0]:
            raise ValueError(
                f"Expected {self.qadr.shape[0]} body joints, got {joints_mj.shape[0]}"
            )
        data = self.data
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0  # identity quaternion before the base is written
        data.qpos[0:3] = np.asarray(base_pos, dtype=float).reshape(3)
        data.qpos[3:7] = np.asarray(base_quat_wxyz, dtype=float).reshape(4)
        data.qpos[self.qadr] = joints_mj
        for address, value in self.finger_qpos.items():
            data.qpos[address] = value
        mujoco.mj_forward(self.model, self.data)

    def contacts(self, tolerance: float = 1e-3) -> list[tuple[str, str, float]]:
        """Robot-vs-robot contacts penetrating deeper than ``tolerance``.

        Ground contact is excluded: the feet are supposed to be on the floor.
        """
        found = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 in self.world_geoms or g2 in self.world_geoms:
                continue
            if contact.dist < -tolerance:
                found.append((self.body_name(g1), self.body_name(g2), float(contact.dist)))
        return found

    def in_contact(
        self,
        joints_mj: np.ndarray,
        base_pos: np.ndarray,
        base_quat_wxyz: np.ndarray,
        tolerance: float = 1e-3,
    ) -> bool:
        """True if the robot self-collides in this configuration."""
        self.pose(joints_mj, base_pos, base_quat_wxyz)
        return bool(self.contacts(tolerance))

    def clearances(
        self, joints_mj: np.ndarray, base_pos: np.ndarray, base_quat_wxyz: np.ndarray
    ) -> tuple[float, float]:
        """(min EE-to-trunk/leg gap, min EE-to-EE gap) in metres; ``inf`` if
        nothing is within the halo. Needs ``ee_margin > 0``."""
        self.pose(joints_mj, base_pos, base_quat_wxyz)
        body_gap, ee_gap = np.inf, np.inf
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 in self.world_geoms or g2 in self.world_geoms:
                continue
            a, b = g1 in self.ee_geoms, g2 in self.ee_geoms
            gap = float(contact.dist)
            if a and b:
                # opposite hands only; a hand against itself is not a gap
                if self.body_name(g1).startswith("left") != self.body_name(g2).startswith("left"):
                    ee_gap = min(ee_gap, gap)
            elif (a and g2 in self.trunk_leg_geoms) or (b and g1 in self.trunk_leg_geoms):
                body_gap = min(body_gap, gap)
        return body_gap, ee_gap


def clearance_rig(
    scene_xml: str | Path = DEFAULT_COLLISION_SCENE, finger_closure: float = 0.9
) -> CollisionRig:
    """A rig with the hand detection halo enabled, for clearance measurement."""
    return CollisionRig(scene_xml, finger_closure, ee_margin=_EE_MARGIN)

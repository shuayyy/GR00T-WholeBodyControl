"""Joint-order authorities: MuJoCo, IsaacLab and Pinocchio orders.

Every conversion is by NAME, never position: the 43-DoF sim model interleaves
finger joints, so positional mapping silently misassigns the arms.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: MJCF model whose document order defines "MuJoCo order" for the 29 joints.
MJCF_29DOF_PATH = (
    _REPO_ROOT
    / "gear_sonic"
    / "data"
    / "assets"
    / "robot_description"
    / "mjcf"
    / "g1_29dof_rev_1_0.xml"
)

#: MJ_TO_IL[mj] = il.  Source and verification: module docstring.
MJ_TO_IL = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11,
     15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
)

NUM_BODY_JOINTS = 29

#: The sim bridge admits a scene joint into the body-motor list iff its name
#: contains one of these (gear_sonic/utils/mujoco_sim/base_sim.py:230).
_BODY_JOINT_KEYWORDS = ("hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist")


def mjcf_joint_names(xml_path: str | Path = MJCF_29DOF_PATH) -> list[str]:
    """The 29 body-joint names in MJCF document order.

    Excludes free and finger joints with the sim bridge's filter, so the sim
    scene XML yields ``LowState.motor_state`` order.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"MJCF not found: {xml_path}")

    names: list[str] = []

    def walk(path: Path) -> None:
        root = ET.parse(path).getroot()
        for element in root.iter():
            if element.tag == "include":
                include = element.get("file")
                if include:
                    walk(path.parent / include)
            elif element.tag == "joint":
                name = element.get("name")
                if name is None or element.get("type") == "free":
                    continue
                if any(keyword in name for keyword in _BODY_JOINT_KEYWORDS):
                    names.append(name)

    walk(xml_path)
    if len(names) != NUM_BODY_JOINTS:
        raise ValueError(
            f"Expected {NUM_BODY_JOINTS} body joints in {xml_path}, found "
            f"{len(names)}"
        )
    return names


def isaaclab_joint_names(xml_path: str | Path = MJCF_29DOF_PATH) -> list[str]:
    """The 29 joint names in IsaacLab order (deploy CSV / pose-topic order)."""
    mj_names = mjcf_joint_names(xml_path)
    il_names = [""] * NUM_BODY_JOINTS
    for mj_idx, il_idx in enumerate(MJ_TO_IL):
        il_names[il_idx] = mj_names[mj_idx]
    return il_names


def name_permutation(src_names: list[str], dst_names: list[str]) -> np.ndarray:
    """Indices such that ``values[perm]`` reorders src -> dst. Raises if the
    name sets differ, so a silent misassignment is impossible."""
    if sorted(src_names) != sorted(dst_names):
        missing = sorted(set(dst_names) - set(src_names))
        extra = sorted(set(src_names) - set(dst_names))
        raise ValueError(
            f"Joint name sets differ: missing from source {missing}, "
            f"unexpected in source {extra}"
        )
    src_index = {name: idx for idx, name in enumerate(src_names)}
    return np.array([src_index[name] for name in dst_names], dtype=int)

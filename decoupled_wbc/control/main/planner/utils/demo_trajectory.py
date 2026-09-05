"""Load a demo trajectory (.npz with joint_names + joint_pos/qpos, or (N, 17) .npy)
as a planning-order joint path."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from decoupled_wbc.control.main.planner.simulation.robot import JOINT_NAMES_UP

JOINT_ARRAY_KEYS = ("joint_pos", "qpos")


def resolve_path(path_value: str, base_dir: Optional[Path] = None) -> Path:
    """Absolute path; relative paths are taken from ``base_dir`` (default: cwd)."""
    path = Path(path_value)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_planning_trajectory(path_value: str, base_dir: Optional[Path] = None) -> np.ndarray:
    """``(N, 17)`` float64 joint path in ``JOINT_NAMES_UP`` order."""
    path = resolve_path(path_value, base_dir)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            key = next((k for k in JOINT_ARRAY_KEYS if k in data.files), None)
            if key is None or "joint_names" not in data.files:
                raise KeyError(
                    f"{path} needs 'joint_names' and one of {JOINT_ARRAY_KEYS}; "
                    f"has {sorted(data.files)}"
                )
            joints = np.asarray(data[key], dtype=np.float64)
            names = [str(n) for n in data["joint_names"].tolist()]
        if joints.ndim != 2 or joints.shape[1] != len(names):
            raise ValueError(f"{path}: {key} width {joints.shape} does not match joint_names")
        if len(names) != len(set(names)):
            raise ValueError(f"{path}: joint_names contains duplicates")
        missing = [n for n in JOINT_NAMES_UP if n not in names]
        if missing:
            raise ValueError(f"{path} is missing planning joints: {missing}")
        trajectory = joints[:, [names.index(n) for n in JOINT_NAMES_UP]]
    elif path.suffix == ".npy":
        trajectory = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    else:
        raise ValueError(f"{path}: trajectory must be a .npz or .npy file")

    if trajectory.ndim != 2 or trajectory.shape[1] != len(JOINT_NAMES_UP):
        raise ValueError(f"{path}: expected shape (N, {len(JOINT_NAMES_UP)}), got {trajectory.shape}")
    if trajectory.shape[0] < 2:
        raise ValueError(f"{path}: trajectory needs at least two frames")
    if not np.isfinite(trajectory).all():
        raise ValueError(f"{path}: trajectory contains non-finite values")
    return trajectory

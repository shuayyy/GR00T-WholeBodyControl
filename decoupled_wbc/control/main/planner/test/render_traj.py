"""Render a --record recording to mp4: --source exec (robot state) or plan
(interpolated plan over the same duration).  Output: <npz stem>_<source>.mp4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Literal

import mujoco
import numpy as np
import tyro

from decoupled_wbc.control.main.planner.utils.run_metrics import streamed_window
from decoupled_wbc.control.main.planner.utils.trajectory_ops import (
    arc_length,
    interp_rows,
)

PLANNER_DIR = Path(__file__).resolve().parents[1]
SCENE_XML = PLANNER_DIR / "simulation" / "envs" / "g1_free.xml"
# camera as in retarget_pipeline/render_config.yaml; azimuth = median base yaw + 180
WIDTH, HEIGHT = 640, 480
CAM_LOOKAT, CAM_DISTANCE, CAM_ELEVATION = (0.0, 0.0, 0.75), 2.3, -8.0


@dataclass
class RenderConfig:
    npz: str
    """Recording written by run_planner_server.py --record."""
    source: Literal["exec", "plan"] = "exec"
    fps: float = 50.0
    plan_frames: int = 1000
    """plan only: interpolate to this many waypoints (saved as <npz stem>_plan_interp.npz)."""
    trim: bool = True
    """exec only: keep from 0.3 s before streaming starts to 1 s after the last goal change."""
    out: str = ""


def time_grid(rec: dict, cfg: RenderConfig) -> np.ndarray:
    """Frame times shared by the exec and plan videos (trimmed to the streamed motion)."""
    t = rec["t"]
    t0, t1 = t[0], t[-1]
    if cfg.trim:
        stream_t0, stream_t1 = streamed_window(rec)
        t0, t1 = max(t[0], stream_t0 - 0.3), min(t[-1], stream_t1 + 1.0)
    return np.arange(t0, t1, 1.0 / cfg.fps)


def exec_frames(rec: dict, cfg: RenderConfig) -> tuple[np.ndarray, np.ndarray]:
    """(q, base_pose) at cfg.fps over the trimmed window."""
    grid = time_grid(rec, cfg)
    return interp_rows(rec["t"], rec["q"], grid), interp_rows(rec["t"], rec["base_pose"], grid)


def interpolate_plan(plan: np.ndarray, n_points: int) -> np.ndarray:
    """``plan`` linearly interpolated to ``n_points`` evenly spaced along its arc length."""
    s = arc_length(plan)
    return interp_rows(s, plan, np.linspace(0.0, s[-1], n_points)).astype(np.float32)


def plan_frames(
    rec: dict, plan_dense: np.ndarray, cfg: RenderConfig, full_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """The interpolated plan played at constant speed over the exec video's duration,
    legs/base frozen at the first recorded frame."""
    grid = time_grid(rec, cfg)
    frames = interp_rows(
        np.linspace(0.0, 1.0, len(plan_dense)), plan_dense, np.linspace(0.0, 1.0, len(grid))
    )
    q = np.repeat(rec["q"][:1], len(frames), axis=0)
    cols = [full_names.index(str(n)) for n in rec["planning_joint_names"]]
    q[:, cols] = frames
    base = np.repeat(rec["base_pose"][:1], len(frames), axis=0)
    return q, base


def render(q: np.ndarray, base: np.ndarray, full_names: list[str], fps: float, out: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    qadr = [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in full_names
    ]

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    # base +x axis in the world, median over the motion (tools/facing_azimuth.py)
    w, x, y, z = base[:, 3], base[:, 4], base[:, 5], base[:, 6]
    fwd = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)], axis=1)
    fwd = np.median(fwd, axis=0)
    cam.azimuth = (np.degrees(np.arctan2(fwd[1], fwd[0])) + 180.0) % 360.0
    cam.elevation, cam.distance = CAM_ELEVATION, CAM_DISTANCE
    cam.lookat = np.array(CAM_LOOKAT)

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-r",
            str(int(fps)),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-profile:v",
            "baseline",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for qi, bi in zip(q, base):
            data.qpos[:7] = bi  # free joint: xyz + wxyz, same layout as floating_base_pose
            data.qpos[qadr] = qi
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, cam)
            ffmpeg.stdin.write(renderer.render().tobytes())
    finally:
        ffmpeg.stdin.close()
        ffmpeg.wait()
        renderer.close()


def main(cfg: RenderConfig) -> None:
    npz = Path(cfg.npz).resolve()
    rec = dict(np.load(npz, allow_pickle=True))
    full_names = [str(n) for n in rec["full_joint_names"]]
    if cfg.source == "exec":
        q, base = exec_frames(rec, cfg)
    else:
        plan_dense = interpolate_plan(rec["plan_qpos"], cfg.plan_frames)
        np.savez(
            npz.with_name(f"{npz.stem}_plan_interp.npz"),
            waypoints=plan_dense,
            joint_names=rec["planning_joint_names"],
            source_npz=np.str_(str(npz)),
            note=np.str_(
                "plan_qpos linearly interpolated along arc length, evenly spaced, no smoothing"
            ),
        )
        q, base = plan_frames(rec, plan_dense, cfg, full_names)
    out = Path(cfg.out) if cfg.out else npz.with_name(f"{npz.stem}_{cfg.source}.mp4")
    render(q, base, full_names, cfg.fps, out)
    print(f"saved {out} ({len(q)} frames, {len(q) / cfg.fps:.1f}s)")


if __name__ == "__main__":
    main(tyro.cli(RenderConfig))

"""Plan through every obstacle scene with PhaseRRT* and render the result.

Git-ignored scratch tool.  Planning and kinematic playback only: no controller, no
physics step, no robot.

    python obstacle_plan.py --run       # plan all nine scenes, save to obstacles/plans/
    python obstacle_plan.py --video     # render <scene>_plan.mp4 next to the demo video
    python obstacle_plan.py --report    # deviation table

The shipped sampling width cannot solve an obstacle case: with sigma = 0.0018 * L
and uniform_fraction = 0 every sample lands in a +-1.5 degree sleeve around the
demonstration, so a blocked sleeve leaves no detour to find, and the planner
returns "Approximate solution" no matter how long it runs.  SIGMA_SCALE below is
the only departure from the shipped configuration; it solved 15/15 seeds across
three scenes where the shipped width solved 0/15 (report/ICRA/tables.md section 6).

OMPL seeds its RNG once per process, so each plan runs in its own subprocess.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import obstacle_lab as lab
from obstacle_lab import ENVS, PLANNER, REPO

from decoupled_wbc.control.main.planner.utils.demo_trajectory import (  # noqa: E402
    load_planning_trajectory,
)
from decoupled_wbc.control.main.planner.utils.phaserrtstar import (  # noqa: E402
    PhaseRRTstarPlanner,
    phase_defaults,
    reference_arclength,
)
from decoupled_wbc.control.main.planner.utils.run_metrics import path_deviation  # noqa: E402
from decoupled_wbc.control.main.planner.utils.trajectory_ops import (  # noqa: E402
    resample_waypoints,
)
from simulation.robot import JOINT_NAMES_UP  # noqa: E402

OUT = REPO / "obstacles" / "plans"
PREVIEW = REPO / "obstacles" / "preview"

SIGMA_SCALE = 10.0  # see the module docstring; everything else is the shipped config
VIDEO_FPS = 50.0  # playback rate; the frame count then follows from the duration
PLANNER_HZ = 20.0  # goals are streamed at this rate, so the plan takes waypoints/HZ
TIMEOUT = 20.0
MAX_JOINT_STEP = 0.025  # rad per waypoint, the execution spacing used for all results


@dataclass(frozen=True)
class Case:
    """One planning problem: a demonstration and the scene it has to get through."""

    key: str
    demo: str
    scene: Path


# The six scenes obstacle_lab builds, plus the three frozen floating-box scenes that
# predate the 2-7% rule and are kept as they are (see obstacles/TODO.md).
CASES: tuple[Case, ...] = tuple(
    [Case(s.key, s.demo, s.path) for s in lab.SCENES]
    + [
        Case(d, d, ENVS / f"g1_obstacle_{d}.xml")
        for d in ("pour", "single_sweep", "dualarm_sweep")
    ]
)
BY_KEY = {c.key: c for c in CASES}


def planner_params(reference: np.ndarray) -> dict:
    """The shipped PhaseRRT* defaults with the sampling width widened."""
    base = phase_defaults(reference_arclength(reference))
    return {"sigma": base["sigma"] * SIGMA_SCALE}


def plan_one(case: Case) -> None:
    """One plan, in its own process, saved to obstacles/plans/<key>.npz."""
    reference = load_planning_trajectory(f"dataset/ICRA/{case.demo}/traj.npz", PLANNER)
    robot, _ = lab.build_robot(case.scene)
    planner = PhaseRRTstarPlanner(
        robot, reference, validity_resolution=0.01, phase_params=planner_params(reference), log=False
    )
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        path = planner.plan(
            reference[0], reference[-1], "upper_body",
            timeout=TIMEOUT, smooth_path=False, shortcut_path=False,
        )
    except RuntimeError as exc:
        np.savez(OUT / f"{case.key}.npz", failed=np.str_(str(exc)),
                 reference=reference.astype(np.float32), key=np.str_(case.key))
        print(f"FAILED {case.key}: {exc}")
        robot.close()
        return

    path = np.asarray(path, dtype=np.float32)
    hits = lab.frames_hitting(robot, robot.model, path, "obstacle")
    robot.close()
    np.savez(
        OUT / f"{case.key}.npz",
        plan=path,
        waypoints=resample_waypoints(path, MAX_JOINT_STEP),
        reference=reference.astype(np.float32),
        joint_names=np.array(JOINT_NAMES_UP),
        key=np.str_(case.key),
        demo=np.str_(case.demo),
        scene=np.str_(case.scene.name),
        sigma_scale=np.float64(SIGMA_SCALE),
        timeout=np.float64(TIMEOUT),
    )
    print(f"saved {case.key}: {len(path)} waypoints, {len(hits)} colliding")


def load(case: Case) -> dict | None:
    f = OUT / f"{case.key}.npz"
    if not f.exists():
        print(f"missing {f.name}, run --run first")
        return None
    d = np.load(f, allow_pickle=True)
    if "failed" in d.files:
        print(f"skipping {case.key}: {d['failed']}")
        return None
    return d


def render(case: Case) -> None:
    """The planned path played through its scene, matched to the demo video.

    Both videos run for as long as the robot would actually take: the resampled
    plan is 0.025 rad per waypoint streamed at PLANNER_HZ, so 0.5 rad/s, which is
    the speed every simulation and hardware result was recorded at.  Frame count
    follows from that duration at VIDEO_FPS, which is what keeps the motion smooth
    without slowing it down.  The camera is identical for both.
    """
    d = load(case)
    if d is None:
        return
    demo = load_planning_trajectory(f"dataset/ICRA/{case.demo}/traj.npz", PLANNER)

    # The camera frames the demonstration, measured in the obstacle-free scene, so
    # both videos of a scene use exactly the same viewpoint.
    robot, model = lab.build_robot(ENVS / "g1_free.xml")
    paths = lab.wrist_paths(robot, model, demo)
    robot.close()
    path = paths[lab.moving_side(paths)]
    lookat = np.array([path[:, 0].mean(), path[:, 1].mean(), path[:, 2].mean()])

    plan = np.asarray(d["plan"], dtype=float)
    seconds = len(d["waypoints"]) / PLANNER_HZ
    n_frames = max(int(round(seconds * VIDEO_FPS)), 2)

    PREVIEW.mkdir(parents=True, exist_ok=True)
    robot, model = lab.build_robot(case.scene)
    try:
        for source, suffix in ((plan, "_plan"), (np.asarray(demo, float), "")):
            out = PREVIEW / f"{case.key}{suffix}.mp4"
            lab.render_video(robot, model, interpolate(source, n_frames), lookat, out, VIDEO_FPS)
            print(f"{out.relative_to(REPO)}: {n_frames} frames from {len(source)} "
                  f"waypoints, {seconds:.1f}s at 0.5 rad/s")
    finally:
        robot.close()


def interpolate(plan: np.ndarray, n: int) -> np.ndarray:
    """``plan`` resampled to ``n`` points evenly spaced along its arc length."""
    step = np.linalg.norm(np.diff(plan, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(step)])
    target = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(target, s, plan[:, j]) for j in range(plan.shape[1])], axis=1)


def report() -> None:
    rows = []
    for case in CASES:
        d = load(case)
        if d is None:
            continue
        plan = np.asarray(d["plan"], float)
        reference = np.asarray(d["reference"], float)
        dev = path_deviation(plan, reference)
        robot, _ = lab.build_robot(case.scene)
        hits = len(lab.frames_hitting(robot, robot.model, plan, "obstacle"))
        robot.close()
        rows.append((case.key, len(plan), dev.joint_mean, dev.right.pos_mm, dev.left.pos_mm, hits))

    print("\n| scene | waypoints | joint dev (rad) | right EE (mm) | left EE (mm) | collides |")
    print("|---|---|---|---|---|---|")
    for key, n, joint, right, left, hits in rows:
        print(f"| {key} | {n} | {joint:.3f} | {right:.0f} | {left:.0f} | {hits} |")
    print(f"\nPhaseRRT*, sigma x{SIGMA_SCALE:.0f}, {TIMEOUT:.0f} s, no smoothing. "
          "Joint deviation is |plan - demo| after arc-length matching, mean over 17 joints.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", action="store_true", help="plan every scene")
    ap.add_argument("--video", action="store_true", help="render every saved plan")
    ap.add_argument("--report", action="store_true", help="print the deviation table")
    ap.add_argument("--scene", choices=sorted(BY_KEY), help="limit to one scene")
    ap.add_argument("--worker", metavar="KEY", help=argparse.SUPPRESS)
    cfg = ap.parse_args()

    if cfg.worker:
        plan_one(BY_KEY[cfg.worker])
        return
    cases = (BY_KEY[cfg.scene],) if cfg.scene else CASES
    if cfg.run:
        for case in cases:
            print(f"\n>>> planning {case.key} ({case.scene.name}), {TIMEOUT:.0f} s")
            subprocess.run([sys.executable, __file__, "--worker", case.key], cwd=REPO)
    if cfg.video:
        for case in cases:
            render(case)
    if cfg.report:
        report()


if __name__ == "__main__":
    main()

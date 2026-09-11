"""Build the obstacle planning scenes and check what they block.

Git-ignored scratch tool.  A scene is a list of components plus one obstacle; the
obstacle's size and pose are searched so that it blocks a small slice out of the
middle of the demonstration and leaves the endpoints reachable.

    python obstacle_lab.py --all                 # build every scene, write previews
    python obstacle_lab.py --scene pass          # just one
    python obstacle_lab.py --scene wave --view   # interactive viewer
    python obstacle_lab.py --all --video         # also render the demo as an mp4

Placement rules (see obstacles/TODO.md):

* the first and last ``--margin`` frames must be clear of the obstacle, otherwise
  the plan has no valid start or goal and OMPL cannot even begin;
* the obstacle must block 2-7% of the demonstration frames -- small enough to be
  an obstacle rather than a wall.  Only contacts naming the obstacle count; the
  table is scenery and its own contacts are reported but never counted;
* ``wave`` is exempt from the percentage rule: its ceiling takes the highest
  position that blocks the motion at all.

The three floating-box scenes built earlier (pour, single_sweep, dualarm_sweep)
are deliberately NOT regenerated here -- they predate the percentage rule and are
frozen so their planning results stay valid.  See obstacles/TODO.md.

A search moves the obstacle by writing ``model.body_pos``.  It never resizes a
geom in place: MuJoCo compiles a bounding-volume hierarchy for welded bodies, and
that hierarchy is not rebuilt when ``geom_size`` changes, so a resized obstacle
silently stops colliding.  Each candidate size therefore gets its own scene file
and its own model load, and only translation happens inside a pass.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

REPO = Path(__file__).resolve().parent
PLANNER = REPO / "decoupled_wbc" / "control" / "main" / "planner"
ENVS = PLANNER / "simulation" / "envs"
PREVIEW = REPO / "obstacles" / "preview"
sys.path.insert(0, str(PLANNER))

import mujoco  # noqa: E402

from decoupled_wbc.control.main.planner.utils.demo_trajectory import (  # noqa: E402
    load_planning_trajectory,
)
from decoupled_wbc.control.robot_model.instantiation.g1 import (  # noqa: E402
    instantiate_g1_robot_model,
)
from simulation.robot import G1Up, JOINT_NAMES_UP  # noqa: E402

BASE_POS, BASE_QUAT = np.array([0.0, 0.0, 0.74]), np.array([1.0, 0.0, 0.0, 0.0])

# MuJoCo reports a contact for every near-touch; in_contact() only believes one
# deeper than this, so the obstacle count has to use the same threshold.
PENETRATION = 1e-3

BAND = (0.02, 0.07)  # fraction of demo frames the obstacle must block
BAND_TARGET = 0.05  # aim for this within the band
# Where along the motion the block may fall.  An obstacle clipping the opening of a
# gesture is hard to see and hardly tests the planner; it belongs in the body of it.
BLOCK_WINDOW = (0.20, 0.80)
PILLAR_FOOTPRINTS = (0.03, 0.04, 0.05, 0.06, 0.08)  # box edge, smallest tried first
# A post standing directly under the path is swept by the whole forearm and blocks
# far more than BAND allows, so candidates are also offset sideways from the path
# and tried at a few heights; the post then clips the motion instead of barring it.
PILLAR_HEIGHTS = (0.5, 0.75, 1.0)  # fraction of the wrist reach above the table
PILLAR_OFFSETS = np.arange(-0.09, 0.0901, 0.02)  # lateral offset from the path, m
PILLAR_FRAME_STEP = 2  # candidate frames along the path; the path is dense at 50 Hz
# An overhead panel, not a room-wide ceiling: the wave passes close to the head,
# and a wide plate touches the head on every frame instead of the raised hand.
CEILING_HALF = np.array([0.15, 0.15, 0.02])
# The planner leaves no margin of its own, and the RL controller tracks its goals to
# about 0.045 rad -- roughly 40 mm at the wrist -- so a path that merely grazes the
# box is driven straight into it on execution.  The collision geom is therefore
# grown by this much on every side while the visual geom keeps the true size: the
# planner avoids a fat box, everything reported measures the real one.
#
# 10 mm.  The tracking error would justify far more, but the demonstrations end
# close to their obstacle -- at 40 mm the inflated box swallows the goal frame and
# no plan exists -- and a wide margin visibly fattens a 3 cm post.  Execution can
# still clip the box.  The smoother is validated against the true-size scene
# (SceneSpec.true_path), so it may spend part of this margin.
CLEARANCE = 0.01


@dataclass(frozen=True)
class SceneSpec:
    """One obstacle scene: which components it includes and what blocks the demo.

    ``kind`` is "pillar" (a post standing on the table) or "ceiling" (a panel
    hanging above the robot).  ``suffix`` distinguishes a second scene for a demo
    that already has one, e.g. single_sweep -> g1_obstacle_single_sweep_table.xml.
    """

    demo: str
    components: tuple[str, ...]
    kind: str
    suffix: str = ""

    @property
    def key(self) -> str:
        return f"{self.demo}{self.suffix}"

    @property
    def path(self) -> Path:
        return ENVS / f"g1_obstacle_{self.key}.xml"

    @property
    def true_path(self) -> Path:
        """Same scene with the obstacle at its real size, for validating smoothed
        paths: pass it as --validation-xml alongside --planning-xml."""
        return ENVS / f"g1_obstacle_{self.key}_true.xml"


TABLE = ("floor", "g1", "table")
TABLE_OFFSETS = np.arange(0.0, 1.001, 0.05)  # how far the table is pushed out, m

# The lab_scene table is one coarse collision block sitting at x=0.5 that swallows the
# robot: every demonstration frame collides with it, start included, so OMPL rejects the
# start before it samples anything.  Each scene therefore gets its own copy of the
# component, pushed out along +x until the demonstration's endpoints are clear.
TABLE_COMPONENT = """<!-- Lab table for {key}: components/table.xml pushed out {dx:.2f} m along +x so
     the demonstration's start and goal are clear of its collision block.
     Generated by obstacle_lab.py; edit that, not this. -->
<mujoco model="table_{key}_component">
  <asset>
    <mesh name="lab_scene_visual_mesh" file="../../../assets/lab_scene/textured.obj"/>
    <mesh name="lab_scene_collision_mesh" file="../../../assets/lab_scene/textured_box.stl"/>
    <texture name="lab_scene_texture" type="2d" file="../../../assets/lab_scene/texture_map.png"/>
    <material name="lab_scene_material" texture="lab_scene_texture"/>
  </asset>
  <worldbody>
    <body name="lab_scene" pos="{x:.4f} 0 0.7" euler="0 0 1.5708">
      <geom name="lab_scene_visual_geom" type="mesh" mesh="lab_scene_visual_mesh" material="lab_scene_material"
            pos="0.0 -0.508 -0.7" euler="0 0 -1.57079632679" contype="0" conaffinity="0" group="0"/>
      <geom name="lab_scene_collision_geom" type="mesh" mesh="lab_scene_collision_mesh"
            pos="0.0 -0.508 -0.7" euler="0 0 -1.57079632679" contype="1" conaffinity="1" group="3"/>
    </body>
  </worldbody>
</mujoco>
"""
TABLE_BASE_X = 0.5  # components/table.xml puts the body here
SCENES: tuple[SceneSpec, ...] = (
    SceneSpec("pass", ("floor", "g1"), "box"),
    SceneSpec("pass2", ("floor", "g1"), "box"),
    SceneSpec("handover", ("floor", "g1"), "box"),
    SceneSpec("pour", ("floor", "g1"), "box", suffix="_box"),
    SceneSpec("single_sweep", TABLE, "pillar", suffix="_table"),
    SceneSpec("dualarm_sweep", TABLE, "pillar", suffix="_table"),
    SceneSpec("wave", ("floor", "g1"), "ceiling"),
)
BY_KEY = {s.key: s for s in SCENES}


# --------------------------------------------------------------------------- scene

def scene_xml(spec: SceneSpec, pos: np.ndarray, half: np.ndarray, table: str = "table",
              clearance: float = 0.0) -> str:
    """The scene file: component includes plus the obstacle body.

    ``table`` names the table component to include; each scene gets its own, moved
    so the demonstration's endpoints clear it (see TABLE_COMPONENT).

    ``clearance`` grows the collision geom on every side without touching the
    visual one, so the planner keeps that much distance from the real obstacle.
    """
    names = [table if c == "table" else c for c in spec.components]
    includes = "\n".join(f'  <include file="components/{c}.xml"/>' for c in names)
    p = " ".join(f"{v:.6f}" for v in pos)
    s = " ".join(f"{v:.6f}" for v in half)
    sc = " ".join(f"{v:.6f}" for v in np.asarray(half) + clearance)
    return f"""<mujoco model="g1_obstacle_{spec.key}">
  <compiler angle="radian" autolimits="true"/>
  <visual><global offwidth="1280" offheight="960"/></visual>
{includes}
  <worldbody>
    <body name="obstacle" pos="{p}">
      <geom name="obstacle_visual" type="box" size="{s}"
            rgba="0.85 0.25 0.2 1" contype="0" conaffinity="0" group="0"/>
      <geom name="obstacle" type="box" size="{sc}"
            rgba="0.85 0.25 0.2 1" contype="1" conaffinity="1" group="3"/>
    </body>
  </worldbody>
</mujoco>
"""


def build_robot(scene: Path) -> tuple[G1Up, mujoco.MjModel]:
    """G1 in the given scene, legs and base at the WBC standing pose."""
    robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
    cwd = os.getcwd()
    os.chdir(PLANNER)  # the scene's <include> paths are relative to the planner package
    try:
        model = mujoco.MjModel.from_xml_path(str(scene))
    finally:
        os.chdir(cwd)
    names = {model.joint(i).name for i in range(model.njnt)}
    fixed = {
        n: float(v)
        for n, v in zip(robot_model.joint_names, robot_model.default_body_pose)
        if n not in JOINT_NAMES_UP and n in names
    }
    return G1Up(model=model, fixed_qpos=fixed, base_pose=(BASE_POS, BASE_QUAT)), model


TABLE_GEOM = "lab_scene_collision_geom"


def table_surface(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, tuple[float, float, float, float]]:
    """The table's top height and (xmin, xmax, ymin, ymax) footprint, from the model.

    Read rather than assumed, so moving the table component cannot silently leave
    a post hanging in the air beside it.  The lab_scene table is a mesh, so its
    extent comes from the compiled bounding box, rotated into the world by the
    body's own orientation.
    """
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM)
    if gid < 0:
        raise ValueError(f"scene has no '{TABLE_GEOM}' geom")
    rot = data.geom_xmat[gid].reshape(3, 3)
    centre = data.geom_xpos[gid] + rot @ model.geom_aabb[gid][:3]
    half = np.abs(rot) @ model.geom_aabb[gid][3:]
    top = float(centre[2] + half[2])
    return top, (
        float(centre[0] - half[0]),
        float(centre[0] + half[0]),
        float(centre[1] - half[1]),
        float(centre[1] + half[1]),
    )


def write_table(spec: SceneSpec, dx: float) -> str:
    """Write this scene's table component and return its name."""
    name = f"table_{spec.key}"
    (ENVS / "components" / f"{name}.xml").write_text(
        TABLE_COMPONENT.format(key=spec.key, dx=dx, x=TABLE_BASE_X + dx)
    )
    return name


def table_offset(spec: SceneSpec, demo: np.ndarray, margin: int) -> float:
    """Smallest push along +x that clears the table from the demo's first and last frames.

    Only the endpoints have to be clear: they become the plan's start and goal, and an
    invalid one makes the problem unsolvable.  The table may still cross the middle of
    the motion, which is a legitimate obstacle for the planner to route around.
    """
    for dx in TABLE_OFFSETS:
        name = write_table(spec, float(dx))
        spec.path.write_text(scene_xml(spec, demo[0][:3] * 0, np.full(3, 0.01), table=name))
        robot, model = build_robot(spec.path)
        ends = list(range(margin)) + list(range(len(demo) - margin, len(demo)))
        hit = bool(frames_hitting(robot, model, demo[ends], TABLE_GEOM))
        robot.close()
        if not hit:
            return float(dx)
    raise RuntimeError(
        f"{spec.key}: the table still touches the demonstration's endpoints at "
        f"+{TABLE_OFFSETS[-1]:.2f} m; it cannot be moved out of the way along x alone"
    )


# ------------------------------------------------------------------------- measure

def wrist_paths(robot: G1Up, model: mujoco.MjModel, demo: np.ndarray) -> dict:
    """World-frame wrist positions for every demo frame, per side."""
    bid = {
        s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_yaw_link")
        for s in ("left", "right")
    }
    out = {s: [] for s in bid}
    for q in demo:
        robot.set_joint_qpos(q)
        for s, b in bid.items():
            out[s].append(robot.data.xpos[b].copy())
    return {s: np.asarray(v) for s, v in out.items()}


def moving_side(paths: dict) -> str:
    return max(paths, key=lambda s: np.linalg.norm(np.diff(paths[s], axis=0), axis=1).sum())


def frames_hitting(
    robot: G1Up,
    model: mujoco.MjModel,
    demo: np.ndarray,
    geom: str,
    cap: int | None = None,
) -> list[int]:
    """Demo frames where the arm penetrates the named geom.

    Contacts with anything else -- the table above all -- are ignored, so the
    percentage rule measures the obstacle alone.

    ``cap`` stops the sweep as soon as more than that many frames are hit.  Most
    candidate placements bar the motion outright, and a caller that will reject
    them anyway does not need the exact count; the returned list is then partial
    and only its length past ``cap`` is meaningful.
    """
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
    if gid < 0:
        raise ValueError(f"scene has no '{geom}' geom")
    hits = []
    for i, q in enumerate(demo):
        robot.set_joint_qpos(q)
        d = robot.data
        for c in range(d.ncon):
            if d.contact.dist[c] < -PENETRATION and gid in (
                d.contact.geom1[c],
                d.contact.geom2[c],
            ):
                hits.append(i)
                break
        if cap is not None and len(hits) > cap:
            break
    return hits


def endpoints_clear(hits: Sequence[int], n: int, margin: int) -> bool:
    """The plan's start and goal must be reachable, so the ends stay untouched."""
    return not hits or (min(hits) >= margin and max(hits) < n - margin)


# -------------------------------------------------------------------------- search

@dataclass
class Placement:
    pos: np.ndarray
    half: np.ndarray
    hits: list[int]


def scan(
    spec: SceneSpec,
    demo: np.ndarray,
    half: np.ndarray,
    positions: Sequence[np.ndarray],
    accept: Callable[[list[int]], bool],
    score: Callable[[np.ndarray, list[int]], tuple],
    cap: int | None = None,
    table: str = "table",
) -> Placement | None:
    """Try one obstacle size at many positions; return the best accepted placement.

    The scene is written once at this size and loaded once; candidates differ only
    by ``body_pos``, which is safe to change on a compiled model.
    """
    spec.path.write_text(scene_xml(spec, positions[0], half, table=table))
    robot, model = build_robot(spec.path)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obstacle")
    best: tuple[tuple, Placement] | None = None
    try:
        for pos in positions:
            model.body_pos[bid] = pos
            hits = frames_hitting(robot, model, demo, "obstacle", cap)
            if not accept(hits):
                continue
            s = score(pos, hits)
            if best is None or s < best[0]:
                best = (s, Placement(pos.copy(), half.copy(), list(hits)))
    finally:
        robot.close()
    return None if best is None else best[1]


def path_normal(path: np.ndarray, i: int) -> np.ndarray | None:
    """Horizontal unit vector perpendicular to the path's direction at frame i."""
    d = path[min(i + 3, len(path) - 1)] - path[max(i - 3, 0)]
    normal = np.array([-d[1], d[0], 0.0])
    length = float(np.linalg.norm(normal))
    return None if length < 1e-9 else normal / length


def search_pillar(
    spec: SceneSpec,
    demo: np.ndarray,
    path: np.ndarray,
    top: float,
    footprint: tuple[float, float, float, float],
    margin: int,
    bite: float,
    table: str,
) -> Placement | None:
    """A post standing on the table, offset sideways so it clips the motion.

    Footprints are tried smallest first and heights shortest first, and the first
    combination with an in-band placement wins, so the obstacle is never larger
    than it has to be.  Candidates that would leave the table top are discarded.
    """
    n = len(demo)
    reach = float(path[:, 2].max() + bite - top)
    if reach <= 0:
        return None
    band_centre = sum(BAND) / 2
    x_min, x_max, y_min, y_max = footprint

    def accept(hits: list[int]) -> bool:
        return (
            bool(hits)
            and endpoints_clear(hits, n, margin)
            and BAND[0] <= len(hits) / n <= BAND[1]
            and BLOCK_WINDOW[0] <= hits[0] / n
            and hits[-1] / n <= BLOCK_WINDOW[1]
        )

    def score(pos: np.ndarray, hits: list[int]) -> tuple:
        return (abs(len(hits) / n - BAND_TARGET),)

    for edge in PILLAR_FOOTPRINTS:
        for fraction in PILLAR_HEIGHTS:
            height = reach * fraction
            if height <= edge:  # shorter than it is wide: not a post
                continue
            half = np.array([edge / 2, edge / 2, height / 2])
            positions = []
            for i in range(margin, n - margin, PILLAR_FRAME_STEP):
                normal = path_normal(path, i)
                if normal is None:
                    continue
                for offset in PILLAR_OFFSETS:
                    p = path[i] + normal * offset
                    # the whole base has to be on the table, not just its centre:
                    # a centre-only test lets a post hang 40% of its width off the edge
                    if not (
                        x_min <= p[0] - half[0] and p[0] + half[0] <= x_max
                        and y_min <= p[1] - half[1] and p[1] + half[1] <= y_max
                    ):
                        continue
                    positions.append(np.array([p[0], p[1], top + height / 2]))
            if not positions:
                continue
            found = scan(spec, demo, half, positions, accept, score, int(BAND[1] * n), table)
            if found is not None:
                return found
    return None


def search_box(
    spec: SceneSpec, demo: np.ndarray, path: np.ndarray, margin: int, bite: float
) -> Placement | None:
    """A cube floating in the path, for demonstrations that work away from any table.

    Same rules as the pillar -- BAND, BLOCK_WINDOW, clear endpoints, smallest cube
    first -- but the obstacle hangs at the wrist's own height instead of standing on
    something, so nothing constrains where it may go.
    """
    n = len(demo)

    def accept(hits: list[int]) -> bool:
        return (
            bool(hits)
            and endpoints_clear(hits, n, margin)
            and BAND[0] <= len(hits) / n <= BAND[1]
            and BLOCK_WINDOW[0] <= hits[0] / n
            and hits[-1] / n <= BLOCK_WINDOW[1]
        )

    def score(pos: np.ndarray, hits: list[int]) -> tuple:
        return (abs(len(hits) / n - BAND_TARGET),)

    for edge in PILLAR_FOOTPRINTS:
        half = np.full(3, edge / 2)
        positions = []
        for i in range(margin, n - margin, PILLAR_FRAME_STEP):
            normal = path_normal(path, i)
            if normal is None:
                continue
            for offset in PILLAR_OFFSETS:
                positions.append(path[i] + normal * offset)
        if not positions:
            continue
        found = scan(spec, demo, half, positions, accept, score, int(BAND[1] * n))
        if found is not None:
            return found
    return None


def search_ceiling(
    spec: SceneSpec, demo: np.ndarray, path: np.ndarray, margin: int, min_frames: int, bite: float
) -> Placement | None:
    """An overhead panel, slid along the wrist path so it caps the raised hand.

    Exempt from BAND.  The highest workable panel wins -- a ceiling sits at the top
    of the motion, not beside it -- with the fewest blocked frames as the tie-break.
    """
    n = len(demo)
    half = CEILING_HALF

    def accept(hits: list[int]) -> bool:
        return len(hits) >= min_frames and endpoints_clear(hits, n, margin)

    def score(pos: np.ndarray, hits: list[int]) -> tuple:
        return (-pos[2], len(hits))

    positions = [
        np.array([path[i][0], path[i][1], path[i][2] - bite + half[2]])
        for i in range(margin, n - margin)
    ]
    return scan(spec, demo, half, positions, accept, score)


# -------------------------------------------------------------------------- render

def _camera(lookat: np.ndarray) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation, cam.distance = 150.0, -10.0, 2.0
    cam.lookat = lookat
    return cam


def _options() -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1  # obstacle and table visual geoms
    opt.geomgroup[2] = 1  # robot visual meshes (class "visual" in g1.xml)
    return opt


def render_image(robot, model, demo, frames, lookat, out: Path) -> None:
    """One 2x2 image with the robot drawn at four demo frames."""
    renderer = mujoco.Renderer(model, height=600, width=800)
    cam, opt = _camera(lookat), _options()
    tiles = []
    for f in frames:
        robot.set_joint_qpos(demo[f])
        renderer.update_scene(robot.data, cam, opt)
        tiles.append(renderer.render().copy())
    renderer.close()
    grid = np.concatenate(
        [np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:4], axis=1)], axis=0
    )
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{grid.shape[1]}x{grid.shape[0]}", "-i", "-", "-frames:v", "1", str(out)],
        stdin=subprocess.PIPE,
    )
    p.stdin.write(grid.tobytes())
    p.stdin.close()
    p.wait()


def render_video(robot, model, demo, lookat, out: Path, fps: float) -> None:
    """The whole demonstration played through the scene."""
    w, h = 960, 720
    renderer = mujoco.Renderer(model, height=h, width=w)
    cam, opt = _camera(lookat), _options()
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(int(fps)), "-i", "-", "-c:v", "libx264", "-crf", "18",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )
    try:
        for q in demo:
            robot.set_joint_qpos(q)
            renderer.update_scene(robot.data, cam, opt)
            p.stdin.write(renderer.render().tobytes())
    finally:
        p.stdin.close()
        p.wait()
        renderer.close()


# ----------------------------------------------------------------------------- run

def play(spec: SceneSpec, cfg: argparse.Namespace) -> None:
    """Play the demonstration through the scene as it stands and save the video.

    No search and no planning: the scene file is loaded exactly as it is on disk,
    so this shows what the obstacle and the table actually do to the demo.  Played
    at the demonstration's own rate, one video frame per captured frame.
    """
    if not spec.path.exists():
        print(f"{spec.key}: no scene at {spec.path.relative_to(REPO)}, build it first")
        return
    demo = load_planning_trajectory(f"dataset/ICRA/{spec.demo}/traj.npz", PLANNER)
    n = len(demo)

    robot, model = build_robot(ENVS / "g1_free.xml")
    paths = wrist_paths(robot, model, demo)
    robot.close()
    path = paths[moving_side(paths)]
    lookat = np.array([path[:, 0].mean(), path[:, 1].mean(), path[:, 2].mean()])

    robot, model = build_robot(spec.path)
    hits = frames_hitting(robot, model, demo, "obstacle")
    table_hits = frames_hitting(robot, model, demo, TABLE_GEOM) if "table" in "".join(
        spec.components
    ) else []
    PREVIEW.mkdir(parents=True, exist_ok=True)
    out = PREVIEW / f"{spec.key}.mp4"
    render_video(robot, model, demo, lookat, out, cfg.fps)
    robot.close()

    span = f"{hits[0]}..{hits[-1]}" if hits else "-"
    print(f"{spec.key:22s} {out.relative_to(REPO)}  {n} frames, {n/cfg.fps:.1f}s  "
          f"obstacle {len(hits)}/{n} ({len(hits)/n:.1%}, {span})  table {len(table_hits)}/{n}")


def build(spec: SceneSpec, cfg: argparse.Namespace) -> None:
    demo = load_planning_trajectory(f"dataset/ICRA/{spec.demo}/traj.npz", PLANNER)
    n = len(demo)

    # Where does the wrist go?  Measured in the obstacle-free scene.
    robot, model = build_robot(ENVS / "g1_free.xml")
    paths = wrist_paths(robot, model, demo)
    robot.close()
    side = moving_side(paths)
    path = paths[side]

    if spec.kind == "pillar":
        # the table height comes from the scene, so write a provisional one to read it
        dx = table_offset(spec, demo, cfg.margin)
        table = write_table(spec, dx)
        spec.path.write_text(scene_xml(spec, path[n // 2], np.full(3, 0.025), table=table))
        robot, model = build_robot(spec.path)
        top, footprint = table_surface(model, robot.data)
        robot.close()
        print(f"[{spec.key}] table pushed +{dx:.2f} m to clear the endpoints; "
              f"top {top:.3f} m, footprint x[{footprint[0]:.2f},{footprint[1]:.2f}] "
              f"y[{footprint[2]:.2f},{footprint[3]:.2f}]")
        found = search_pillar(spec, demo, path, top, footprint, cfg.margin, cfg.bite, table)
    elif spec.kind == "box":
        table = "table"
        found = search_box(spec, demo, path, cfg.margin, cfg.bite)
    else:
        table = "table"
        found = search_ceiling(spec, demo, path, cfg.margin, cfg.min_frames, cfg.bite)

    if found is None:
        spec.path.unlink(missing_ok=True)
        print(f"\n=== {spec.key} ===\nno placement found: no {spec.kind} blocks "
              f"{BAND[0]:.0%}-{BAND[1]:.0%} of the demo with {cfg.margin} clear frames "
              f"at each end.  Widen PILLAR_FOOTPRINTS or relax BAND.")
        return

    # Measure the band against the true obstacle: how much of the demonstration the
    # real box blocks is a property of the demo, not of the margin we plan with.
    spec.path.write_text(scene_xml(spec, found.pos, found.half, table=table))
    robot, model = build_robot(spec.path)
    hits = frames_hitting(robot, model, demo, "obstacle")
    table_hits = (
        frames_hitting(robot, model, demo, TABLE_GEOM) if "table" in spec.components else []
    )
    size = found.half * 2
    ok = bool(hits) and endpoints_clear(hits, n, cfg.margin)

    print(f"\n=== {spec.key} ===")
    print(f"components   : {', '.join(spec.components)}")
    print(f"moving wrist : {side}, z {path[:, 2].min():.2f}-{path[:, 2].max():.2f} m")
    print(f"obstacle     : {spec.kind}, {size[0]*100:.0f}x{size[1]*100:.0f}x{size[2]*100:.0f} cm "
          f"at [{found.pos[0]:.3f} {found.pos[1]:.3f} {found.pos[2]:.3f}]")
    if hits:
        print(f"blocks       : {len(hits)}/{n} frames ({len(hits)/n:.1%}), "
              f"frames {hits[0]}..{hits[-1]} = {hits[0]/n:.0%}-{hits[-1]/n:.0%} of the motion")
    else:
        print(f"blocks       : nothing")
    print(f"endpoints    : first {cfg.margin} and last {cfg.margin} frames clear -> "
          f"{'OK' if ok else 'FAIL'}")
    if "table" in spec.components:
        print(f"table        : touches {len(table_hits)}/{n} frames (scenery, not counted)")
    print(f"scene        : {spec.path.relative_to(REPO)}")
    if not ok:
        raise RuntimeError(f"{spec.key}: the written scene does not match the search result")

    # Now write it for real, with the collision geom grown by CLEARANCE.  The visual
    # geom stays true size, so previews and videos still show the real obstacle.
    robot.close()
    spec.true_path.write_text(scene_xml(spec, found.pos, found.half, table=table, clearance=0.0))
    spec.path.write_text(scene_xml(spec, found.pos, found.half, table=table, clearance=CLEARANCE))
    robot, model = build_robot(spec.path)
    inflated = frames_hitting(robot, model, demo, "obstacle")
    print(f"clearance    : collision geom +{CLEARANCE*1000:.0f} mm per side "
          f"(planner sees {len(inflated)}/{n} blocked frames, visual geom unchanged)")
    print(f"validation   : {spec.true_path.relative_to(REPO)}  (true size, for --validation-xml)")

    lookat = np.array([path[:, 0].mean(), path[:, 1].mean(), path[:, 2].mean()])
    if cfg.view:
        import mujoco.viewer

        robot.set_joint_qpos(demo[hits[len(hits) // 2]])
        with mujoco.viewer.launch_passive(model, robot.data) as v:
            print("viewer open, close the window to continue")
            while v.is_running():
                v.sync()
    else:
        out = PREVIEW
        out.mkdir(parents=True, exist_ok=True)
        mid = hits[len(hits) // 2]
        img = out / f"{spec.key}.png"
        render_image(robot, model, demo, [0, mid // 2, mid, n - 1], lookat, img)
        print(f"preview      : {img.relative_to(REPO)}  (frames 0, {mid//2}, {mid}, {n-1})")
        if cfg.video:
            vid = out / f"{spec.key}.mp4"
            render_video(robot, model, demo, lookat, vid, cfg.fps)
            print(f"video        : {vid.relative_to(REPO)}  ({n} frames, {n/cfg.fps:.1f}s)")
    robot.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scene", choices=sorted(BY_KEY), help="build one scene")
    ap.add_argument("--all", action="store_true", help="build every scene")
    ap.add_argument("--margin", type=int, default=5,
                    help="frames at each end that must stay clear (default 5)")
    ap.add_argument("--bite", type=float, default=0.02,
                    help="how far the obstacle reaches past the wrist, m (default 0.02)")
    ap.add_argument("--min-frames", type=int, default=3,
                    help="frames the ceiling must block (wave only, default 3)")
    ap.add_argument("--view", action="store_true", help="open the viewer instead of a preview")
    ap.add_argument("--video", action="store_true", help="also render the demo as an mp4")
    ap.add_argument("--play", action="store_true",
                    help="render the demo through the scene as it stands; no search, no planning")
    ap.add_argument("--fps", type=float, default=50.0, help="video frame rate (demos are 50 Hz)")
    cfg = ap.parse_args()

    if cfg.all:
        specs = SCENES
    elif cfg.scene:
        specs = (BY_KEY[cfg.scene],)
    else:
        ap.error("pass --scene or --all")
    for spec in specs:
        (play if cfg.play else build)(spec, cfg)


if __name__ == "__main__":
    main()

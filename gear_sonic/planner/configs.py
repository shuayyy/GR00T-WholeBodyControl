"""Configs for the SONIC constrained whole-body planner and its ZMQ stream.

Velocity is explicit, never a hardcoded per-frame step: the deploy runs at a
fixed 50 Hz, so the step derives from ``max_joint_velocity * time_scale``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The C++ deploy steps streamed-motion references at 50 Hz
#: (gear_sonic_deploy .../localmotion_kplanner.hpp).  A deploy constant,
#: not a tuning knob.
DEPLOY_FPS = 50.0


@dataclass
class VelocityConfig:
    """How fast streamed trajectories move.  These are YOUR knobs."""

    max_joint_velocity: float = 0.5
    """Per-joint speed cap in rad/s (max-norm across the 29 joints)."""

    max_base_velocity: float = 0.25
    """Base translation speed cap in m/s."""

    max_base_angular_velocity: float = 0.5
    """Base rotation speed cap in rad/s."""

    time_scale: float = 1.0
    """Global slow-motion multiplier in (0, 1]: 0.5 = half speed.
    Effective caps are ``max_*_velocity * time_scale``."""

    def effective_joint_velocity(self) -> float:
        return self.max_joint_velocity * self.time_scale

    def effective_base_velocity(self) -> float:
        return self.max_base_velocity * self.time_scale

    def effective_base_angular_velocity(self) -> float:
        return self.max_base_angular_velocity * self.time_scale

    def validate(self) -> None:
        if not 0.0 < self.time_scale <= 1.0:
            raise ValueError(f"time_scale must be in (0, 1], got {self.time_scale}")
        if self.max_joint_velocity <= 0.0:
            raise ValueError("max_joint_velocity must be > 0")
        if self.max_base_velocity <= 0.0:
            raise ValueError("max_base_velocity must be > 0")
        if self.max_base_angular_velocity <= 0.0:
            raise ValueError("max_base_angular_velocity must be > 0")


@dataclass
class StreamConfig:
    """ZMQ streamed-motion transport to the C++ deploy."""

    zmq_bind: str = "*"
    """Interface the pose/command publisher binds on (deploy connects)."""

    zmq_port: int = 5556
    """Deploy's ZMQ input port (g1_deploy_onnx_ref default; its output uses 5557)."""

    pose_topic: str = "pose"
    command_topic: str = "command"

    chunk_frames: int = 50
    """Frames per pose message (50 = 1 s at deploy rate)."""

    realtime: bool = True
    """Pace chunks at the deploy rate (with a small lead) instead of
    sending the whole trajectory at once."""

    catch_up: bool = False
    """False disables the merger's MAX_GAP_FRAMES reset -- correct for a
    precomputed trajectory; True is for live sources that may fall behind."""


@dataclass
class PhaseRRTstarConfig:
    """PhaseRRTstar knobs, used only when planner_kind == "phaserrtstar".

    Values left at 0.0 are derived from the reference arclength by
    ``phase_defaults``; set one here to override.
    """

    reference_npz: str = str(
        Path(__file__).resolve().parent / "goal" / "wholebody_box_fixed.npz"
    )
    """Reference trajectory defining the phase coordinate.  Phase is only
    meaningful over the reference it was built from: planning a sub-segment
    requires slicing this and recomputing the arclength-scaled constants."""

    timeout: float = 100.0
    """Anytime budget [s].  The source study measured ~28-31 states in 100 s
    (delta 0.15 with a Python FK per geodesic step)."""

    validity_resolution: float = 0.05
    projection_delta: float = 0.15
    projection_lambda: float = 10.0
    rewire_factor: float = 0.5
    """Coarser than the plain planner by design -- every geodesic step is a
    Python FK round trip, and RRT*'s default neighbour count is unpayable."""

    quiet_base_sample_weights: bool = False
    """Use [0.1]*6 + [1.0]*29 sample weights: keep the reference's base in
    samples and let the projection handle the feet."""

    seed: int = 0
    """Seeds numpy only -- this OMPL binding exposes no RNG seeding, so seeds
    otherwise just label independent repeats."""

    d_alpha_min: float = 0.0
    d_alpha_max: float = 0.0
    sigma: float = 0.0
    uniform_fraction: float = -1.0
    w_alphastate: float = 0.0
    range: float = 0.0
    phase_grid: float = 0.0
    goal_bias: float = 0.0
    goal_threshold: float = 0.0
    """Overrides on phase_defaults; sentinel values mean "derive from L"
    (uniform_fraction uses -1.0 because 0.0 is its tuned value)."""

    def overrides(self) -> dict:
        """Non-sentinel fields, as a phase_params dict."""
        out = {}
        for key in ("d_alpha_min", "d_alpha_max", "sigma", "w_alphastate",
                    "range", "phase_grid", "goal_bias", "goal_threshold"):
            value = getattr(self, key)
            if value:
                out[key] = float(value)
        if self.uniform_fraction >= 0.0:
            out["uniform_fraction"] = float(self.uniform_fraction)
        return out


@dataclass
class PlannerConfig:
    """Constrained whole-body planning (feet pinned, CoM-stable)."""

    planner_kind: Literal["constrained", "phaserrtstar"] = "constrained"
    """Which planner to build.  "constrained" = ConstrainedOMPLPlanner
    (RRTConnect/AORRTC/RRTstar on the feet manifold, no reference).
    "phaserrtstar" = ConstrainedPhaseRRTstarPlanner, which searches
    (q, alpha) along a reference trajectory and REQUIRES one."""

    phase: "PhaseRRTstarConfig" = field(default_factory=lambda: PhaseRRTstarConfig())
    """Only read when planner_kind == "phaserrtstar"."""

    urdf_path: str = str(
        _REPO_ROOT
        / "decoupled_wbc"
        / "control"
        / "robot_model"
        / "model_data"
        / "g1"
        / "g1_29dof_with_hand.urdf"
    )
    """Free-flyer planning model source."""

    planner: str = "RRTConnect"
    """OMPL planner.  AORRTC corrupts trivially-connectable problems in
    this build (ConstrainedOMPLPlanner docstring); keep RRTConnect."""

    timeout: float = 10.0
    """Max seconds per OMPL solve."""

    com_margin: float = 0.005
    """Static-stability margin [m].  The wholebody_box_fixed reference dips to
    +0.0083 during the get-up, so 0.03 rejects the bend poses outright; 0.005
    admits the whole motion with ~3 mm of slack below the reference minimum.
    Use 0.03 for stand-to-stand problems, where the conservative standard
    costs nothing."""

    check_collisions: bool = True
    """Reject self-colliding states in the validity checker.  Costs a MuJoCo
    FK + broadphase per sampled state, so planning is slower."""

    collision_scene: str = ""
    """Scene to check against; empty = the sim's own scene_43dof.xml."""

    finger_closure: float = 0.0
    """Fingers are absent from the planning model, so the collision rig holds
    them at this fraction of the deploy's full-close pose.  0.0 (open) matches
    the wholebody_box_fixed reference and PhaseRRT's fixed-open hand model --
    at 0.9 the curled fingertips clip each other on the box-carry frames
    (157-292, 13 mm) even though the reference is clean.  Use 0.9 when
    planning grasp-style motions measured at that closure."""

    collision_tolerance: float = 1e-3
    """Penetration depth [m] below which a contact is ignored."""

    goal_threshold: float = 0.0075
    """Goal-region radius in the constrained space.  OMPL's setGoalState
    default is machine epsilon (no region), which is far below the manifold's
    own 1e-3 constraint tolerance.  Measured against the PROJECTED goal, so
    it does not absorb the projection correction reported as goal_error."""

    extend_range: float = 0.5
    validity_resolution: float = 0.01
    projection_delta: float = 0.05
    projection_lambda: float = 10.0

    smooth_path: bool = True
    shortcut_path: bool = True

    feet_retarget_tolerance: float = 1e-2
    """If a new start's feet poses deviate from the currently pinned feet
    targets by more than this, the planner is rebuilt around the new start
    (the manifold is defined by the reference feet poses)."""


@dataclass
class ServerConfig:
    """run_planner_server.py -- plan on request, stream to the deploy."""

    planning: PlannerConfig = field(default_factory=PlannerConfig)
    velocity: VelocityConfig = field(default_factory=VelocityConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)

    service_port: int = 5601
    """ZMQ REP port for plan requests (dict in / dict out, msgpack)."""

    reference_start_npz: str = str(Path(__file__).resolve().parent / "goal" / "start.npz")
    """Endpoint .npz used as the start when a request supplies none and
    robot state is unavailable."""

    ramp_duration: float = 5.0
    """Seconds for the smoothstep ramp from the robot's measured pose to a
    trajectory's first frame (``ramp_to_start`` / ``go_home``).  Streaming a
    plan whose start differs from the robot's pose is a step command -- the
    lunge measured as a ~60 deg elbow transient in tracking tests."""

    start_tolerance: float = 0.05
    """Max per-joint error [rad] accepted after a ramp before tracking."""

    ramp_leg_threshold: float = 0.3
    """Refuse to ramp if any leg joint deviates more than this [rad]:
    ramps are straight-line joint interpolation with no feet/CoM checking,
    safe for arm-dominant differences only.  Must sit ABOVE the policy's
    steady-state leg tracking error (~0.15 rad measured on a standing
    robot, or a matching stance can never ramp) and well below genuine
    stance changes (the deep bend differs by ~1.7 rad)."""

    read_robot_state: bool = True
    """Read the start configuration from the sim's DDS topics
    (rt/lowstate + rt/odostate).  Sim-only; requires the MuJoCo sim loop
    to be running."""

    dds_domain_id: int = 0
    dds_interface: str = "lo"
    """DDS channel settings, matching the sim's wbc yaml (DOMAIN_ID: 0,
    INTERFACE: "lo")."""

    sim_scene_xml: str = str(
        _REPO_ROOT
        / "gear_sonic"
        / "data"
        / "robot_model"
        / "model_data"
        / "g1"
        / "scene_43dof.xml"
    )
    """Scene the sim runs; its filtered joint order defines LowState motor
    order for the state reader."""

    verbose: bool = True

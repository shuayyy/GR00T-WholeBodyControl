"""One-shot plan request against the SONIC planner server.

Builds start/goal, fires one request, then drives the phased execution:
ramp to start, track, verify goal, go home.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import msgpack
import msgpack_numpy as mnp
import tyro
import zmq

from gear_sonic.planner.configs import ServerConfig

mnp.patch()

_PLANNER_DIR = Path(__file__).resolve().parent


@dataclass
class RequestConfig:
    start_npz: str = str(_PLANNER_DIR / "goal" / "start.npz")
    """Start endpoint .npz; empty string = let the server choose
    (robot state if enabled, else its reference start)."""

    goal_npz: str = str(_PLANNER_DIR / "goal" / "goal.npz")
    """Goal endpoint .npz."""

    execute: bool = True
    """Stream the planned trajectory to the deploy after planning."""

    host: str = "localhost"
    yes: bool = False
    """Skip all interactive prompts (unattended / remote runs)."""

    server: ServerConfig = field(default_factory=ServerConfig)
    """Only ``service_port`` and the request timeout are used."""


def _print_report(result: dict) -> None:
    if not result.get("ok"):
        raise RuntimeError(f"Planner error: {result.get('error')}")
    print(
        "Plan diagnostics:\n"
        f"  waypoints:            {result['num_waypoints']}\n"
        f"  frames @ 50 Hz:       {result['num_frames']} "
        f"({result['duration_s']:.2f} s)\n"
        f"  planning time:        {result['planning_time_s']:.2f} s\n"
        f"  peak joint velocity:  {result['peak_joint_velocity']:.3f} rad/s "
        f"(cap {result['joint_velocity_cap']:.3f})\n"
        f"  peak base velocity:   {result['peak_base_velocity']:.3f} m/s "
        f"(cap {result['base_velocity_cap']:.3f})\n"
        f"  endpoint errors:      start {result['start_error']:.2e}, "
        f"goal {result['goal_error']:.2e}\n"
        f"  max feet error:       {result['max_feet_error_frames']:.2e}\n"
        f"  plan stats:           {result['plan_stats']}\n"
        f"  executed:             {result['executed']}"
    )
    if result["peak_joint_velocity"] > result["joint_velocity_cap"] * 1.001:
        raise RuntimeError("Velocity cap violated in the built frames")


def build_request(config: RequestConfig) -> dict:
    request: dict = {
        "action": "plan",
        "goal": {"npz": config.goal_npz},
        "execute": config.execute,
    }
    request["start"] = {"npz": config.start_npz} if config.start_npz else None
    return request


def _confirm(prompt: str, auto: bool) -> bool:
    if auto:
        print(f"{prompt} [auto-yes]")
        return True
    return input(f"{prompt} [Enter=yes / n=skip] ").strip().lower() != "n"


def run_against_server(config: RequestConfig) -> int:
    """Phased interactive execution: plan -> ramp -> track -> home -> replay."""
    import time

    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, int((config.server.planning.timeout + 60) * 1000))
    socket.connect(f"tcp://{config.host}:{config.server.service_port}")

    def call(request: dict) -> dict:
        socket.send(msgpack.packb(request, default=mnp.encode))
        reply = msgpack.unpackb(socket.recv(), object_hook=mnp.decode)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error"))
        return reply

    try:
        result = call(build_request(config))
        _print_report(result)
        if not config.execute:
            return 0

        while True:
            ramp = call({"action": "ramp_to_start"})
            print(
                f"ramped to start in {ramp['duration_s']:.1f}s -- error "
                f"{ramp['achieved_error']:.3f} rad "
                f"({'OK' if ramp['within_tolerance'] else 'ABOVE tolerance'})"
            )
            if not _confirm("at start -- track the trajectory?", config.yes):
                break
            track = call({"action": "track"})
            print(f"tracking {track['num_frames']} frames ({track['duration_s']:.1f}s)...")
            time.sleep(track["duration_s"] + 1.5)
            goal = call({"action": "verify_goal"})
            print(
                f"goal reached -- max err {goal['goal_max_error']:.3f} rad, "
                f"RMSE {goal['goal_rmse']:.3f} rad"
            )
            if _confirm("go home?", config.yes):
                home = call({"action": "go_home"})
                print(f"home -- error {home['achieved_error']:.3f} rad")
            if config.yes or input("replay? [y/N] ").strip().lower() != "y":
                break
    finally:
        socket.close(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_against_server(tyro.cli(RequestConfig)))

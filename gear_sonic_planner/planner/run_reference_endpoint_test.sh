#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv_sim/bin/python"
OMPL_LIB="$REPO_ROOT/.venv_sim/lib/python3.10/site-packages/ompl"
TRAJECTORY="$SCRIPT_DIR/dataset/windex_l_place16.npz"
SERVER_SCRIPT="$SCRIPT_DIR/run_planner_server.py"
PREPARE_SCRIPT="$SCRIPT_DIR/prepare_trajectory_start.py"
MOTION_SCRIPT="$SCRIPT_DIR/run_comparison_motion.py"
OUTPUT_DIR="$REPO_ROOT/outputs/planner_reference_comparison"

planner="BITstar"
reference_mode="with"
transition_seconds="8.0"
home_position_tolerance="0.10"
planner_frequency="20.0"
initial_transition_time="2.0"
execution_margin="0.5"
server_pid=""

usage() {
  cat <<'EOF'
Interactive simulation test for trajectory endpoint planning.

The G1 control loop must already be running with --interface sim in another
terminal. The script slowly pre-positions the robot at qpos[0], then accepts:
P to plan with BIT*, D to replay the dataset, H to return to qpos[0], or Q to
quit. Every P/D motion saves the measured 17-joint upper-body trajectory.

Usage:
  run_reference_endpoint_test.sh --with-reference
  run_reference_endpoint_test.sh --without-reference

Options:
  --with-reference       Enable reference sampling and SimilarityObjective.
  --without-reference    Use the planner without reference sampling/cost.
  --planner NAME         OMPL planner name (default: BITstar).
  --transition-seconds N Slow pre-position duration (default: 8.0).
  -h, --help             Show this help.
EOF
}

# Interactive simulation test for trajectory endpoint planning.
#
# The G1 control loop must already be running in another terminal. This script:
#   1. starts the selected planner server;
#   2. directly interpolates the live robot to trajectory qpos[0];
#   3. accepts P for planning or D for direct dataset replay;
#   4. records measured upper-body qpos for every P/D run;
#   5. accepts H to return to qpos[0] and repeat.
#
# Usage:
#   run_reference_endpoint_test.sh --with-reference
#   run_reference_endpoint_test.sh --without-reference
#
# Options:
#   --with-reference       Enable reference sampling and SimilarityObjective.
#   --without-reference    Use the planner without reference sampling/cost.
#   --planner NAME         OMPL planner name (default: BITstar).
#   --transition-seconds N Slow pre-position duration (default: 8.0).
#   -h, --help             Show this help.

while (($#)); do
  case "$1" in
    --with-reference|--reference)
      reference_mode="with"
      shift
      ;;
    --without-reference|--no-reference)
      reference_mode="without"
      shift
      ;;
    --planner)
      if (($# < 2)); then
        echo "--planner requires a value" >&2
        exit 2
      fi
      planner="$2"
      shift 2
      ;;
    --transition-seconds)
      if (($# < 2)); then
        echo "--transition-seconds requires a value" >&2
        exit 2
      fi
      transition_seconds="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Virtualenv Python not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$TRAJECTORY" ]]; then
  echo "Trajectory not found: $TRAJECTORY" >&2
  exit 1
fi
if [[ ! -f "$MOTION_SCRIPT" ]]; then
  echo "Comparison motion script not found: $MOTION_SCRIPT" >&2
  exit 1
fi
if pgrep -f "[r]un_planner_server.py" >/dev/null; then
  echo "A planner server is already running." >&2
  echo "Stop it with Ctrl+C before starting this test." >&2
  exit 1
fi
mapfile -t control_processes < <(
  pgrep -af "[r]un_g1_control_loop.py" || true
)
if ((${#control_processes[@]} == 0)); then
  echo "The G1 control loop is not running." >&2
  echo "Start the simulation control loop in Terminal 1 first." >&2
  exit 1
fi
if ((${#control_processes[@]} != 1)); then
  echo "Expected exactly one G1 control loop, found ${#control_processes[@]}." >&2
  echo "Stop extra control loops before running this test." >&2
  exit 1
fi
control_pid="${control_processes[0]%% *}"

assert_control_loop_healthy() {
  if ! kill -0 "$control_pid" 2>/dev/null; then
    echo "The G1 control loop exited." >&2
    exit 1
  fi
  control_state="$(ps -o stat= -p "$control_pid" | tr -d '[:space:]')"
  if [[ -z "$control_state" || "$control_state" == T* || "$control_state" == Z* ]]; then
    echo "The G1 control loop is not running (process state: ${control_state:-missing})." >&2
    echo "Restart it in Terminal 1; do not leave it suspended with Ctrl+Z." >&2
    exit 1
  fi
}

if ! grep -Eq -- '--interface(=|[[:space:]])sim([[:space:]]|$)' \
  <<<"${control_processes[0]}"; then
  echo "Refusing to run: this endpoint test is simulation-only." >&2
  echo "Restart the control loop with explicit '--interface sim'." >&2
  exit 1
fi
if ! grep -Eq -- '--direct-waist-control([[:space:]]|$)' \
  <<<"${control_processes[0]}"; then
  echo "The G1 control loop is not using direct planner waist control." >&2
  echo "Restart it with '--interface sim --direct-waist-control'." >&2
  exit 1
fi
assert_control_loop_healthy

export LD_LIBRARY_PATH="$OMPL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1

cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -INT "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

reference_args=(--reference-trajectory-path "$TRAJECTORY")
if [[ "$reference_mode" == "with" ]]; then
  reference_args+=(--use-reference)
else
  reference_args+=(--no-use-reference)
fi

echo "WARNING: simulation only; collision checking is currently disabled."
echo "Planner: $planner"
echo "Reference objective: $reference_mode"
echo "Trajectory: $TRAJECTORY"
echo "Recorded trajectories: $OUTPUT_DIR"

move_to_trajectory_home() {
  assert_control_loop_healthy
  "$PYTHON_BIN" "$PREPARE_SCRIPT" \
    --trajectory-path "$TRAJECTORY" \
    --duration "$transition_seconds" \
    --position-tolerance "$home_position_tolerance"
}

run_recorded_motion() {
  local mode="$1"
  assert_control_loop_healthy
  "$PYTHON_BIN" "$MOTION_SCRIPT" \
    --mode "$mode" \
    --trajectory-path "$TRAJECTORY" \
    --output-dir "$OUTPUT_DIR" \
    --planner-name "$planner" \
    --reference-mode "$reference_mode" \
    --planner-frequency "$planner_frequency" \
    --initial-transition-time "$initial_transition_time" \
    --execution-margin "$execution_margin"
}

"$PYTHON_BIN" "$SERVER_SCRIPT" \
  --ompl-planner "$planner" \
  --planner-frequency "$planner_frequency" \
  --initial-transition-time "$initial_transition_time" \
  --no-visualize-planning \
  --no-smooth-path \
  --no-shortcut-path \
  --no-hold-final-pose \
  "${reference_args[@]}" &
server_pid=$!

sleep 2
if ! kill -0 "$server_pid" 2>/dev/null; then
  echo "Planner server exited during startup." >&2
  wait "$server_pid"
fi

move_to_trajectory_home
at_home=true

while true; do
  echo
  if [[ "$at_home" == true ]]; then
    echo "Ready at trajectory qpos[0]."
  else
    echo "Motion finished. Press H before another P/D comparison."
  fi
  read -r -p \
    "P = $planner plan, D = dataset replay, H = home, Q = quit: " \
    next_action
  case "${next_action,,}" in
    p)
      if [[ "$at_home" != true ]]; then
        echo "Press H first so both tests use the same start state."
        continue
      fi
      at_home=false
      if ! run_recorded_motion planner; then
        echo "Planner run failed; the interactive test remains active." >&2
      fi
      ;;
    d)
      if [[ "$at_home" != true ]]; then
        echo "Press H first so both tests use the same start state."
        continue
      fi
      at_home=false
      if ! run_recorded_motion dataset; then
        echo "Dataset replay failed; the interactive test remains active." >&2
      fi
      ;;
    h)
      move_to_trajectory_home
      at_home=true
      ;;
    q)
      exit 0
      ;;
    *)
      echo "Please enter P, D, H, or Q."
      ;;
  esac
done

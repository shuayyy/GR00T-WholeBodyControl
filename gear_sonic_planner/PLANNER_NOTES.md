# gear_sonic_planner — origin and changes

## Where this came from

`gear_sonic_planner/` is a copy of `gear_sonic/`. Everything except three
entries is unchanged:

- `planner/` — added
- `planner.zip` — added
- `reference/` — added

Not related to `gear_sonic_deploy/` (that's the C++ deployment tree; it has its
own unrelated `planner/`).

## Where `planner/` came from

`gear_sonic_planner/planner/` is a copy of
`decoupled_wbc/control/main/planner/`.

The `decoupled_wbc` copy is still there and still works. This one diverged.

## Changes in this copy of `planner/`

### Imports and paths

All intra-package imports repointed from `decoupled_wbc.control.main.planner.*`
to `gear_sonic_planner.planner.*`. Framework imports
(`decoupled_wbc.control.{constants,robot_model,utils,envs,policy}`) still point
at `decoupled_wbc` — that package is a dependency, not something that was
copied.

Touched for imports/paths only: `debug/view_qpos.py`,
`prepare_trajectory_start.py`, `run_comparison_motion.py`,
`run_g1_control_loop.py`, `run_reference_endpoint_test.sh` (also fixed
`REPO_ROOT` depth — the copy sits two levels shallower).

`simulation/robot.py` additionally lost its `sys.path.append` hack, since the
package is importable now.

### `utils/ompl_planning.py`

- `RightGoal` no longer hardcodes `GOAL_IDXS = [0,1,2,10,...,16]`. Goal indices
  are derived from the planner's joint names via `GOAL_JOINT_NAMES`, so it works
  in any planning space that contains those joints (17-DoF upper body, 29/35-DoF
  whole body, ...).
- `goal_type` accepts `"whole_body"` alongside `"upper_body"`; both mean a
  full-configuration goal in the planner's joint space.
- Start/goal DoF are validated against `n_dof` with an explicit error instead of
  silently mis-indexing.

### `configs/configs.py`

- `trajectory_path` default is now relative to the planner package directory,
  not the `decoupled_wbc` package.
- `goal_type` literal gained `"whole_body"`.

### New: `planner/constraints/`

Constraint layer for constrained whole-body planning. NumPy/Pinocchio only — no
OMPL, no MuJoCo, no planner imports.

- `embedding.py` — `PlanningEmbedder`, maps planning vector <-> full Pinocchio
  `q`. Handles the free-flyer base as 6 planning slots
  (`[position(3), rotation vector(3)]`); `idx_q` for configurations, `idx_v` for
  Jacobian columns.
- `base_constraint.py` — `Constraint` ABC (mirrors `ompl::base::Constraint`) plus
  a central-difference `numeric_jacobian` used to validate every analytic one.
- `feet_constraint.py` — 12 rows, both feet pinned to reference poses.
  `log6(target^-1 · current)`, analytic Jacobian in `pin.LOCAL`.
- `com_constraint.py` — 1 row, hinge on the static stability margin. Exactly 0.0
  when the CoM is inside the support polygon.
- `composable_constraint.py` — stacks constraints; `per_constraint_error()` for
  debugging which one isn't converging.
- `planner_constraints.py`, `stability_visualizer.py` — pre-existing FK/CoM/hull
  backend, unmodified.
- `test_constraints.py` — 55 tests, runs without OMPL/MuJoCo.

### New: `planner/utils/constrained_planning.py`

`ConstrainedOMPLPlanner` — plans in `ProjectedStateSpace` over the feet
constraint (35-D ambient, 12 constraint rows -> 23-D manifold). CoM stability
lives in the validity checker, not in the constraint, because the hinge defines a
region rather than a surface and would corrupt OMPL's manifold-dimension
bookkeeping.

No optimization objective — this is start-to-goal planning only, no similarity
cost.

Workarounds for two defects in the local OMPL 2.0.0 build, both documented in
the file:

1. `ProjectingStateSampler` — the build's sampler returns raw ambient samples
   instead of projecting them onto the manifold. Naive projection doesn't fix
   it either: a projected uniform 35-D sample virtually never respects the
   joint limits, and off-manifold or out-of-bounds targets make every tree
   extension collapse onto its own tree node (measured 6–12 vertices in 240 s,
   trees effectively never grow). The sampler therefore perturbs *anchors* —
   known-feasible configurations (reference posture, current start/goal) —
   projects, and accepts only in-bounds results. Acceptance ~83 %, and solve
   times went from tens of seconds (or never) to well under a second.
2. Corrupted-path rejection — **AORRTC bug** (`AOXRRTConnect.cpp` L360-373 in
   the build's source): its first iteration primes a straight-line connect
   while the bookkeeping still points at the goal tree's root, so when the
   direct start→goal connect succeeds the returned path is built from the
   goal tree only — `[goal, goal]`, length 0, start never appears. Such a
   path carries no route information, so `plan()` raises `RuntimeError`
   naming the planner and the symptom. There is no geodesic fallback;
   `degenerate_path_recovered` in `last_plan_stats` is kept for downstream
   readers but is always `False` when `plan()` returns.

Prefer RRTConnect (the default). AORRTC hits bug 2 on every trivially
connectable problem, and its `getPlannerData()` is wiped by its own anytime
reset, so `planner_vertices` in `last_plan_stats` only means something for
single-shot planners.

`last_plan_stats` also records `planner_vertices` / `planner_edges` — on a
problem whose direct geodesic is invalid, real search shows as vertices > 2,
`degenerate_path_recovered=False`, and run-to-run varying path lengths
(verified: 5/5 runs, 13–22 vertices, lengths 3.228–3.318).

Collision checking is still disabled, carried over from `ompl_planning.py` with
its original comment. Re-enable before hardware.

### New data directories

- `planner/goal/` — `start.npz`, `goal.npz`, `original_start.npz`,
  `original_goal.npz` (frames 0 and 122, standing -> deep squat);
  `yaw_start.npz`, `yaw_goal.npz` (engineered CoM-valley pair: arms forward,
  waist yaw +1.6 -> -1.6 — the direct geodesic is unstable at
  `com_margin=0.038` while both endpoints are fine, so it actually requires
  search; the only pair here that does, since collision is off).
- `planner/test_traj/` — planned trajectories per problem
  (`original/`, `grasp/`, `yaw_swap/`), `npz/` plus deploy-ready `csv/`.
  See `test_traj/commands.md`.

## Planner settings

Defaults are `RRTConnect` with `extend_range=0.5, projection_lambda=10.0`;
both dataset problems and the yaw-swap problem solve in under a second per
plan. With the fixed sampler, OMPL auto-range also works — the small range is
kept as the deliberate default. The old note that RRTConnect needed 5-60 s
per run predates the sampler fix and is obsolete. AORRTC raises
`RuntimeError` on trivially connectable pairs (bug 2 above).

## Known data issue

`waist_pitch` in the reference trajectories reaches 0.5218 rad, above the URDF
limit of 0.52 (frames 88-213). Goals in that range are rejected on bounds unless
clamped. Fix at export time.

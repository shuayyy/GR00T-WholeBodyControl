# Similarity Planner

Reference-guided motion planning: instead of asking OMPL for the *shortest* path
between two configurations, ask for the path that most closely resembles a
recorded human demonstration.

All code lives in [`utils/ompl_planning.py`](utils/ompl_planning.py).

---

## Quick start

Terminal 1 — simulation control loop:

```bash
source .venv_sim/bin/activate
python decoupled_wbc/control/main/planner/run_g1_control_loop.py \
    --interface sim --direct-waist-control
```

Terminal 2 — interactive A/B test:

```bash
decoupled_wbc/control/main/planner/run_reference_endpoint_test.sh --with-reference
```

The script pre-positions the robot at the demo's frame 0, then accepts:

| Key | Action |
|-----|--------|
| `D` | Replay the recorded demo — this is the ground truth |
| `P` | Plan from the same start to the demo's end pose, and execute |
| `H` | Return to frame 0 so both runs share a start state |
| `Q` | Quit |

Every `P`/`D` run writes the measured 17-joint trajectory to
`outputs/planner_reference_comparison/*.npz`, so the two can be compared offline.
Use `--without-reference` for the baseline (plain path-length objective).

---

## The idea

A planner compares candidate paths by **score**, and returns the lowest-scoring
one. Changing the scoring rule changes the answer:

| Objective | Scores a path by | You get |
|-----------|------------------|---------|
| `PathLengthOptimizationObjective` (default) | its length | a straight line |
| `SimilarityObjective` (this) | how far it strays from the demo | a demo-shaped path |

**Lawn analogy.** The demo is a worn footpath across a lawn. Walking *on* the
footpath is free. Walking on the grass fines you *how far off you are* × *how far
you walk*. "Find the cheapest route from A to B" then means "stay on the
footpath".

Two independent levers implement this, both set in
[`OMPLGeometricPlanner.plan()`](utils/ompl_planning.py#L142-L149):

```python
if ref_traj is not None:
    # Lever 1 — where the planner LOOKS
    self.ss.getStateSpace().setStateSamplerAllocator(
        lambda space: RefStateSampler(space, ref_traj)
    )
    # Lever 2 — what the planner PREFERS
    objective = SimilarityObjective(self.si, ref_traj, ref_weights)
    self.pdef.setOptimizationObjective(objective)
```

`ref_traj` is the demo as an `(N, 17)` array of joint angles — 200 frames for
`dataset/windex_l_place16.npz`.

---

## Lever 2 — `SimilarityObjective` (the score)

[`utils/ompl_planning.py:377`](utils/ompl_planning.py#L377)

### `stateCost(q)` — the fine at one pose

[`utils/ompl_planning.py:398`](utils/ompl_planning.py#L398)

**In:** one pose (17 joint angles). **Out:** one number.

```python
diff      = self.weighted_ref_traj - weighted_config   # compare against ALL demo frames
dists_sq  = np.sum(diff * diff, axis=1)                # distance to each one
min_dist  = np.sqrt(np.min(dists_sq))                  # keep the CLOSEST
return ob.Cost(float(min_dist))
```

That is: **Euclidean distance to the nearest demo frame.** Zero on the demo,
growing as you move away.

`weights` defaults to all-ones, and `run_planner_server.py` never passes
`ref_weights` ([`run_planner_server.py:297-305`](run_planner_server.py#L297-L305)),
so every joint currently counts equally.

Measured along the straight line from start to goal:

```
step  stateCost   nearest demo frame
   0     0.0000   frame   0     <- on the demo (it is the start)
   2     0.0866   frame   0     ########
   4     0.1645   frame  14     ################
   5     0.2030   frame  14     ####################   <- furthest off the footpath
   7     0.1297   frame 198     ############
  10     0.0000   frame 199     <- on the demo (it is the goal)
```

> **`min` has no sense of order.** Between steps 5 and 6 the nearest frame jumps
> `14 → 198` and the cost does not notice. This is a *proximity* rule ("am I near
> the demo?"), not a *path-following* rule ("am I at the right point along it?").
> See [Known issue 4](#4-the-cost-has-no-notion-of-progress).

### `motionCost(s1, s2)` — the fine over one edge

[`utils/ompl_planning.py:326`](utils/ompl_planning.py#L326), inherited from
`StateCostIntegralObjective`.

`stateCost` scores a point, but an edge is a continuous move — so integrate along it:

```
motionCost = ∫ stateCost(q(s)) ds
```

Numerically, by the trapezoid rule:

1. Split the edge into `nd` sub-segments (`nd = validSegmentCount(s1, s2)`)
2. Each sub-segment costs `0.5 × length × (cost_left + cost_right)`
3. Sum them

Multiplying by length is what makes this an *area* rather than a bare number —
it is why a long detour off the demo costs more than a short one.

**Worked example** — the actual edge from the run above (`nd = 3`, so 4 states):

```
state 0   t=0.000   stateCost = 0.0000     <- prev_cost, primed before the loop
state 1   t=0.333   stateCost = 0.1404
state 2   t=0.667   stateCost = 0.1441
state 3   t=1.000   stateCost = 0.0000
```

| Segment | Calculation | Cost | Running total |
|---------|-------------|------|---------------|
| 1 | 0.5 × 0.1443 × (0.0000 + 0.1404) | 0.0101 | 0.0101 |
| 2 | 0.5 × 0.1443 × (0.1404 + 0.1441) | 0.0205 | 0.0307 |
| 3 | 0.5 × 0.1443 × (0.1441 + 0.0000) | 0.0104 | **0.0411** |

`motionCost = 0.0411` — exactly the solution cost OMPL printed.

These 4 states are **scratch** (`temp1`, `temp2`). They are scored, summed, and
discarded. They are *not* path waypoints.

### How many states per edge

```
nd     = ceil(edge_distance / longestValidSegment)
states = nd + 1                                     # fence-post: 3 segments, 4 posts
longestValidSegment = validity_resolution × maximumExtent
```

`maximumExtent` is the **diagonal of the whole 17-D joint box**, not 1.0:

| Quantity | Value |
|----------|-------|
| `maximumExtent` | 16.5167 rad |
| `validity_resolution` | 0.01 |
| `longestValidSegment` | **0.1652 rad** (≈ 9.5°) |
| Edge distance in the example | 0.4330 rad |
| `nd` | `ceil(0.4330 / 0.1652)` = **3** |

See [Known issue 3](#3-validity-resolution-is-far-too-coarse).

### Path cost

Path cost is the sum of `motionCost` over every edge:

```
A ──edge1──> B ──edge2──> C ──edge3──> D      path cost = 0.012 + 0.008 + 0.015
   0.012        0.008        0.015                      = 0.035
```

**The demo itself scores exactly `0.000000`** — every point sits on the reference,
so the integrand is zero everywhere. The demo is therefore the **global optimum**
of this objective. Verified numerically:

| Path | Length | Cost |
|------|--------|------|
| Straight line start → goal | 0.433 | 0.0461 |
| The demo | 1.639 | **0.000000** |

The demo is 3.8× longer and still wins, because the score is deviation, not
length. That inversion is the whole point: it lets the planner "pay" extra travel
to hug the demo. With a path-length objective, adding vertices always makes a
path worse; here it can make it better.

---

## Lever 1 — `RefStateSampler` (where to look)

[`utils/ompl_planning.py:177`](utils/ompl_planning.py#L177)

A correct objective is not enough. Optimizing planners improve by throwing random
poses into the space and trying to route the path through them. Uniform sampling
over 17 dimensions will essentially never land near the demo, so the planner would
never find the low-cost region. This sampler puts the samples where they matter.

**Setup** ([`:206-239`](utils/ompl_planning.py#L206-L239)) — turn the discrete
demo into a continuous curve:

1. Drop duplicate frames
2. Arc-length parameterize → `u ∈ [0, 1]` (measures *distance travelled*, not time)
3. Fit a cubic B-spline → `q_ref(t)`, a smooth 17-D curve

**Sampling** ([`:290`](utils/ompl_planning.py#L290)):

```
10%  →  uniform over the whole 17-D joint box     (preserves completeness)
90%  →  q_ref(t) + Gaussian noise (σ = 0.3 rad)   (a "fat noodle" around the demo)
          ├─ 50%: t is progress-biased — drifts with sample_count through a
          │        1000-sample cycle, sweeping the demo start → end
          └─ 50%: t uniform random
```

The 10% uniform fraction matters: without it the planner could never find a
solution requiring a departure from the tube.

---

## Measured results

From `outputs/planner_reference_comparison/`, comparing a `P` run against the
`D` run that preceded it.

**Endpoints — excellent:**

| | max joint error | L2 |
|---|---|---|
| Start | 0.0007 rad | 0.0015 |
| Goal | **0.0137 rad (0.8°)** | 0.0143 |

**Path shape — no resemblance:**

| Metric | Value |
|--------|-------|
| Demo path length | 1.639 rad |
| Planner path length | 0.447 rad |
| Straight line start → goal | 0.434 rad |
| **Planner / straight line** | **1.03** ← it *is* the straight line |
| Demo / straight line | 3.78 |

Arc-length-aligned deviation: mean L2 **0.28 rad**, max **0.52 rad** at 44% along
the path. Worst joint `right_elbow_joint`, max **0.47 rad (27°)**.

Range collapse tells the same story:

| Joint | Demo range | Planner range |
|-------|------------|---------------|
| `right_elbow_joint` | 0.637 | 0.213 |
| `right_wrist_roll_joint` | 0.191 | **0.017** |
| `waist_pitch_joint` | 0.136 | **0.001** |

**The objective is correct; the search never used it.** See below.

---

## Known issues

### 1. BIT* silently discards `RefStateSampler`

`setStateSamplerAllocator` only affects `si->allocStateSampler()`. BIT* does not
call that — it draws samples from `objective->allocInformedStateSampler()`
(`bitstar/ImplicitGraph.cpp:579`). `SimilarityObjective` does not override that
method, so OMPL falls back to `RejectionInfSampler`, whose base class does:

```cpp
baseSampler_ = StateSampler::space_->allocDefaultStateSampler();   // InformedStateSampler.cpp:150
```

`allocDefaultStateSampler()` **ignores the custom allocator** — that is the point
of "Default" in the name. So every sample is uniform over the full 17-D box.

Symptom in the log:

```
Found a solution of cost 0.0411 (2 vertices) from 0 samples ...
Finished ... from 16300 samples ... 0 rewirings. The final graph has 2 vertices.
```

`0 rewirings` and `2 vertices` after 16300 samples: not one landed usefully close
to the demo.

**Affected planners:** BIT*, AIT*, EIT*, and RRT* *only if* `setInformedSampling`
or `setRejectionSampling` is enabled. Plain RRT*, RRTConnect, and PRM* use
`si_->allocStateSampler()` (`RRTstar.cpp:1115`) and **do** honour the custom
sampler.

**Quick test:** run with `--planner RRTstar`. If the path suddenly hugs the demo,
this is confirmed.

**Proper fix:** override `allocInformedStateSampler` on `SimilarityObjective` to
return a reference-biased informed sampler.

### 2. No cost-to-go heuristic

This is the warning OMPL prints:

```
RejectionInfSampler: The optimization objective does not have a cost-to-go heuristic
defined. Informed sampling will likely have little to no effect.
```

`SimilarityObjective` inherits the default `costToGo` ≡ 0, so BIT*'s informed
pruning and edge-queue ordering have no signal. Define `costToGo` and
`motionCostHeuristic`.

### 3. Validity resolution is far too coarse

`validity_resolution = 0.01` reads like "1% steps", but it is 1% of the **16.5 rad
space diagonal** = 0.165 rad ≈ 9.5° chunks.

Cost error, measured on the example edge:

| Segments | Cost |
|----------|------|
| **3** (what OMPL used) | **0.0411** |
| 100 | 0.0461 |
| 2000 | 0.0461 (true value) |

The cost is under-reported by ~11%, making deviant paths look better than they are.

**The larger risk:** the same `nd` drives **collision checking**. Once collision
checking is re-enabled, a 0.43 rad arm sweep is checked at only 3 intermediate
poses — the arm can pass through an obstacle unnoticed.

Suggested: `validity_resolution = 0.001` (≈ 0.0165 rad ≈ 1°).

### 4. The cost has no notion of progress

`stateCost` takes the `min` over all demo frames, so it only asks *"am I near the
demo?"* — never *"am I at the right point along it?"* All of the following score
**0.0**, identical to executing the demo perfectly:

- Jumping straight to the end pose and sitting there
- Running the demo backwards
- Zig-zagging between frame 10 and frame 190
- Visiting 3 of the 200 frames and skipping the rest

**Fix:** make the cost depend on phase. Track a monotonically increasing `t` along
the demo and penalize deviation from `q_ref(t)` *at that phase*, rather than the
minimum over all frames. Then backtracking and skipping ahead both cost.

### 5. Collision checking is disabled

[`utils/ompl_planning.py:97-104`](utils/ompl_planning.py#L97-L104) — `in_contact`
is hard-coded to `False`:

```python
# TEMPORARILY DISABLED FOR SIMULATION TESTING ONLY.
# Restore this before real-robot deployment; otherwise OMPL may return
# paths that collide with the robot or the environment.
```

This makes the direct start→goal edge trivially valid, so it is always found on
iteration 1 and becomes the incumbent immediately. `run_reference_endpoint_test.sh`
enforces `--interface sim` and refuses to run otherwise.

**Must be restored before any real-robot use.**

---

## File map

| File | Role |
|------|------|
| [`utils/ompl_planning.py`](utils/ompl_planning.py) | `OMPLGeometricPlanner`, `SimilarityObjective`, `RefStateSampler`, `StateCostIntegralObjective` |
| [`run_planner_server.py`](run_planner_server.py) | ROS service wrapper; loads the demo ([`:469`](run_planner_server.py#L469)), calls `plan()` ([`:297`](run_planner_server.py#L297)) |
| [`run_reference_endpoint_test.sh`](run_reference_endpoint_test.sh) | Interactive A/B harness (`P`/`D`/`H`/`Q`) |
| [`run_comparison_motion.py`](run_comparison_motion.py) | Executes one `P` or `D` motion and records measured qpos |
| [`prepare_trajectory_start.py`](prepare_trajectory_start.py) | Slow, safe pre-positioning to demo frame 0 |
| `dataset/windex_l_place16.npz` | The reference demo (200 frames × 17 joints) |
| `outputs/planner_reference_comparison/` | Recorded `*_planner.npz` / `*_dataset.npz` runs |

### Recording format

Each `.npz` contains `measured_qpos (T, 17)`, `commanded_qpos`, `joint_names`,
`timestamps`, `wall_timestamps`, `fps`, plus metadata (`mode`, `planner`,
`reference_mode`, `source_trajectory`, `completed`).

Compare two runs by resampling both on normalized arc length — the durations
differ (demo ≈ 7.2 s, planner ≈ 3.2 s), so a plain index-wise diff is meaningless.

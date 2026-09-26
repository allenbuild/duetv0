# Playground stage `gaze_proxy`: head-forward as a stand-in for gaze, validated on CoMind

Date: 2026-09-26. Code: `src/duet/playground/gaze_proxy.py`, `scripts/eval_gaze_proxy_comind.py`, tests in
`tests/test_playground_gaze_proxy.py`. All numbers below were produced by the commands quoted next to them; nothing is
copied from elsewhere.

## What the stage does

No playground rig has an eye tracker, so the stage uses the ego camera's optical axis as the gaze ray. Two paths, chosen
per frame:

| path | when | gaze ray | partner head | objects |
|---|---|---|---|---|
| 3D (`method = "headpose"`) | `derived/headpose/<ego>.npz` has a finite `T_world_cam` for the frame | camera centre along the camera +Z column (OpenCV) | `world3d/head_<partner>`, else the partner's own camera centre | `world3d/object_<name>` (tag positions), else straight from `derived/tags` + `rig.json` `object_tags` |
| 2D (`method = "image"`) | no head pose for the frame | image centre | mean of the COCO nose/eye keypoints (conf >= 0.3) of the best person slot in this ego's `body2d`; angular offset from the optical axis through the calib intrinsics (nominal 110 deg pinhole when the board was never seen) | YOLO-World boxes in the ego frame (box centre) |

Outputs per ego stream, per common-timeline frame (`derived/gaze_proxy/<ego>.npz` and `records_extra.parquet`, merged
into `records.parquet` with the prefix `gaze_proxy_`): `angle_to_partner_deg`, `looking_at_partner` (angle < 15 deg),
`angle_to_nearest_object_deg`, `nearest_object`, `looking_at_object` (name if < 15 deg, else ""), `method`, and
`mutual_gaze` (both egos looking at each other). The status line reports which path each stream used and its coverage.

On `comind_43276420_clip` (Aria + GoPro, no board, so no head pose and no world frame) the stage runs entirely on the 2D
path:

```
.venv/bin/python scripts/playground.py run comind_43276420_clip --stages gaze_proxy --force
  gaze_proxy: done leader: image, partner seen 67%, looking 0%; helper: image, partner seen 31%, looking 0%; mutual 0%
```

Partner-head angles on that clip: leader median 53 deg (p10-p90 44-56), helper median 50 deg (47-54); minimum 37 / 28
deg. The two are cooking side by side at a counter and the partner's head sits in the top corner of the 110 deg fisheye
image while the wearer looks at the pan (checked visually on frames 294 and 381 of the leader stream), so 0 % "looking
at partner" is the right answer for this clip, not a failure of the path. Objects: `angle_to_nearest_object_deg` is
finite on 98 % / 97 % of frames.

Known approximations of the 2D path: the pinhole angle model is applied to a fisheye image (Aria: 608 px equidistant
focal on 1408 px, i.e. f = 0.432 W, vs the nominal 110 deg pinhole f = 0.350 W), which over-states off-axis angles by a
few degrees at 15 deg (72 px vs 60 px on a 640 px frame) and more towards the edge; and the partner's head must be inside
the ego image at all.

## Validation on CoMind (real eye tracking vs the head-forward ray)

CoMind has Project Aria eye gaze (yaw/pitch in CPF) and, for 11 of the 44 labelled recordings, Multi-SLAM poses of both
wearers in one world frame. `scripts/eval_gaze_proxy_comind.py` reuses `shared_world_features` (CPF -> device via
`T_Device_CPF`, `load_poses`, `rgb_forward_axis`) and the kinematics caches; nothing is re-derived.

Head-forward = the camera-rgb optical axis, read from the first record of each recording's
`mps_<role>_trimmed_vrs/slam/online_calibration.jsonl` when present (74 of 88 role-recordings) and otherwise the nominal
axis `NOMINAL_RGB_AXIS_DEVICE` (38.7 deg off device +Z). The per-recording axes deviate from the nominal one by a median
0.18 deg (mean 0.45, p90 1.11, n = 74), so the nominal axis is a safe fallback.

Command (parts a and b; 44 recordings, 4.6 M valid gaze frames, gaze validity 99.7 %; 8.8 min, almost all of it reading
the online-calibration files; run twice, identical numbers):

```
.venv/bin/python scripts/eval_gaze_proxy_comind.py --out <scratch>/gaze_proxy_eval_ab.json
```

### (a) Angle between the real gaze direction and the head-forward ray (device frame)

| frames | role | n | median | mean | p90 | share < 15 deg |
|---|---|---|---|---|---|---|
| all | leader | 2,303,900 | 18.6 | 18.2 | 27.7 | 34 % |
| all | helper | 2,292,471 | 17.6 | 17.5 | 27.4 | 38 % |
| all | both | 4,596,371 | 18.1 | 17.9 | 27.6 | 36 % |
| inside annotated joint attention | leader | 345,841 | 17.9 | 17.6 | 27.0 | 37 % |
| inside annotated joint attention | helper | 345,846 | 16.3 | 16.4 | 26.0 | 44 % |
| inside annotated joint attention | both | 691,687 | 17.1 | 17.0 | 26.5 | 40 % |

Per-recording medians: leader 18.7 +- 4.2 deg (joint attention 17.8 +- 4.8), helper 17.9 +- 5.4 (16.8 +- 5.3), mean +- sd
over 44 recordings. So the eye is typically 17-19 deg away from where the camera points, and only marginally closer
during joint attention. Whether this is a fixed bias (people look below the camera axis) or scatter is in the
"mean gaze axis" block below.

**Mean gaze axis (bias vs scatter).** Pooling all valid gaze directions per role in the device frame and normalising the
mean gives the axis people actually look along on average:

| role | mean gaze axis (device) | angle to RGB axis | angle to device +Z | residual gaze-vs-mean-axis median / mean / p90 | share < 15 deg | residual inside joint attention median / p90 |
|---|---|---|---|---|---|---|
| leader | (0.340, -0.559, 0.756) | 15.0 deg | 40.9 deg | 9.7 / 11.1 / 20.2 | 77 % | 9.4 / 17.9 |
| helper | (0.324, -0.576, 0.751) | 13.9 deg | 41.4 deg | 10.2 / 11.6 / 21.3 | 75 % | 10.0 / 19.3 |

Expressed in CPF (+Z forward) the mean gaze has yaw 0.3 / -0.7 deg and pitch -22.3 / -21.3 deg (leader / helper), while
the RGB axis sits at yaw -1.2, pitch -7.4 deg: the 15 deg offset is entirely pitch, in the same direction the glasses
already tilt the camera, with no yaw component. So of the 18 deg median gaze-vs-head-forward angle, ~15 deg is a fixed
"people look further down than the camera points" bias and ~10 deg (median) is true eye-in-head scatter. For
reference, device +Z (what one would use without the RGB calibration) is 42 deg from the mean gaze; the RGB axis 15 deg;
a pitched-down "mean gaze" axis 10 deg. The playground stage uses the camera axis (no per-wearer offset is applied
because the playground rigs are not Aria and this bias was measured on Aria wearers doing a counter task); a 15 deg
downward pitch of the ray is the obvious first calibration if a rig turns out to match.

### (b) Joint attention from `angle(ray, partner head)`: head-forward vs real gaze

11 recordings with both Multi-SLAM trajectories (the `world_v0.npz` subset), frames where both poses and both gazes are
valid (100 % of frames in 10 recordings, 75 % in one; joint-attention prevalence 4-25 %). Score = -angle; AUROC of the
per-frame `ja_active` label. The recomputed forward angle matches the cached cross-person block (`world_v0.npz` cols
29/30) to <= 0.08 deg on every recording, so the stage's 3D path and the benchmark use the same quantity.

| score | per-recording AUROC mean +- sd | pooled AUROC |
|---|---|---|
| head-forward, leader | 0.648 +- 0.164 | 0.603 |
| head-forward, helper | 0.646 +- 0.103 | 0.639 |
| head-forward, min over both people | **0.788 +- 0.095** | **0.754** |
| real gaze, leader | 0.630 +- 0.155 | 0.580 |
| real gaze, helper | 0.629 +- 0.106 | 0.631 |
| real gaze, min over both people | 0.776 +- 0.091 | 0.747 |

Per recording (forward_min / gaze_min): 0.779/0.766, 0.703/0.700, 0.773/0.772, 0.852/0.823, 0.904/0.870, 0.905/0.888,
0.833/0.821, 0.780/0.788, 0.823/0.814, 0.768/0.757, 0.550/0.535. The head-forward ray is at least as predictive of
the joint-attention label as the measured gaze in 9 of 11 recordings; the two never differ by more than 0.034. For this
label, then, head pose is not a degraded substitute for eye tracking; eye-in-head rotation adds nothing beyond what the
head already says, at least at the "angle to the partner's head" level. (Joint attention in CoMind is attention to a
shared object, not mutual gaze: the median angle to the partner's head is 60-95 deg, and a single person's angle is only
weakly informative, 0.63-0.65; the minimum over both people is what carries the signal.)

### (c) The GBDT benchmark without gaze

What a rig without an eye tracker loses on the benchmark tasks. `scripts/baseline_gbdt.py` is not edited; the script
imports its `person_base` / `featurize` / `STRIDE` and replicates `main()`'s folds (rng seed 0), trees and 5 Hz rows,
then zeroes the gaze columns of the per-person block (`person_base` cols 15-18: gaze point xyz + gaze valid) and, on the
11-recording paired subset, also the gaze-derived columns of the cross-person block (`world_v0` cols 25-28: gaze point
to partner wrists / head), keeping the head-forward cosines (cols 29-30). Speech on, no objects, i.e. the
`baseline_gbdt_grasp.json` / `baseline_gbdt_world_cpf.json` configurations. 41 min:

```
.venv/bin/python scripts/eval_gaze_proxy_comind.py --skip-ab --gbdt --out <scratch>/gaze_proxy_eval_c.json
```

| recordings | view | condition | handover in progress | joint attention |
|---|---|---|---|---|
| 44 | helper only | with gaze (reproduction; stored json: 0.632 +- 0.042 / 0.670 +- 0.054) | 0.627 +- 0.042 | 0.671 +- 0.056 |
| 44 | helper only | gaze zeroed | 0.611 +- 0.023 | 0.661 +- 0.036 |
| 44 | both, paired | with gaze (stored json: 0.637 +- 0.045 / 0.703 +- 0.057) | 0.640 +- 0.045 | 0.704 +- 0.055 |
| 44 | both, paired | gaze zeroed | 0.614 +- 0.050 | 0.706 +- 0.043 |
| 11 (world) | both + cross-person block | with gaze (stored json: 0.586 +- 0.019 / 0.834 +- 0.110) | 0.586 +- 0.019 | 0.834 +- 0.110 |
| 11 (world) | both + cross-person block | all gaze zeroed, head-forward cosines kept | 0.583 +- 0.025 | 0.816 +- 0.110 |

Per fold, joint attention, 44 both: with gaze 0.785, 0.749, 0.639, 0.687, 0.660; zeroed 0.764, 0.716, 0.642, 0.734,
0.675. Handover, 44 both: 0.660, 0.694, 0.580, 0.673, 0.592 vs 0.647, 0.614, 0.536, 0.684, 0.591. World subset, joint
attention: 0.900, 0.910, 0.617, 0.855, 0.886 vs 0.898, 0.869, 0.599, 0.844, 0.868.

Reading: removing gaze costs 0.026 AUROC on handover-in-progress (paired view) and nothing on joint attention (0.704 ->
0.706 in the paired view, 0.671 -> 0.661 helper-only); on the paired subset with head-forward geometry, joint attention
drops from 0.834 to 0.816 when the gaze-point features go. Head-forward geometry carries almost all of what gaze
contributed to these labels; the handover task is where the eye tracker earns its 0.02-0.03. The 44-recording
reproductions differ from the stored JSONs by <= 0.005 (thread-level nondeterminism of the tree fits; the 11-recording
run reproduces exactly).

## What was not verified

- The 3D path on real data: no playground episode with a head pose exists yet (the CoMind clip has none), so the 3D
  path is covered only by the synthetic test (`tests/test_playground_gaze_proxy.py::test_headpose_path`: two posed
  egos, a tagged object, angles checked to 1e-4 deg).
- `looking_at_partner` / `mutual_gaze` against ground truth: CoMind has no mutual-gaze annotation, and in the clip the
  partner never comes within 15 deg. The 15 deg threshold is a choice, not a fitted value; part (a) says that real gaze
  sits ~18 deg from the head ray at the median, so a 15 deg gate on the head ray is tight relative to eye-in-head
  motion.
- The 2D angle model on non-Aria lenses (GoPro / ZED egos).

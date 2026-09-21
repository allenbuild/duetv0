# Duet capture specification v0 (paired human-human interaction)

Derived from the Lawrence, Kaichen, Haoyu and Sebastian conversations and from what the CoMind processing pipeline needed. This is what we record when we start collecting our own data; the file layout mirrors CoMind so the existing adapters run unchanged.

## Sensors per participant (2 participants minimum)

| stream | rate | why | source of the requirement |
|---|---|---|---|
| egocentric RGB (fisheye ok) | 30 fps | video for labels, later object perception | everyone |
| depth or stereo | ≥ 10 fps | 3D reconstruction is where the field is moving | Lawrence, Haoyu |
| 3D hand pose, 21 landmarks per hand, with per-hand confidence and explicit missing sentinel | 30 Hz | "the most important thing" | Kaichen, Lawrence |
| head pose (SLAM), per-participant and collaborative | ≥ 30 Hz | shared world frame; projecting partner's hands into my view | Lawrence |
| gaze (3D point) | ≥ 10 Hz | joint attention, intent | our results (+0.08 from geometry) |
| audio, word-level ASR with speaker identity | – | operator product; anticipation cue arrives 3 s before the reach | our measurement; Kaichen says not for robots |
| optional: wrist camera | 30 fps | manipulation detail | Haoyu |
| optional: one exocentric RGB-D camera | 30 fps | whole-body pose (the item CoMind lacks) | Kaichen |

Meta Aria Gen 1/2 satisfies every non-optional row out of the box. Two headsets with TICSync-style hardware time sync.

## Synchronisation and frames (non-negotiable)

1. One shared frame index at 30 fps across all streams; every stream carries its own device timestamp and the mapping to the shared index is stored, never recomputed.
2. Store the frame-0 device timestamp in the exported video metadata (CoMind does; it is what made 80 recordings processable in an evening).
3. Collaborative SLAM output: `T_world_device` per participant in ONE graph. Verify with the minimum inter-wrist distance at annotated handovers (should be a few cm).
4. Metres, Hamilton quaternions xyzw in CSV, `[w,[x,y,z]]` in JSON, following Project Aria conventions already handled in `mps.py`.

## Labels

- Event boundaries: handover (giver, receiver, object, cue type), joint attention, socially conditioned action (verb, noun, cue). Keep CoMind's field names.
- Sub-stages inside each handover: approach / transfer / retract. Generated kinematically by `scripts/handover_quality_metrics.py`, then human-verified.
- Per-event quality: receiver response latency, giver hold time, synchrony, duration (computed, not annotated).
- Object identity per hand at contact (new; required for anticipation).
- Task stage language labels every few seconds ("approaching the glass", "grasping", "handing over") for VLA-style training (Lawrence).

## Tasks to record (coordination-hard, control-easy)

1. Object handover, both directions, ≥ 20 object types, requested / gestured / implicit.
2. Two-person rod or board carry through a constrained path.
3. Hold-and-fasten assembly step (one holds, one fixes).
4. Only record tasks that cannot be completed alone (Kaichen). Cooking-style parallel work is context, not target.

## Quantity and diversity

- ≥ 50 full episodes per task per scene (Lawrence); ≥ 10 scenes; ≥ 20 participant pairs; vary lighting and body size.
- Track collection economics from day one: recording hours → labelled collaborative episodes → cost per usable episode (the buyer's question).

## Quality gates before a recording is accepted

- Hand-tracking valid fraction ≥ 90 % per hand; pose coverage ≥ 99 %; shared graph for both participants; audio transcript present; at least one annotated collaborative event per 5 minutes.

## File layout (mirrors CoMind)

```
recordings/<uuid>/mp4s/<role>_sync.mp4
recordings/<uuid>/mps_<role>/hand_tracking/hand_tracking_results.csv
recordings/<uuid>/mps_<role>/eye_gaze/general_eye_gaze.csv
recordings/<uuid>/multislam_output/<role>/slam/closed_loop_trajectory.csv
recordings/<uuid>/transcripts/<role>_transcript.json
annotations/{handover,joint_attention,scoia}.json
```

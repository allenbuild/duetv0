# Stage `track`: stable person identities (2026-09-26)

`src/duet/playground/track.py`. Re-links the per-frame `body2d` detections into tracks with stable ids, per stream, and
matches the tracks across the fixed views. Runs in ~5 s on the 466-frame CoMind clip (4 streams).

## Why

`body2d` fills up to 4 slots per frame and links them frame to frame by IoU alone (stale slots dropped every 1.5 s). Slot
identities therefore swap or fragment whenever people overlap, one is lost for a moment, or YOLO returns two boxes for
one person (a counter-cut box and a full box; 19 such duplicate frames in `gopro_front`, 112 in `helper`).

## Method

Per stream with `derived/body2d/<stream>.npz`:

1. **Duplicate suppression** within a frame: two detections with IoU > 0.55, or whose shared confident keypoints sit
   within 10 % of the box height of each other, are one person; keep the higher confidence. Not applied when their torso
   appearances clearly differ (Bhattacharyya > 0.4), so two overlapping people are kept. Containment alone is not used
   (a small person can sit fully inside a big person's box: this happened in `gopro_front` and cost the man 7 s of track
   when containment was tried).
2. **Appearance**: hue x saturation histogram (16 x 8, value ignored) of the torso, read from the extracted frame jpg.
   Torso = between the shoulders (COCO 5, 6) and hips (11, 12) when those keypoints are confident, else the middle of the
   box. Per track an EMA (rate 0.15).
3. **Prediction**: constant-velocity box (velocity = EMA of consecutive displacements, extrapolation capped at 5 frames).
4. **Assignment**: Hungarian on `cost = 1.0 * (1 - IoU(predicted, detection)) + 1.0 * appearance distance`, matches with
   cost > 1.4 rejected; unmatched detections start tracks; a track survives 2 s without a match, then closes.
5. **Selection**: tracks < 1 s dropped, the rest ranked by presence, top 4 kept and re-numbered 0..P-1.

Outputs `derived/track/<stream>.npz`: `ids [N,4]` (track id per body2d slot, -1 none/dropped/duplicate), `kpts [N,P,17,3]`,
`boxes [N,P,5]` re-ordered by id, `n_tracks`, `track_len_s [P]`, `presence [P]`, `switch [N]` (a kept track's centre jumps
> 40 % of the image width between consecutive frames), `slot_remap [N]` (a track id changed body2d slot: a raw swap that
was corrected), `global_ids [P]`. Plus `report.json` (validation numbers) and `records_extra.parquet` with
`<stream>_n_people`, `<stream>_switch`, `<stream>_slot_ids` (json), `<stream>_t{i}_cx/cy` (normalised centre),
`<stream>_t{i}_global`; the exporter prefixes them with `track_`.

Ego streams are tracked the same way; the wearer is never visible, so ego tracks are the other person (and partial
detections of arms), and they are not matched across views.

## Cross-view identity

- **world3d reprojection** (`match_tracks_to_world`): when `derived/world3d/world3d.npz` has triangulated bodies and
  `derived/calib/calib.json` gives `T_world_cam` for the fixed camera, every body is projected with
  `geometry.project(intr, inv(T_world_cam), X)` (same convention as `world.py`), and each track is assigned to the body
  with the smallest mean keypoint distance (Hungarian; needs >= 5 frames and < 10 % of the image width). Exercised only
  by the synthetic unit test: the CoMind clip has no board, so `world3d` bodies are all NaN.
- **fallback** (`match_tracks_across_views`, used on the clip): tracks of the other fixed views are assigned to the tracks
  of the first fixed view by Hungarian on `appearance distance + (1 - temporal co-occurrence)`. The report gives the
  cost matrices, the left/right rank agreement of the chosen assignment under a "same" and a "mirrored" orientation
  hypothesis, the cost margin to the next-best assignment and an `ambiguous` flag (margin < 0.1). Without calibration
  the orientation is undetermined; the two cameras facing each other see the same left/right order mirrored, so the
  rank agreement can only say which hypothesis the assignment implies, not whether the assignment is right.

## Validation on `comind_43276420_clip`

Command: `.venv/bin/python scripts/playground.py run comind_43276420_clip --stages track --force`

Identity switch (as specified): a slot's box centre jumps > 40 % of the image width between consecutive frames while both
frames have exactly 2 people.

| stream | role | frames with 2 people | switches before (raw slots) | switches after (track ids) | tracks (raw -> kept) | track lengths s | mean s | duplicate detections | slot remaps |
|---|---|---|---|---|---|---|---|---|---|
| gopro_front | exo | 419 | 0 | 0 | 6 -> 3 | 46.6, 46.6, 1.2 | 31.5 | 19 | 15 |
| gopro_back | exo | 466 | 0 | 0 | 2 -> 2 | 46.6, 46.6 | 46.6 | 0 | 0 |
| leader | ego | 98 | 0 | 0 | 11 -> 4 | 32.9, 6.7, 5.8, 4.9 | 12.6 | 30 | 15 |
| helper | ego | 152 | 0 | 0 | 8 -> 4 | 17.1, 11.2, 10.4, 8.9 | 11.9 | 112 | 42 |

The 40 %-width jump metric is 0 before and after on every stream: the raw body2d linker does not produce big jumps
within a slot, its failure mode is fragmentation (a slot goes empty and the same person reappears in another slot: 10
such frames in `gopro_front`, 18 in `leader`, 10 in `helper`, 0 in `gopro_back`) and duplicate boxes. After tracking
`gopro_front` has both people on one id each for the whole 46.6 s (checked visually on 11 frames including the 15 slot
remaps around k=100-164: woman = t0, man = t1 throughout); the third 1.2 s track is a spurious detection. `gopro_back`
was already clean (2 raw slots, no remap). The ego tracks are fragmented (17.1 s best on `helper`): the partner is only
partly in view, the camera moves, and many detections are arms only.

Cross-view (fallback): `gopro_front` t0 -> `gopro_back` t0, t1 -> t1, appearance costs 0.84/0.91 vs 0.85/0.86, cost margin
0.056, **ambiguous**. Left/right agreement 1.0 for "same". Visual check (front: woman left, man right; back: man left,
ponytail/woman right) says the true relation is **mirrored**, i.e. the fallback assignment is wrong on this clip. Both
people wear dark tops, so torso colour cannot separate them across views; the flag is set and the report says so. A
calibrated rig (board in view) makes the reprojection path available, which does not depend on colour.

Appearance noise: consecutive (5-frame) torso-histogram distance of the same person in `gopro_front` is median 0.13,
p90 0.23, max 0.36, while the two people differ by ~0.34: the appearance term is a tie-breaker on this footage, not a
discriminator, which is why the IoU term keeps weight 1.0 and the cost cap is 1.4.

## Tests

`tests/test_playground_track.py` (10 tests): two synthetic people crossing with red/blue torsos drawn into generated
frames, raw slots swapping at the crossing, ids must not swap (checked for 20/30/40/60-frame crossings; frames where the
two true boxes overlap with IoU > 0.5 are excluded from the check because one torso is drawn over the other and the box
identity is undecidable); occlusion within and beyond the 2 s memory; duplicate suppression (and the
small-person-inside-big-box case must NOT be suppressed); the switch metric; world reprojection matching on a synthetic
pinhole camera; the fallback matcher on same/mirrored/identical-colour views; the stage end to end on a tmp episode.

## Not verified / limits

- The world3d reprojection path only ran on synthetic data; no calibrated episode was available.
- Cross-view assignment on the clip is flagged ambiguous and is visually wrong; nothing downstream should trust
  `global_ids` when `cross_view.pairs[*].ambiguous` is true.
- No ground-truth identity labels exist for the clip; the "after" claim rests on the 40 %-jump metric (0) and on visual
  inspection of 11 `gopro_front` frames and 6 `gopro_back` frames.
- `export` was not re-run (existing stage); `records_extra.parquet` was checked for shape (466 rows) and columns only.

# Metrics stage: operator numbers per session

Built 2026-09-26. `src/duet/playground/metrics.py` (stage), `api_metrics.py` (`GET /api/episode/{name}/metrics`),
`static/metrics.html` + `metrics.js` (page, `?ep=<name>`). Tests: `tests/test_playground_metrics.py`.

The stage turns whatever the pipeline produced for an episode into the numbers an operator would look at after a
recording: how long, how many handovers, how fast the receiver reacted, how long the giver waited, how synchronised
the two people's hands were, how idle each was, how much each spoke, and one "coordination score". It never fails
for lack of inputs: every metric that has no data is `null`, and the source used for handovers is recorded.

## Outputs

`derived/metrics/session.json` (everything below) and `derived/metrics/records_extra.parquet` with per-frame
`hand_speed_<person>` (float), `idle_<person>` (bool), `hand_visible_<person>` (bool); the export stage merges these
into `records.parquet` as `metrics_hand_speed_<person>` etc.

## Hand speed per person

The signal everything else is built on.

| when | signal | unit |
|---|---|---|
| world3d `hands3d_<stream>` wrist (landmark 0) is finite in >= 20 % of frames | wrist speed in the world frame | m/s |
| otherwise | MediaPipe 2D wrist (landmark 0) in that person's ego view, divided by the frame width | frame-widths/s (fw/s) |

Max over both hands, NaN when no hand is visible, 3-frame (0.3 s at 10 fps) moving average. The 2D fallback is a
proxy: an ego camera moves with the head, so 2D wrist speed mixes hand and head motion. The "still" thresholds are
0.10 m/s (as in `scripts/handover_quality_metrics.py`) and 0.06 fw/s (0.10 m/s at ~0.5 m from a ~110 deg lens is
roughly 0.07 fw/s). Both are in `session.json.definitions`.

## Handover events: source priority

The first source that yields at least one event is used and named in `handovers.source`.

1. `derived/contact/events.json` (`{"t","stream","person","hand","object","type": "grasp"|"release"}`). Grasp/release
   pairs are first turned into hold spans per (person, hand, object). A handover is person A releasing object O after
   holding it >= 0.5 s, and person B != A grasping O within 1.5 s and keeping it >= 0.5 s. Each release is used at most
   once; overlapping events with the same object/giver/receiver are merged. Event window: 0.3 s before the earlier of
   (release, grasp) to 0.3 s after the later. The 0.5 s minimum hold exists because two people working over the same
   bowl produce grasp/release chatter every few hundred milliseconds; without it the CoMind clip yields 19 "handovers".
2. `derived/autolabel/proposals.json` (`{"event","t0","t1","confidence","giver","receiver","status"}`): events whose name
   contains "handover" and whose status is not "rejected".
3. world3d `feat_min_wrist_dist_m` < 0.25 m for >= 0.3 s (windows closer than 1 s merged, then padded 1 s each side).
4. 2D wrist proximity in the exo view that sees two people most often: the two largest person slots, min inter-person
   wrist distance divided by the mean torso length (mid-shoulder to mid-hip, all four joints conf > 0.3), < 0.5 torso
   lengths for >= 0.3 s, same merging and padding.

For sources 3 and 4 giver and receiver are not observable. The person whose reach onset comes first is called the
giver (`giver_basis = "onset_order"`); `receiver_response_s` is then >= 0 by construction and should be read as such.
If no onset is found, `giver_basis = "unknown"` and the first person is used.

## Per-handover metrics (ported from `scripts/handover_quality_metrics.py`)

Same symmetric windows for giver and receiver, scaled from 30 fps to `proc_fps`:

| quantity | definition |
|---|---|
| `giver_reach_onset_s` | first run of >= 5/30 s (2 frames at 10 fps) where the giver's speed exceeds baseline mean + 2 sd + floor (0.05 m/s or 0.03 fw/s), searched from t0 - 1.5 s for 6 s; baseline = the 3 s before the search start; relative to t0 |
| `receiver_response_s` | receiver onset - giver onset with IDENTICAL windows (negative = receiver moved first; anticipated) |
| `transfer_s` | argmin of the metric inter-wrist distance in [t0, t1] when world3d has it, else the giver's speed minimum after its speed peak; relative to t0 |
| `giver_hold_s` | seconds between the giver's speed peak and the transfer with giver speed below the still threshold |
| `synchrony`, `synchrony_lag_s` | max Pearson correlation of the two speed profiles over [t0 - 1 s, t1 + 1 s], lag within +-2 s (positive lag = receiver lags giver) |
| `min_wrist_dist_m` | the metric minimum, when available |

Deviation from the script: the script normalised both signals once over the window and averaged the product over
each lag's overlap, which can exceed 1 (it gave 1.045 on this clip). Here each lag's overlap is correlated on its own,
so the value is bounded by 1. NaN speed (no hand) counts as 0, as in the script. The script skips handovers whose
baseline would start before the clip; here the baseline is clamped to the clip and, if shorter than 3 frames, the whole
signal is used.

## Session metrics

- `duration_s` = common_end - common_start; `handovers.count`, `rate_per_min`, medians of receiver response and giver
  hold, `anticipation_fraction` (share of measurable handovers with negative response).
- idle fraction per person: frames belonging to a run of >= 2 s with speed below the still threshold. Frames without
  a visible hand are not idle (we cannot tell).
- hands-visible fraction per person (any hand detected in the ego view).
- speaking fraction per person from `derived/speech/records_extra.parquet` columns `speaking_<person>` (or
  `speech_speaking_<person>`), when present.
- `synchrony.session`: the same cross-correlation on the two whole-session speed signals.
- `per_minute`: for each 60 s bin (last one partial): handovers starting in the bin, mean idle fraction over persons,
  mean hands-visible fraction, mean speech fraction.

## Coordination score (0-100)

    score = 100 * mean(C_sync, C_resp, C_hold, C_engage)
    C_sync   = clip(session synchrony, 0, 1)
    C_resp   = 1 - clip(median |receiver_response_s| / 2 s, 0, 1)   needs >= 1 handover with a measurable response
    C_hold   = 1 - clip(median giver_hold_s / 3 s, 0, 1)            needs >= 1 handover
    C_engage = 1 - mean idle fraction over persons

Components without data are left out of the mean and the score is flagged `partial` when C_resp or C_hold is
missing. It is a first, explicit definition to argue about, not a validated construct: the 2 s and 3 s scales are
chosen so that a response within a few hundred ms and a hold under a second score near 1.

## Numbers on `comind_43276420_clip` (47 s cooking clip, 2 ego + 2 exo, 10 fps), 2026-09-26

Run with `.venv/bin/python -c "from duet.playground.episode import Episode; from duet.playground import metrics; metrics.metrics(Episode.load('data/playground/episodes/comind_43276420_clip'))"`
(or the "Recompute metrics" button); the API test `tests/test_playground_metrics.py::test_metrics_api_on_comind_clip` does the same.

- duration 46.6 s, 466 frames; persons leader, helper.
- hand speed: 2D ego fallback for both (world3d exists but is 100 % NaN: no calibration); hands visible 94 % / 86 %;
  mean speed 0.32 / 0.55 fw/s; idle 0 % / 0 % (10 % of frames are below the still threshold but never for 2 s;
  it is 47 s of continuous cooking, so 0 is plausible, but the ego proxy's head-motion floor also works against idle).
- speaking (speech stage): leader 20 %, helper 27 %.
- handovers: source `contact`, 10 events, 12.9/min. This is an upper bound: the bowl "changes hands" 8 times in
  26 s, which is two people with their hands in the same bowl, something the contact stage cannot separate from a
  transfer. The 0.5 s hold filter also drops a plausible cutting-board pass at 30.5 s (the helper held it 0.4 s).
  A 47 s cooking clip realistically has 2-4 handovers; the pan pass at 19.6-20.6 s looks real.
- receiver response measurable on 3 of 10 (onsets are hard to find on noisy 10 fps 2D signals): median -0.20 s,
  67 % anticipation. Median giver hold 0.00 s (0.8 s on one event).
- synchrony: per-handover median 0.69; session 0.21 at the -2.0 s lag limit (weak, plausible for whole-session
  signals). Coordination score 75 (C_sync 0.21, C_resp 0.80, C_hold 1.00, C_engage 1.00): the score is high mostly
  because idle and hold are zero, which is exactly the weakness of the definition on short busy clips.

## Page

`metrics.html?ep=<name>`: metric cards, per-minute bar chart (handovers on the left axis, idle/speech/hands-visible
fractions on the right, plain canvas), a timeline strip (handover blocks, per-person idle and speaking strips from the
per-frame arrays the API attaches under `frames`, grey where no hand is visible), the per-handover table and the
definitions. "Recompute metrics" posts `run?stages=metrics&force=true` and polls.

## Not done / caveats

- No ground truth: none of the handover numbers on the CoMind clip were checked against the CoMind annotations
  (the clip is 47 s of a longer recording whose annotations live in the CoMind adapter, not in the playground).
- Idle and hold on 2D ego signals depend on the 0.06 fw/s threshold; it was reasoned, not fitted.
- Exo-view person slots are not identities (track stage pending), so source 4 cannot name giver/receiver.

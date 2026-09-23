# Paired benchmark v2: closing the gaps the experts named

Date: 2026-09-21. Follows [v0](paired_kinematics_benchmark_v0.md) and [v1](paired_kinematics_benchmark_v1.md). Same data and protocol: 44 annotated CoMind pairs, 21 h, 248 handovers, recording-level 5-fold CV; no pixels; gradient-boosted trees on causal window statistics unless stated; AUROC with chance 0.500.

## Checklist against what Lawrence, Kaichen, Sebastian and Haoyu asked for

| ask | who | status after v2 |
|---|---|---|
| single-vs-paired ablation with a control | Sebastian | done (v0–v1), now on four tasks |
| a collaboration metric defined before anything else | Sebastian | done: receiver response latency, giver hold time, transfer moment, synchrony, duration; `scripts/handover_quality_metrics.py` |
| predict the collaborator's future action / trajectory | Lawrence, Kaichen | done: helper-action anticipation (below) and partner wrist forecasting (below) |
| helper-as-robot framing: helper anticipates the leader's needs | Kaichen, ETH's own task 2 | done: SCOIA labels, 756 helper actions on the 44 benchmark pairs (986 across all 55 annotated recordings) |
| shared world frame, partner's hands in my frame | Lawrence | built; limited to 11 labelled pairs (v1) |
| sub-stage labels inside a handover | Lawrence | done kinematically (approach / transfer / retract), saved as `handover_stages_v0.npz`, not yet human-verified |
| object pose / hand-object contact | Kaichen, Lawrence, Haoyu | done at category level from ego video (hand-object and gaze-object association, both views, 44 pairs); specific-object tracking across views not done |
| whole-body pose | Kaichen | not possible from Aria alone; in the capture spec |
| a written proposal to review | Sebastian | done: `research_proposal_paired_interaction_data.md` |
| capture spec for our own recordings | all | done: `capture_spec_v0.md` |
| downstream policy evaluation | Lawrence | not started; proposal Section 4 |

## New result 1: the helper's next action is predictable from the leader's body

ETH's SCOIA labels mark every action the helper takes in response to a social cue from the leader (grab, receive, pass, open, add; 756 events on the 44 pairs used here, 986 across all 55 annotated recordings). Task: at every frame, will the helper start such an action within 2 s / 5 s, and is one in progress now.

| view | helper action in progress | starts within 2 s | starts within 5 s |
|---|---|---|---|
| leader's body only | 0.744 ± 0.020 | 0.697 ± 0.029 | 0.660 ± 0.025 |
| helper's body only | 0.730 ± 0.034 | 0.694 ± 0.047 | 0.663 ± 0.029 |
| both, partner time-shuffled | 0.742 ± 0.028 | 0.709 ± 0.044 | 0.659 ± 0.039 |
| **both, paired** | **0.778 ± 0.032** | **0.728 ± 0.037** | **0.685 ± 0.027** |

Two things to read off. The **leader's** kinematics alone predict the **helper's** upcoming action as well as the helper's own kinematics do (0.697 vs 0.694): the cue is in the leader's body. And the paired model beats the de-synchronised control on all three horizons. This is the cleanest anticipation result we have and it is the one that matches how a helper robot would be used.

## New result 2: collaboration-quality metrics from kinematics

Per handover, computed with no annotation beyond the event boundaries (246 handovers, 41 pairs):

| metric | median [IQR] | what it means |
|---|---|---|
| receiver response latency | +0.17 s [−0.90, 1.40] | the receiver starts moving **before** the giver's reach in 44 % of measurable handovers (n = 147), i.e. no systematic anticipation |
| giver hold time | 0.00 s [0.00, 0.10] | givers rarely wait with an extended hand |
| transfer moment | 2.23 s after annotated start | |
| minimum inter-wrist distance at transfer (shared frame, 73 handovers) | 3 cm [1, 20] | validates the Multi-SLAM geometry: the hands meet |
| synchrony (peak cross-correlation of wrist speeds) | 0.52 [0.36, 0.73] | |

By cue type: gestural handovers are the shortest (1.9 s) and most synchronous (0.72); implicit ones the least synchrony (0.46). Receiver latency no longer separates cleanly by cue type (verbal 0.00 s, implicit +0.13 s, gestural +0.58 s). This is the operator product in miniature: waiting, reaction, and coordination quality measured per event, split by how the request was made.

> **Correction (2026-09-22).** The first version of this section reported a receiver response
> latency of -0.60 s and "the receiver moves first in 67 % of handovers", and the research
> proposal repeated it. That was an artefact of `handover_quality_metrics.py` searching the
> **receiver's** reach onset from 1 s before the annotated start while searching the **giver's**
> from the annotated start itself. When both people begin moving before the annotator's mark --
> which is common -- the giver's onset is clamped to the mark and the receiver's is not, so the
> difference is forced negative. Re-running with identical search windows for both roles:
>
> | onset search window | n | median receiver response | receiver first |
> |---|---|---|---|
> | original, asymmetric (giver 0 s, receiver 1.0 s lead) | 152 | -0.67 s | 68 % |
> | symmetric, 0.0 s lead | 150 | +0.07 s | 45 % |
> | symmetric, 1.0 s lead | 156 | -0.02 s | 50 % |
> | symmetric, 1.5 s lead (now used) | 147 | +0.17 s | 44 % |
> | symmetric, 2.0 s lead | 150 | +0.28 s | 43 % |
>
> Every symmetric choice gives 43-50 %, i.e. the receiver moves first about half the time and
> there is **no evidence of systematic receiver anticipation** in these kinematics. Duration,
> giver hold time and synchrony are unchanged by the fix (synchrony 0.52 either way).

## New result 3: partner-trajectory forecasting

Target: the helper's two wrist positions 0.5 s and 1.0 s ahead (displacement in the helper's own frame). Causal TCN, 5 folds, helper target only (the leader target was stopped to save compute once the pattern was clear). Mean Euclidean error:

| input | 0.5 s, all frames | 1.0 s, all frames | 0.5 s, within ±3 s of a handover | 1.0 s, within ±3 s of a handover |
|---|---|---|---|---|
| baseline: hand stays still | 7.3 cm | 10.6 cm | 10.0 cm | 14.8 cm |
| baseline: constant velocity | 11.7 cm | 22.8 cm | 15.1 cm | 29.9 cm |
| model, helper only | 7.2 ± 0.4 | 10.2 ± 0.6 | 9.8 ± 0.7 | 14.1 ± 0.8 |
| model, both, partner shuffled | 7.3 ± 0.4 | 10.3 ± 0.6 | 9.8 ± 0.6 | 14.2 ± 0.8 |
| model, both, paired | 7.3 ± 0.4 | 10.3 ± 0.6 | 9.8 ± 0.6 | 14.2 ± 0.8 |

Negative result, stated plainly: the model beats "the hand stays where it is" by 4 % and the three input conditions are indistinguishable. A small regression model trained with an L1-type loss on multimodal hand futures collapses to predicting near-zero motion, which is exactly the failure Lawrence described ("the same observation can correspond to multiple valid actions; models overfit to one trajectory"). Hand forecasting needs a multimodal or goal-conditioned output (predict where the hand is going, e.g. toward which object or toward the partner's hand), not a point estimate. The classification results above (helper-action onset 0.73 AUROC) are the useful form of the same question.

## New result 4: grasp-state proxies as an object-perception stand-in

Added per hand: thumb–index aperture, fingertip spread, palm-to-wrist extent (all from the 21 Aria landmarks), summarised over 1 s and 3 s windows, on top of the v1 feature set.

| view | handover in progress | onset < 5 s | joint attention |
|---|---|---|---|
| leader only | 0.582 (v1: 0.621) | 0.571 (0.576) | 0.623 (0.633) |
| helper only | 0.632 (0.624) | 0.571 (0.589) | 0.670 (0.675) |
| both, shuffled | 0.597 (0.607) | 0.580 (0.574) | 0.614 (0.635) |
| both, paired | 0.637 (0.662) | 0.581 (0.609) | 0.703 (0.719) |

No gain anywhere; every change is inside the fold-to-fold spread. Hand shape does not stand in for object identity. Object perception has to come from the video.

## New result 5: object identity from the ego video

Built `scripts/build_comind_objects.py`: frames at 3 Hz from both ego videos (164 GB downloaded for the 44 pairs), an open-vocabulary detector (YOLO-World) with a 60-word CoMind kitchen vocabulary grouped into 8 categories (a plain COCO detector names only 12 % of CoMind's objects), and the wearer's Aria 3D hand landmarks and gaze point projected into the image through the official online camera calibration. A hand "holds" the box containing ≥ 3 of its 7 keypoints; gaze "is on" the smallest box containing the gaze pixel. Sanity check on the demo recording: across the annotated bowl handover the receiver's holding flag goes 0.33 → 1.00 and the giver's drops after passing the spoon. Twelve participants have no exported calibration; they use the partner's or a nominal Aria calibration (near-identical across units; adequate for box containment at 640 px).

Same 44 pairs, same folds, same trees, with and without the object block (AUROC):

| view | handover in progress | onset < 5 s | helper action < 2 s | joint attention |
|---|---|---|---|---|
| leader only, + objects | 0.601 (0.582) | 0.568 (0.571) | 0.708 (0.697) | 0.637 (0.623) |
| helper only, + objects | 0.645 (0.632) | 0.561 (0.571) | 0.696 (0.694) | 0.689 (0.670) |
| both, shuffled, + objects | 0.601 (0.597) | 0.558 (0.580) | 0.704 (–) | 0.637 (0.614) |
| **both, paired, + objects** | **0.666** (0.637) | **0.609** (0.581) | **0.741** (0.728) | **0.732** (0.703) |

(parenthesis = matched run without objects). Object identity lifts the paired model on every task by +0.01 to +0.03 and widens the paired-vs-shuffled gap (joint attention 0.732 vs 0.637; helper action 0.741 vs 0.704). It does **not** move handover-onset anticipation beyond where speech left it (0.609). Category-level "what is held / looked at" is not the missing cue for onsets; the specific object requested probably is, and that needs object tracking across both views in one frame, which is the shared-world problem again.


## Review of 22 Sep (PR #1, Andrew) and what it changed

Independent reproduction on a second machine matched every single-view and paired number in this document exactly. Three corrections were made and are now reflected here and in the proposal:

1. **The de-synchronised control cannot establish "interaction".** Rolling the partner destroys its alignment with the labels as well as with the other person, so the control scores like a single view. It rules out "more input columns"; it cannot separate *two independent signals combined* from *the relationship between them*. The right null is **late fusion**: average the leader-only and helper-only models' held-out probabilities. Reproduced here (seed 0, 44 pairs, same folds):

| task | best single | both, de-sync | late fusion | both, paired | paired − late fusion | folds won |
|---|---|---|---|---|---|---|
| joint attention | 0.670 | 0.614 | 0.688 | **0.703** | **+0.016** [+0.012, +0.027] | 5/5 |
| handover in progress | 0.632 | 0.578 | 0.632 | 0.637 | +0.006 [−0.015, +0.036] | 4/5 |
| onset < 5 s | 0.571 | 0.574 | 0.594 | 0.581 | −0.013 [−0.060, +0.020] | 2/5 |
| helper action < 2 s | 0.697 | 0.707 | 0.736 | 0.728 | −0.008 [−0.016, +0.006] | 2/5 |

So: **joint attention is a genuine interaction effect** (survives every control, all folds, bootstrap P = 1.000). For the other three tasks the paired gain over a single view is real but is explained by combining two informative streams; it is not evidence that the model uses their correspondence. Earlier sections that say "the gain comes from the interaction itself" for handovers and helper actions are superseded by this table.

2. **Receiver anticipation was an artefact of asymmetric search windows** (receiver searched from 1 s earlier than the giver). With identical windows: 44 % of receivers move first, median +0.17 s. No systematic anticipation. Corrected in the metrics section above.

3. **Gaze frame.** Gaze is produced in CPF; I treated it as device-frame with a comment calling the offset small. The transform, read from the factory calibration in the VRS header (identical across three units), is a 37.6° rotation and 7 cm offset. It is now applied in the shared-world block (features 25–28) and in the video pipeline's gaze projection, where the previous projection was ~100 px off. World features are rebuilt; object features are being rebuilt (detections cached). The gaze-on-object numbers in "New result 5" predate the fix and will be replaced. The "forward" axis in features 29–30 now uses the measured RGB optical axis (38.7° off device +Z), per the review.

Also from the review: SCOIA count for the 44 pairs is 756 (986 was the 55-recording total); the shuffled control is now deterministic; `scripts/fetch_comind_kinematic_subset.py` restores the MP4-tail fetch that the original pipeline relied on from an uncommitted scratch script; 8 causality tests pin the strictly-causal model.

## Where handover anticipation stands

Best onset-within-5 s is still 0.61 (v1). The helper-action task shows what changes it: when the target is "the responder's next move" rather than "the moment a reach begins", and the cue-giver's body is in the input, anticipation two seconds out reaches 0.73. The remaining gap is object identity (what is being asked for, what each hand holds), which needs video.

## Files

- `src/duet/adapters/comind/labels.py`: SCOIA labels.
- `scripts/baseline_gbdt.py`: grasp proxies, shuffled control, saved predictions.
- `scripts/forecast_partner_trajectory.py`: wrist forecasting, self / paired / shuffled, zero-motion and constant-velocity baselines.
- `scripts/handover_quality_metrics.py`: collaboration metrics and sub-stages; output `outputs/paired_benchmark/handover_quality_metrics.csv`.
- `docs/design/research_proposal_paired_interaction_data.md`, `docs/design/capture_spec_v0.md`.
- `outputs/paired_benchmark/all_runs.md`: every run, every view, every task.

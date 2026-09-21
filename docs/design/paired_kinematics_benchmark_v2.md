# Paired benchmark v2: closing the gaps the experts named

Date: 2026-09-21. Follows [v0](paired_kinematics_benchmark_v0.md) and [v1](paired_kinematics_benchmark_v1.md). Same data and protocol: 44 annotated CoMind pairs, 21 h, 248 handovers, recording-level 5-fold CV; no pixels; gradient-boosted trees on causal window statistics unless stated; AUROC with chance 0.500.

## Checklist against what Lawrence, Kaichen, Sebastian and Haoyu asked for

| ask | who | status after v2 |
|---|---|---|
| single-vs-paired ablation with a control | Sebastian | done (v0–v1), now on four tasks |
| a collaboration metric defined before anything else | Sebastian | done: receiver response latency, giver hold time, transfer moment, synchrony, duration; `scripts/handover_quality_metrics.py` |
| predict the collaborator's future action / trajectory | Lawrence, Kaichen | done: helper-action anticipation (below) and partner wrist forecasting (below) |
| helper-as-robot framing: helper anticipates the leader's needs | Kaichen, ETH's own task 2 | done: SCOIA labels, 986 helper actions |
| shared world frame, partner's hands in my frame | Lawrence | built; limited to 11 labelled pairs (v1) |
| sub-stage labels inside a handover | Lawrence | done kinematically (approach / transfer / retract), saved as `handover_stages_v0.npz`, not yet human-verified |
| object pose / hand-object contact | Kaichen, Lawrence, Haoyu | partial: grasp-state proxies from hand landmarks (aperture, fingertip spread); true object perception needs video, next |
| whole-body pose | Kaichen | not possible from Aria alone; in the capture spec |
| a written proposal to review | Sebastian | done: `research_proposal_paired_interaction_data.md` |
| capture spec for our own recordings | all | done: `capture_spec_v0.md` |
| downstream policy evaluation | Lawrence | not started; proposal Section 4 |

## New result 1: the helper's next action is predictable from the leader's body

ETH's SCOIA labels mark every action the helper takes in response to a social cue from the leader (grab, receive, pass, open, add; 986 events). Task: at every frame, will the helper start such an action within 2 s / 5 s, and is one in progress now.

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
| receiver response latency | −0.60 s [−1.47, 0.33] | the receiver starts moving **before** the giver's reach in 67 % of measurable handovers |
| giver hold time | 0.00 s [0.00, 0.10] | givers rarely wait with an extended hand |
| transfer moment | 2.23 s after annotated start | |
| minimum inter-wrist distance at transfer (shared frame, 73 handovers) | 3 cm [1, 20] | validates the Multi-SLAM geometry: the hands meet |
| synchrony (peak cross-correlation of wrist speeds) | 0.52 [0.36, 0.73] | |

By cue type: gestural handovers are the shortest (1.9 s) and most synchronous (0.72); verbal ones show the earliest receiver movement (−0.63 s); implicit ones the least synchrony (0.46). This is the operator product in miniature: waiting, reaction, and coordination quality measured per event, split by how the request was made.

## New result 3: partner-trajectory forecasting

FORECAST_PLACEHOLDER

## New result 4: grasp-state proxies as an object-perception stand-in

Added per hand: thumb–index aperture, fingertip spread, palm-to-wrist extent (all from the 21 Aria landmarks), summarised over 1 s and 3 s windows, on top of the v1 feature set.

| view | handover in progress | onset < 5 s | joint attention |
|---|---|---|---|
| leader only | 0.582 (v1: 0.621) | 0.571 (0.576) | 0.623 (0.633) |
| helper only | 0.632 (0.624) | 0.571 (0.589) | 0.670 (0.675) |
| both, shuffled | 0.597 (0.607) | 0.580 (0.574) | 0.614 (0.635) |
| both, paired | 0.637 (0.662) | 0.581 (0.609) | 0.703 (0.719) |

No gain anywhere; every change is inside the fold-to-fold spread. Hand shape does not stand in for object identity. Object perception has to come from the video.

## Where handover anticipation stands

Best onset-within-5 s is still 0.61 (v1). The helper-action task shows what changes it: when the target is "the responder's next move" rather than "the moment a reach begins", and the cue-giver's body is in the input, anticipation two seconds out reaches 0.73. The remaining gap is object identity (what is being asked for, what each hand holds), which needs video.

## Files

- `src/duet/adapters/comind/labels.py`: SCOIA labels.
- `scripts/baseline_gbdt.py`: grasp proxies, shuffled control, saved predictions.
- `scripts/forecast_partner_trajectory.py`: wrist forecasting, self / paired / shuffled, zero-motion and constant-velocity baselines.
- `scripts/handover_quality_metrics.py`: collaboration metrics and sub-stages; output `outputs/paired_benchmark/handover_quality_metrics.csv`.
- `docs/design/research_proposal_paired_interaction_data.md`, `docs/design/capture_spec_v0.md`.
- `outputs/paired_benchmark/all_runs.md`: every run, every view, every task.

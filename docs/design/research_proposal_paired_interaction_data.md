# Research proposal: does paired human-human interaction data change the scaling curve for collaborative robot skills?

Duet Labs, 2026-09-21. Draft for feedback from Sebastian Sartor (MIT FutureTech), Lawrence (Georgia Tech), Kaichen Zhou (Stanford/MIT). Two pages. Everything in Section 3 is already measured; Sections 4–6 are what we propose to run.

## 1. Hypothesis

For robot capabilities that involve a human partner (handover, shared manipulation, anticipating a partner's next action), training data that records **both** humans in a collaboration, synchronised and in one spatial frame, produces more downstream capability per hour and per dollar than the same quantity of single-person demonstration data. The mechanism is that the second person's behaviour carries information about intent, timing and role that is absent from a single-person recording. If true, this is a data-scaling argument, not merely a dataset argument.

Null hypotheses to falsify, in order: (i) paired data is just more input: a model given the partner's stream does no better than one given a de-synchronised partner stream; (ii) paired data is two independent signals: the paired model does no better than averaging two single-person models (late fusion). Only beating (ii) is an interaction claim.

## 2. Collaboration metrics (defined first, as requested)

Per-frame targets (present-tense state and anticipation):
- joint attention active; handover in progress; handover onset within H seconds; helper's socially-conditioned action onset within H seconds.

Per-event quality metrics, computed from kinematics alone (no annotation beyond the event boundary):
- **receiver response latency**: time from the giver's reach onset to the receiver's first reaching motion (negative = receiver anticipated);
- **giver hold time**: seconds the giver's hand is extended and still before the transfer (waiting);
- **transfer moment**: minimum inter-wrist distance in the shared frame;
- **synchrony**: peak normalised cross-correlation of the two people's wrist-speed profiles within ±2 s;
- **duration** from cue to completion.

These map directly onto the operator product (waiting, hand-offs, coordination breakdowns) and onto what a robot policy must reproduce.

## 3. What is already measured (CoMind, ETH Zurich; 44 annotated pairs, 21 h, 248 handovers, 756 helper actions)

Sensors: two Meta Aria headsets, hardware time-synchronised; onboard 3D hand tracking, gaze, head SLAM; Multi-SLAM shared world for 36 pairs. No pixels are used by any model below. Evaluation: recording-level 5-fold cross-validation; no pair is ever seen in training.

The ablation Sebastian asked for, with two controls: de-synchronised partner (shifted ≥ 60 s) and late fusion (average of the two single-person models). Gradient-boosted trees on causal window statistics; AUROC, chance 0.50; 5-fold recording-level CV, seed 0:

| task | best single person | both, de-synchronised | late fusion | both, paired |
|---|---|---|---|---|
| joint attention now | 0.670 | 0.614 | 0.688 | **0.703** (beats late fusion 5/5 folds, bootstrap +0.020 [+0.012, +0.027]) |
| handover in progress | 0.632 | 0.578 | 0.632 | 0.637 (matches late fusion) |
| handover starts within 5 s | 0.571 | 0.574 | 0.594 | 0.581 (matches late fusion) |
| helper's next action within 2 s | 0.697 (from the leader's body alone) | 0.707 | 0.736 | 0.728 (matches late fusion) |

Reading: joint attention is a genuine interaction effect. On the other three tasks the paired gain over a single person is real but is accounted for by combining two independent signals; the model is not shown to use their correspondence. Adding object identity read from the ego video lifts the paired model further (joint attention 0.732, helper action 0.741); the late-fusion comparison with those features is pending a gaze-frame correction.

Adding shared-world geometry (both people's hands and gaze in one frame) raises joint attention from 0.646 to 0.729 on the 11 labelled pairs that have it. Kinematic collaboration metrics: receivers begin moving before the giver's reach in 44 % of measurable handovers (median response +0.17 s), i.e. no systematic anticipation once giver and receiver onsets are detected with identical search windows; wrist-speed synchrony is 0.52 [0.36, 0.73]; hands meet at 3 cm median separation at transfer.

What does not work yet: anticipating a handover seconds before it starts is weak for every model and feature set (best 0.61). The cue is what is said and what is looked at; neither noisy kitchen ASR nor gaze without object positions resolves it. ETH's own video-language baselines score 13 % on time-to-handover.

## 4. The scaling experiment we propose

Three data regimes, matched by hours and by cost:
- A: single-person egocentric demonstrations (one headset, existing datasets);
- B: paired, synchronised, shared-frame human-human demonstrations (two headsets);
- C: robot teleoperation (reference).

Task suite, chosen so difficulty is in coordination rather than low-level control (Lawrence): (1) object handover in both directions with varied objects; (2) two-person rod carry through a doorway; (3) a joint assembly step where one person holds and the other fastens. Each task, ~50 episodes per scene (Lawrence's rule of thumb), across ≥ 10 scenes and ≥ 20 participant pairs for scene and body diversity.

Measurements at 10, 100 and 1,000 episodes of each regime:
- partner-conditioned prediction: response latency error, partner wrist trajectory error at 0.5 s and 1 s, onset anticipation AUROC;
- downstream policy: a handover policy trained in simulation (Isaac Lab or ManiSkill) from retargeted human trajectories, success rate and the same latency/hold metrics measured on the robot;
- cost per usable collaborative episode (recording hours to labelled episodes; the NVIDIA buyer's question).

Fit performance versus data on log axes per regime; the claim is a different slope or intercept for B on the collaborative metrics, and no penalty on single-agent metrics.

## 5. Capture specification for our own recordings (consensus of the four conversations)

Two Aria-class headsets (or RGB-D head rigs) with hardware time sync; per-headset 3D hand pose, head pose, gaze; collaborative SLAM into one world frame with the partner's hands projectable into each ego view; audio for the operator product only; RGB-D rather than more sensor types; fine-grained stage labels (approach, transfer, retract) generated kinematically then verified; object identity per hand-object contact. Whole-body pose from an exocentric camera is desirable (Kaichen) and the one item CoMind lacks.

## 6. Questions for reviewers

1. Is the de-synchronised-partner control the right null for "is this just more video"?
2. Is per-episode response latency an acceptable collaboration metric, or should the primary metric be downstream policy success only?
3. For the simulation policy step, is retargeting human wrist trajectories to a fixed-base arm sufficient for a handover, or is the embodiment gap large enough that this step is uninformative?
4. Which of the three tasks would you drop, and what would you add?

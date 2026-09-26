# Playground stage `autolabel`: proposed handover / joint-attention events for human review

Date: 2026-09-26. Code: `src/duet/playground/autolabel.py` (featurisation, hysteresis, stage), `scripts/train_autolabel.py`
(training on CoMind, CV, thresholds, clip validation), model `data/playground/models/autolabel_gbdt.pkl` (1.6 MB), tests in
`tests/test_playground_autolabel.py`. Every number below comes from the quoted command.

## Idea

Train once on the 44 labelled CoMind pairs, run on playground episodes. That only works on features both sides can
produce, so the model uses a REDUCED set (`duet.playground.autolabel`), computed by the same code on both sides at 10 Hz
(the playground's `proc_fps`):

| per person (11) | playground source | CoMind source |
|---|---|---|
| wrist speed L/R (image widths / s) | MediaPipe wrist (landmark 0) in `derived/hands`, / image width | Aria wrist (landmark 5, device frame) projected with an equidistant fisheye (f = 608/1408 W) about the RGB optical axis |
| hand present L/R | MediaPipe detection | Aria tracking valid |
| wrist radial offset from the image centre L/R | same | same (roll-invariant, so the arbitrary in-plane basis of the projection does not matter) |
| own L-R wrist distance | same | same |
| held L/R | `derived/contact/<ego>.npz` `held` != "" (zeros if absent) | `objects_v0.npz` holding flags (cols 8 / 17) |
| angle head-forward -> partner head, + valid flag | `derived/gaze_proxy` (`stage_gaze_proxy.md`) | `world_v0.npz` cols 29/30 (11 recordings; invalid elsewhere) |

| inter-person (3) | | |
|---|---|---|
| head distance, min wrist-wrist distance (m), valid | `world3d` `feat_*` (zeros if absent) | `world_v0.npz` cols 6 / 4 (11 recordings) |

Each of the 25 base columns is joined by its causal mean / std / max over 1 s and 3 s windows: 175 features. Person A is
the first ego stream, B the second; training uses both orderings (A,B) and (B,A) so the model does not need to know who
leads. Proposals come from hysteresis on the per-frame probability: an interval starts when p > hi for 0.5 s and ends
when p < lo for 1.0 s; confidence = mean p inside. For handovers a third model (trained on handover frames only)
estimates P(A gives), which sets giver/receiver. Output: `derived/autolabel/proposals.json`
(`[{event, t0, t1, confidence, giver, receiver, status: "pending", source: "autolabel_gbdt"}]`, reference seconds)
and `records_extra.parquet` (`p_joint_attention`, `p_handover`; export prefix `autolabel_`).

Not a heuristic fallback: the CoMind caches (`kinematics_v0.npz`, `objects_v0.npz`, `world_v0.npz`) load in ~10 s, so the
learned version shipped.

## Training and cross-validation on CoMind

```
.venv/bin/python scripts/train_autolabel.py --out <scratch>/autolabel_cv.json
```

44 recordings, recording-level 5-fold CV with the same fold assignment as `scripts/baseline_gbdt.py` (rng seed 0 over
the load order), same trees (HistGradientBoosting, 100 iters, lr 0.15, 31 leaves, min leaf 200, class weight capped at
50), rows from 3 s onward at 5 Hz (~8.7 k per recording; ~600 k training rows per fold with both orderings). Evaluation
on the natural ordering (A = leader). 11 of the 44 recordings have the world block, 44 have object flags. 8.5 min.

| task | reduced features (this model), fold AUROC mean +- sd | per fold | pooled AP | prevalence | full-feature GBDT, `both` view | full-feature, `helper` only |
|---|---|---|---|---|---|---|
| joint attention (`ja_active`) | **0.756 +- 0.034** | 0.786, 0.764, 0.738, 0.793, 0.699 | 0.390 | 0.15 | 0.703 +- 0.057 | 0.670 +- 0.054 |
| handover in progress (`handover_active`) | **0.604 +- 0.112** | 0.650, 0.628, 0.416, 0.756, 0.570 | 0.024 | 0.027 | 0.637 +- 0.045 | 0.632 +- 0.042 |

Full-feature numbers are `outputs/paired_benchmark/baseline_gbdt_grasp.json` (44 recordings, hands + gaze + speech,
no objects; the run quoted in `paired_kinematics_benchmark_v2.md`, "same 44 pairs" table). With objects
(`baseline_gbdt_objects_all44.json`, `both`): joint attention 0.732 +- 0.042, handover 0.666 +- 0.051.

Reading: on joint attention the reduced set is not worse than the full kinematic set; it is better (0.756 vs 0.703),
which is plausible since it carries the held-object flags (the full "+ objects" run is at 0.732) and, on 11 recordings,
the head-forward angle to the partner (the strongest single joint-attention cue, see `stage_gaze_proxy.md` part b), and
it is trained on twice the rows. On handover-in-progress it is weaker (0.604 vs 0.637) and unstable across folds
(0.42-0.76): a 2.7 % prevalence event described only by 2D wrist speeds and presence flags, with no partner geometry on
33 of 44 recordings. Direction model (A gives to B), handover frames only: frame-level AUROC 0.572 +- 0.072, event-level
accuracy (mean probability over each annotated handover) 0.636 on 247 handovers; barely better than the 50 % coin.

### Thresholds

Chosen on the pooled out-of-fold probabilities, target precision 0.5 (frame level); lo = 0.6 hi.

| task | hi | lo | rule | frame P / R at hi | OOF event level (hysteresis on the OOF traces; hit = overlap or onset within 1 s) |
|---|---|---|---|---|---|
| joint attention | 0.742 | 0.445 | precision >= 0.5 reached | 0.50 / 0.27 | 1216 proposals vs 4455 annotated runs: precision 0.665, recall 0.372, F1 0.477 |
| handover | 0.392 | 0.235 | precision 0.5 unreachable at any threshold; hi = OOF quantile (q = 0.85) maximising event-level F1 | 0.026 / 0.14 | 2266 proposals vs 234 handovers: precision 0.068, recall 0.543, F1 0.121 |

Handover grid (quantile, hi, proposals, P, R, F1): 0.80/0.333/2930/0.066/0.671/0.120; 0.85/0.392/2266/0.068/0.543/0.121;
0.90/0.481/1392/0.073/0.342/0.120; 0.93/0.565/843/0.077/0.239/0.117; 0.95/0.643/609/0.066/0.162/0.094;
0.97/0.736/451/0.062/0.120/0.082; 0.98/0.790/347/0.072/0.111/0.087; 0.99/0.855/185/0.070/0.060/0.065;
0.995/0.901/83/0.084/0.030/0.044. Precision stays at ~7 % whatever the threshold: as a handover proposer this model is
a recall device (about half of the handovers get a proposal, at ~10 false proposals per true one), not a precision one.
(The 4455 "annotated runs" for joint attention are the contiguous runs of the 10 Hz `ja_active` mask, i.e. the
annotation intervals, ~100 per recording.)

## Validation on `comind_43276420_clip`

The clip is a 50 s cut of CoMind recording 43276420: matching the clip frames against `leader_trimmed_sync.mp4` at 30 fps
puts the leader clip at exactly 590.000 s (mean abs grey difference 0.45 vs 3.3 for the next-best frame) and the helper
clip at 588.300 s of `helper_trimmed_sync.mp4` (0.41 vs 2.4); the episode's `align` stage found helper offset -1.73 s,
consistent with the 1.70 s difference of the cuts (the raw helper video lags the leader by 7 native frames, 0.23 s,
so the true offset is between -1.47 and -1.70; the align estimate is within 0.3 s). The reference timeline is the leader
clip, so annotation frame f maps to t_ref = f/30 - 590. Common window 1.67-48.27 s.

```
.venv/bin/python scripts/playground.py run comind_43276420_clip --stages gaze_proxy,autolabel --force
  autolabel: done 1 joint-attention + 1 handover proposals (model 2026-09-26)
.venv/bin/python scripts/train_autolabel.py --validate-clip comind_43276420_clip --recording 43276420-701f-4731-b9ab-bebc7fd14994 --clip-offset-s 590.0
```

(`gaze_proxy` had been run before; `contact` output from the parallel stage was present: held 15 % / 36 % of frames for the
leader's L/R hand, 26 % / 26 % helper.)

Annotations in the window (t_ref, s): handovers [18.00, 18.43] leader->helper serveware, [18.67, 19.57] leader->helper
utensil, [41.37, 43.17] helper->leader serveware; joint attention [0.40, 2.87], [3.80, 8.23], [8.87, 12.63],
[13.20, 28.67], [30.37, 31.60], [33.27, 35.37], [37.10, 37.70], [37.67, 38.60], [39.10, 41.40], [42.37, 43.20],
[43.87, 69.50] (80 % of the window's frames are joint attention).

Proposals:

| event | t0 | t1 | confidence | giver -> receiver |
|---|---|---|---|---|
| handover | 19.77 | 20.37 | 0.476 | helper -> leader (P(A gives) 0.155) |
| joint_attention | 24.47 | 28.47 | 0.690 | |

| event | criterion (tolerance 1 s) | precision | recall |
|---|---|---|---|
| handover (1 proposal, 3 annotated) | onset within 1 s | 0/1 = 0.00 | 0/3 = 0.00 |
| | onset and offset within 1 s | 0.00 | 0.00 |
| | any overlap | 0.00 | 0.00 |
| joint attention (1 proposal, 11 annotated) | onset within 1 s | 0.00 | 0.00 |
| | onset and offset within 1 s | 0.00 | 0.00 |
| | any overlap | 1/1 = 1.00 | 1/11 = 0.09 |

The handover proposal starts 0.2 s after the second annotated handover ends (onset 1.1 s after the annotated onset 18.67,
just outside the tolerance) and gets the direction wrong; the third handover (41.4-43.2) and the first (0.4 s long)
are missed. The joint-attention proposal sits inside the long 13.2-28.7 s interval (correct region, but 11 s late and
therefore not an onset match). Per-frame, against the annotation masks on the episode timeline: handover AUROC 0.786
(AP 0.200, prevalence 0.07), joint attention AUROC 0.666 (AP 0.904, prevalence 0.80).

### Where the gap is: model or featurisation?

`--validate-clip` also featurises the same 46.6 s window from the CoMind side (Aria wrists, `objects_v0` held flags; no
world block for this recording) and runs the same model:

| side | handover frame AUROC | handover proposals (t_ref) | event P / R | joint-attention frame AUROC | JA proposals | event P / R |
|---|---|---|---|---|---|---|
| playground features (MediaPipe, YOLO-World contact) | 0.786 | [19.77, 20.37] | 0.00 / 0.00 | 0.666 | [24.47, 28.47] | 1.00 / 0.09 |
| CoMind features (Aria hands, objects_v0) | 0.926 | [8.67, 11.97], [18.77, 21.07], [37.47, 45.07] | 0.67 / 1.00 | 0.574 | [3.17, 18.77], [24.47, 28.57], [30.67, 43.37] | 1.00 / 0.82 |

So the model itself separates the handovers in this window well from CoMind-side features (0.93) and proposes all three
(two of them within tolerance of an annotated one); on the playground-side features it separates them less well (0.79)
and proposes only one. Mean base features over the window, playground vs CoMind side: leader hand present L/R 0.58/0.71
vs 0.99/0.76; wrist speed L/R 0.081/0.225 vs 0.136/0.244 widths/s; wrist radial offset 0.18/0.19 vs 0.36/0.32; own
hand distance 0.080 vs 0.259; held L/R 0.15/0.36 vs 0.44/0.45 (helper similar: present 0.60/0.52 vs 0.95/1.00, held
0.26/0.26 vs 0.60/0.48). The remaining shift is detector coverage (MediaPipe finds the wearer's hands in ~60 % of the
frames where Aria tracks them in 99 %), the contact stage flagging about half as many held frames as the CoMind
object pass, and a ~0.15 W difference in radial offset (the equidistant model over-states off-axis radii relative to the
real Fisheye624 image the MediaPipe wrists live in). None of these is a bug in the sense of the u/v one below; they are
the price of running one model on two perception stacks. The angle-to-partner feature is only present on the playground
side (the recording has no Multi-SLAM cache), which the model handles through its validity flag.

A first version of the feature set carried the wrist u/v position; it was top-left-anchored on the playground and
optical-axis-centred with an arbitrary roll on the CoMind side, and the clip's handover frame AUROC was 0.387 with it.
Replacing u/v by the roll-invariant radial offset (the only change) took it to 0.786; the CV numbers barely moved
(joint attention 0.760 -> 0.756, handover 0.600 -> 0.604). The lesson generalises: every feature in this set must be
invariant to whatever is arbitrary in one side's featurisation.

## Tests

`.venv/bin/python -m pytest tests/test_playground_autolabel.py -q`: 5 tests (feature layout and units, projection
roll-invariance, hysteresis timing, the stage end to end on a synthetic two-ego episode with a stand-in model and
contact-stage-style `held` strings, skip without a model). Together with the gaze tests: 9 passed.

## Not verified / caveats

- Event-level precision/recall on the playground side rests on one 46.6 s clip with 3 handovers; the CoMind-side
  out-of-fold event numbers (handover precision 0.07, recall 0.54; joint attention 0.67 / 0.37) are the representative
  ones.
- The direction model (0.64 event accuracy on CoMind) should be read as a coin flip; giver/receiver in the proposals
  are there for the reviewer to correct, not to trust.
- The head-forward angle and inter-person distances were available for training on 11 of 44 recordings only; on rigs
  with a board (world3d) these features will be present at test time far more often than they were at training time.
- Speed units assume an Aria-like lens on the ego (the projection constant 608/1408 and the 110 deg field); a GoPro or
  ZED ego will shift the speed and radial-offset distributions.
- Hysteresis constants (0.5 s on, 1.0 s off, lo = 0.6 hi) were set, not tuned.

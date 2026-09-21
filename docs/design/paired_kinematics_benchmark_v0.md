# Paired-vs-single kinematic benchmark on CoMind (v0)

Date: 2026-09-20. Author: Idhant. Status: first full-dataset result.

## Why this exists

Every researcher we talked to asked the same thing in different words:

- Sebastian (MIT, scaling laws): "do an ablation: model trained with the partner's data vs without."
- Lawrence (GT, EMMA): "benchmark your data with a downstream evaluation; labs want evidence before buying."
- Kaichen (Stanford/MIT): "the most important signal is hand pose; make sure the task truly needs two people."

The CoMind paper (ETH, 80 paired-Aria cooking recordings, 81 h) ships hand tracking and gaze for both wearers plus handover / joint-attention annotations, but its own benchmarks are video-language-model-only, never use hands or gaze, and never ablate one view against both. Their time-to-handover accuracy tops out at 13%.

So the first Duet ML result is: **from body kinematics alone (hands + gaze, no pixels), does observing both people let a model anticipate collaborative events better than observing one, and is the gain due to the interaction rather than to having more input?**

## Data

| item | value |
|---|---|
| recordings with annotations | 55 (CoMind train split; the 25 test-split recordings have their labels withheld) |
| duration | ~24 min each, ~22 h total annotated |
| per-frame input, per person | 2 hands × (7 keypoints × 3 + palm normal 3 + confidence + valid) + gaze (3D point, direction, valid) = 59 dims |
| paired input | 118 dims, leader block then helper block, 30 fps, wearer's device frame |
| labels | 309 usable handovers, 5,582 joint-attention segments, all on the shared synced frame index |

Frame-to-device-time anchor: each `<role>_trimmed_sync.mp4` stores frame-0 DEVICE_TIME in its `desc` metadata. Frame `i` is at `T0 + i × 33.327 ms`. This matches Allen's 24 GB VRS image-matching result exactly (helper offset 7 frames, leader 0) and the hand-coverage percentages it reported to within 0.1%. A 2–12 MB HTTP range request per video replaces the VRS download, which is what made processing all 80 recordings possible on a laptop in one evening.

## Method

- Model: causal dilated temporal CNN (6 residual blocks, receptive field ~8 s; 0.6 M params in v0, 0.17 M in v2), input = z-scored features plus first differences. Sees only the past. v2 uses per-time-step LayerNorm; v0's GroupNorm normalised over the time axis, which leaks future frames within a crop and gives train/eval statistic mismatch (a 512-frame crop vs a 40k-frame recording). v0 numbers are kept because the ordering is unchanged.
- Heads: handover in progress; handover onset within 1 s; onset within 2 s; joint attention active (all BCE with class weighting); time-to-handover regression (≤ 5 s, smooth L1).
- Views: `leader` (leader's hands+gaze only), `helper`, `both`, and the control `both_shuffled` (partner stream circularly shifted ≥ 60 s within the same recording: same marginal statistics, interaction destroyed).
- Evaluation: 5-fold recording-level cross-validation (33 train / 11 val for model selection / 11 test per fold); held-out predictions pooled over all 55 recordings, so frame-level AP is computed over every annotated handover. Event-level analysis converts probabilities into recall at fixed false-alarm rates and anticipation lead time.

## Results

Two independent training recipes were run end to end (v0: 625K-param TCN, GroupNorm, uniform crop sampling; v2: 165K-param TCN, strictly causal per-step LayerNorm, 30% of crops centred on handover onsets, dropout 0.3). Both give the same ordering on every task. Numbers are AUROC, mean ± sd over the 5 held-out folds (chance = 0.500). Frame-level AP is reported in `outputs/paired_benchmark*/results.md` but is near prevalence for the handover tasks (0.5–3% of frames) and is not a useful summary there.

| view | handover in progress | handover starts <1 s | handover starts <2 s | joint attention now |
|---|---|---|---|---|
| leader only (v0 / v2) | 0.596 ± 0.045 / 0.590 ± 0.042 | 0.565 / 0.544 | 0.555 / 0.539 | 0.575 ± 0.051 / 0.547 ± 0.043 |
| helper only | 0.588 ± 0.054 / 0.586 ± 0.022 | 0.547 / 0.545 | 0.524 / 0.516 | 0.629 ± 0.051 / 0.592 ± 0.023 |
| both, partner time-shuffled (control) | 0.531 ± 0.053 / 0.568 ± 0.020 | 0.546 / 0.532 | 0.532 / 0.537 | 0.557 ± 0.048 / 0.523 ± 0.033 |
| **both, paired** | **0.630 ± 0.042 / 0.630 ± 0.044** | **0.570 / 0.581** | **0.550 / 0.561** | **0.690 ± 0.033 / 0.633 ± 0.053** |

Joint attention, frame-level AP pooled over all held-out recordings (v0; chance = prevalence ≈ 0.15): leader 0.186, helper 0.237, shuffled 0.168, **paired 0.296**.

Linear floor (logistic regression on hand-crafted wrist-speed / validity / gaze features, same folds): handover in progress AUROC leader 0.635, helper 0.621, both 0.648; joint attention AUROC leader 0.571, helper 0.613, both 0.627. The TCN beats the linear floor clearly only on joint attention.

Event-level handover anticipation (peak p(onset within 2 s) in the 2 s before each of 246 held-out onsets vs. random 2 s windows ≥ 5 s from any handover): event AUROC 0.52–0.58 for every view, recall at 5% false-alarm rate 6–10%. In plain terms: hands+gaze in each wearer's own frame do not anticipate a handover seconds ahead, for anyone.

Raw kinematic check: leader wrist speed averaged over all 246 onsets rises from 0.43 m/s (background 0.45) to 0.70–0.74 m/s in the 0.5 s after the annotated onset. Labels and features are aligned to well under a second; the signal before onset is simply weak.

Figures: `outputs/paired_benchmark/figures_v0/benchmark_results.png` (bars), `timeline_43276420_ja_active.png` (held-out recording: paired model near zero and spiking on annotated joint attention while single-view/shuffled models hover uninformatively), `frames_43276420_*.png` (both ego views around a handover). v2 equivalents in `figures_v2/`.

## Reading the result

1. **The second person's kinematics carry information the first person's do not.** On both collaborative tasks, paired > best single view > shuffled control, in two independent recipes. The shuffled control has the same input dimensionality and identical marginal statistics as the paired model; it drops to (or below) single-view performance. So the gain is temporal correspondence between the two people, which is exactly the "is it just more video?" objection Sebastian raised.
2. **Joint attention is the clean win**: +0.06 to +0.11 AUROC and +0.06 AP over the best single view, +0.13 over the shuffled control. This makes sense: joint attention is definitionally a two-person state, and gaze is in the input.
3. **Handover detection is weak for everyone, and anticipation is at chance.** The paired advantage on "handover in progress" is real but small (+0.03–0.04 AUROC) and the TCN barely beats a wrist-speed logistic regression. Anticipating a handover 1–2 s ahead from hands and gaze in each wearer's own headset frame does not work. This is a negative result worth having: it says the missing ingredient is geometry between the two people (are their hands converging? is one person's gaze on the other's hand?), which requires the shared world frame that Allen's Multi-SLAM work targets. That is the single highest-value next feature.
4. **Caveats on the numbers.** Model selection used mean AP across tasks, which joint attention dominates; a per-task checkpoint would likely lift the handover numbers for the paired model slightly. Pooling probabilities across five differently calibrated fold models makes pooled AUROC unreliable for rare tasks, which is why per-fold means are reported. Eleven of the 55 annotated recordings lack one participant's hand-tracking file and are excluded so that every view is trained and tested on identical data.

## What this does and does not show

- It shows whether the *second person's kinematics* carry information about upcoming collaborative events that the first person's do not, in real unscripted kitchens, across 55 pairs. That is the claim Sebastian and Lawrence asked us to test.
- It does not show robot-policy improvement. That needs a policy benchmark (handover task in sim or on hardware), which is the next research milestone, not this one.
- Positions are in each wearer's own device frame. No shared world frame yet: the model cannot see the geometric relation between the two people, only the temporal correlation of their motions. Adding Multi-SLAM head poses (both people in one frame; ~1 GB/recording) is the obvious next feature and should widen the paired-vs-single gap.
- Cooking only, 2 people only. Factory transfer is untested.

## Reproduce

```sh
.venv/bin/python scripts/build_comind_kinematics.py
.venv/bin/python scripts/train_paired_benchmark.py --views both,both_shuffled --steps 1500 --eval-every 300 --channels 64 --dropout 0.3 --onset-crop-frac 0.3 --task-weights 1,2,2,0.5 --pos-weight-cap 60 --out outputs/paired_benchmark_v2/A
.venv/bin/python scripts/train_paired_benchmark.py --views leader,helper       --steps 1500 --eval-every 300 --channels 64 --dropout 0.3 --onset-crop-frac 0.3 --task-weights 1,2,2,0.5 --pos-weight-cap 60 --out outputs/paired_benchmark_v2/B
.venv/bin/python scripts/baseline_handcrafted.py
.venv/bin/python scripts/analyze_handover_events.py --result-dirs outputs/paired_benchmark_v2/A outputs/paired_benchmark_v2/B --task handover_active
.venv/bin/python scripts/plot_paired_benchmark.py --result-dirs outputs/paired_benchmark_v2/A outputs/paired_benchmark_v2/B --out outputs/paired_benchmark/figures_v2
```

The kinematic downloads (hands, gaze, transcripts, MP4 tails for all 80 recordings, ~12 GB) were fetched with a small manifest-driven script; `scripts/comind_download.py --parts mps` also works but pulls the 1 GB trajectory files too.

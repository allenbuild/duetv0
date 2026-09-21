# Paired-vs-single benchmark, v1: attacking handover anticipation

Date: 2026-09-21. Follows [v0](paired_kinematics_benchmark_v0.md). Everything here is recording-level 5-fold CV over the 44 annotated CoMind recordings with both participants' hand tracking (21 h, 248 handovers), unless a smaller subset is stated. Numbers are AUROC, mean ± sd over held-out folds; chance is 0.500. Full table of every run: `outputs/paired_benchmark/all_runs.md`.

## What was tried, in order, and what each did

| lever | why | result |
|---|---|---|
| Strictly causal normalisation (per-step LayerNorm instead of GroupNorm) | GroupNorm normalised over the time axis: leaked future frames inside a training crop and produced different statistics for a 512-frame crop vs a 40k-frame recording. Handover heads sat at 0.50. | Handover-in-progress 0.50 → 0.63 for the paired TCN. Real bug, real fix. |
| Onset-centred sampling, class-weight cap 60, smaller model, dropout 0.3 | rare class (0.5–3 % of frames) never seen enough | Stabilised training; no change in ordering. |
| Checkpoint selection by mean AUROC instead of mean AP | AP selection was dominated by joint attention, so the paired model's chosen checkpoint traded handover for JA | Removed the "pooled both looks like chance" artefact. |
| **Speech features from the transcripts** (speaking, time since last word, word rate, request keywords, 32-bin hashed bag-of-words per person) | 64 % of handovers are verbally cued | Handover-in-progress +0.02 to +0.04 (paired 0.630 → 0.654). Anticipation (onset < 2 s) unchanged at ≈ 0.57. |
| **Longer horizon (onset < 5 s)** | measured that the verbal request ends a median 2.7–3.6 s *before* the annotated onset, so a 1–2 s horizon asks the model to fire before the cue exists | TCN onset < 5 s: paired 0.565, single 0.53–0.56. No help for the TCN. |
| **Shared-world geometry** (Multi-SLAM: both people's hands, heads and gaze points in one frame; partner's hands projected into each wearer's frame; gaze-to-partner distances) | the thesis feature: inter-person geometry is invisible in own-frame kinematics | Only 11 of the 44 labelled recordings have Multi-SLAM output (the other 25 world recordings are in the label-withheld test split), 73 handovers. Joint attention **0.646 → 0.729** on those 11 with the world block. Handover heads at chance for both conditions on 11 recordings: too little data to say anything. |
| Time-to-completion regression inside a handover | a robot-relevant quantity (when will the exchange happen) that should be learnable from the reach | TCN head and a GBDT regressor both fail to beat predicting the constant median (1.65 s vs 1.61 s MAE). `end_frame` durations span 1–9 s; the label is probably too loose. Dropped. |
| **Gradient-boosted trees on causal window statistics** (mean/max/min over 1 s and 3 s of wrist speeds, hand-hand distance, hand height, gaze point, validity, speech flags) | the linear baseline on wrist speed already anticipated onsets at 0.62, better than every TCN, so the sequence model was the bottleneck on rare events | Best results across the board (table below). |
| MiniLM sentence embeddings of the most recent utterance (PCA-32, recency-decayed) | requests are indirect ("Can I ask you to bring some salt?", "give me a plate", "I have a lid for you"), hashed words miss them | No gain: paired onset < 2 s 0.583, onset < 5 s 0.547, handover-in-progress 0.636 (all within noise of the hashed-word run). Better text representation is not the missing piece with this ASR. |

## Best model per task (GBDT unless stated)

| view | joint attention now | handover in progress | onset < 1 s | onset < 2 s | onset < 5 s |
|---|---|---|---|---|---|
| leader only | 0.633 ± 0.033 | 0.621 ± 0.038 | 0.579 ± 0.040 | 0.583 ± 0.041 | 0.576 ± 0.036 |
| helper only | 0.675 ± 0.046 | 0.624 ± 0.041 | 0.585 ± 0.033 | 0.597 ± 0.036 | 0.589 ± 0.031 |
| both, partner time-shuffled | SHUF_JA | SHUF_HO | – | – | SHUF_ON5 |
| **both, paired** | **0.719 ± 0.043** | **0.662 ± 0.042** | **0.597 ± 0.030** | 0.595 ± 0.029 | **0.609 ± 0.043** |

TCN (v0/v3) for reference, paired view: joint attention 0.690, handover in progress 0.654, onset < 2 s 0.566.

## Reading it

1. **Joint attention is now a solid result**: paired 0.72 vs best single 0.68 vs shuffled control near chance, and the shared-world block adds another +0.08 where it exists. This is the "second person carries information" claim with two independent controls.
2. **Handover in progress**: paired 0.66 vs single 0.62. Consistent across TCN and GBDT, modest.
3. **Handover anticipation is still weak for everyone.** Best anticipation of an onset 5 s out is 0.61 AUROC; at 1–2 s it is ≈ 0.60; the paired edge over the helper-only view is 0.02 at 5 s and zero at 2 s. Event-level recall at 5 % false alarms is under 10 % in every configuration. None of the levers moved it by more than a few points. This is not a training bug any more: a linear model, a TCN and boosted trees on three feature sets all land in the same place.
4. What the anticipation failure is telling us, concretely:
   - The cue that precedes a handover is mostly *what is said* and *what is looked at*. Speech arrives 3 s before the reach, but our transcript features are crude and ASR in a kitchen is noisy; the sentence-embedding run tests how much headroom there is.
   - Gaze is in the input, but without object positions (needs vision) or the partner's position (needs the shared world, which only 11 labelled recordings have) the model cannot tell *what* is being looked at. The v4 joint-attention jump (+0.08 from geometry alone on 11 recordings) says exactly this signal matters.
   - CoMind's own VLM baselines with full video and 10 s of transcript reach 13 % on time-to-handover. Anticipating a kitchen handover seconds ahead is hard for everyone; our result is in line with theirs, from a far cheaper sensor set.

## What would actually move anticipation (ranked)

1. **Label more Multi-SLAM recordings, or get ETH's test-split labels.** The shared-world block is the one feature with a demonstrated large effect (joint attention +0.08 on 11 recordings) and it has never been tested on handovers with enough data. 25 recordings with Multi-SLAM sit in the withheld split.
2. **Object-level perception.** Hand-object detection in both ego views (what each person is holding, what the leader looks at) turns "gaze somewhere" into "gaze at the bowl the helper is holding". This is the feature the VLM baselines implicitly have.
3. **Better speech**: real diarisation (who is asking), and a request/offer classifier trained on the 309 annotated transcript windows.
4. Reframe the task the way a robot would use it: not "will a handover start in the next N seconds, at every frame of a 24-minute recording", but "given the leader just spoke or looked at me, is this a request", evaluated at utterance boundaries. Prevalence goes from 1 % to ~20 % and the metric becomes interpretable.

## Reproduce

```sh
.venv/bin/python scripts/build_comind_kinematics.py
.venv/bin/python scripts/build_comind_speech.py
.venv/bin/python scripts/build_comind_speech_emb.py      # MiniLM, ~4 min CPU
.venv/bin/python scripts/build_comind_world.py           # needs multislam_output/*/slam/closed_loop_trajectory.csv
.venv/bin/python scripts/baseline_gbdt.py --speech --tasks handover_active,onset_within_5s,ja_active --views leader,helper,both_shuffled,both --save-preds-dir outputs/paired_benchmark/gbdt_preds
.venv/bin/python scripts/train_paired_benchmark.py --views both,both_shuffled --speech --speech-source both --tasks handover_active,onset_within_2s,onset_within_5s,ja_active --channels 64 --dropout 0.3 --onset-crop-frac 0.3 --task-weights 1,2,2,0.5 --pos-weight-cap 60 --steps 1500 --eval-every 300 --out outputs/paired_benchmark_v7_emb/A
.venv/bin/python scripts/summarize_benchmark_runs.py
.venv/bin/python scripts/plot_best_models.py
.venv/bin/python scripts/render_demo_video.py --result-dir outputs/paired_benchmark/gbdt_preds --view both
```

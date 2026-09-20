# Strict and demonstration export QC

The demonstration profile is an opt-in visualization/export policy. It does not
change the existing strict recommendation, canonical sample acceptance, clock
domains, temporal mappings, calibration, or benchmark eligibility.

## Shared evidence requirements

Both profiles use the existing canonical playback evidence. Every event and
shown context frame must have a VERIFIED MP4-to-native-VRS correspondence for
each participant. Source hashes, native integer capture timestamps, validation
generation, shared-world frame, camera calibration, and timestamp residual
gates are checked as before. An ambiguous mapping is never interpolated or
promoted for export.

The normal context is 120 presentation frames on either side of the inclusive
annotation interval (four seconds at 30 fps). A 90-frame context is allowed
only when the four-second context fails the mapping gate and the complete
three-second context passes. These durations concern video presentation, not
comparison of independent device clocks.

## Profiles

The existing strict profile remains unchanged: in addition to the common
evidence gates, event poses must be complete and at least 90% of event frames
must contain all four accepted hands. Its existing caution rules remain intact.

The demonstration profile requires:

- At least 95% helper pose coverage and at least 95% leader pose coverage,
  evaluated separately for the annotated event and entire shown clip.
- At least 80% event ANY-hand coverage for each participant. ANY means the
  frame-wise union of accepted left and right hands; the visible hand may change
  across frames. It is neither the sum nor maximum of the two side coverages.
- A median confidence of at least 0.90 across available accepted event hand
  observations. Empty confidence observations fail. Reported confidence is
  pooled across tracked hands; it does not establish which hand touches the
  object. No interacting-hand identity is inferred from source labels.
- Valid accepted geometry and no wrist-speed outlier under the existing
  5 m/s limit, considering consecutive accepted observations with positive
  own-clock intervals no longer than 0.1 seconds. Both the event and shown
  context are checked. Missing geometry alone is not a geometry failure.

Canonical hand acceptance remains unchanged, including its existing minimum
sample confidence of 0.5. The profile does not tune that threshold to increase
coverage. Individual left/right coverage, both ANY-hand coverages, all-four
coverage, mapping and pose coverage, confidence, motion QC, and the independent
strict/demo decisions are retained in the demonstration manifest. Rejected
annotations remain in the manifest with explicit reasons.

## Rendering and reporting

Every hand is drawn only for frames with accepted source geometry. Missing or
rejected hands are omitted; no interpolation, carry-forward, synthetic hands,
or guessed skeletal connections are introduced. A small “Partial hand tracking”
caution identifies clips with missing hands. The fixed-view three-panel style
and exclusion of the unresolved scan layer remain unchanged.

The demonstration output directory is separate from the strict artifacts:

```text
outputs/v0_demo/<recording-id>/
  examples_demo_qc/handover_<annotation-id>_<object>.mp4
  demo_qc_manifest.csv
  demo_qc_validation.json
  duet_v0_montage_demo_qc.mp4
```

The montage contains a four-second excerpt centered on each passing event,
with the source labels retained and transitions supplied by the existing
compilation renderer. All emitted MP4s undergo full decode validation.

Metrics for a mapping-rejected annotation may be reported from the already
validated arrays as diagnostics. Such a report grants no playback eligibility;
the unverified context remains rejected in both profiles.

Use the existing exporter with an explicit profile; this command reuses the
cached validation reports and never performs VRS matching:

```bash
.venv/bin/python scripts/export_comind_v0_examples.py \
  --recording-id 2fc0aa53-9070-4c86-81c2-41450253c74d \
  --qc-profile demo \
  --output-dir outputs/v0_demo/2fc0aa53-9070-4c86-81c2-41450253c74d \
  --workers 4 --reuse-clips
```

`--reuse-clips` requires matching render identities and file hashes and repeats
output decode validation. The default profile remains strict; selecting demo
does not overwrite strict clips, manifests, or validation reports.

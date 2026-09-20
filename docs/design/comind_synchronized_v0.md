# Synchronized CoMind Duet V0

Recording `43276420-701f-4731-b9ab-bebc7fd14994` now has complete, evidence-backed MP4-to-VRS frame maps for both participants and a generated synchronized handover viewer. **All 21,109 MP4 frames per participant have a VERIFIED local DEVICE_TIME value.** No UTC fit, constant offset, or frame-rate calculation supplies those timestamps.

The [focused Rerun recording](../../outputs/comind_visualization/43276420-701f-4731-b9ab-bebc7fd14994_synchronized_handover.rrd) contains frames **18120–18372**: 253 paired frames around bowl handover **018240**, including four seconds of video context on each side. Rerun's `rrd verify` reports no errors. The file is 107,592,844 bytes. It includes both ego images, both device and camera poses, calibrated nonlinear frustum outlines, both participants' left/right hands, trajectories, annotation/category, local timestamps, confidence, residuals, and missing-data warnings. The scene remains disabled because its coordinate registration is unresolved.

## Native RGB inventory and exact timestamps

The user supplied the two locally downloaded VRS files and reported that official downloader checksum verification passed. This pass did not copy or rehash those 22.79 GiB files. It used Project Aria 2.3.0, read source metadata without requesting decoded pixels, and checked unchanged source sizes/modification times. Camera RGB is stream `214-1`, label `camera-rgb`, with native 1408×1408 Fisheye624 calibration.

| Native observation | Helper | Leader |
| --- | ---: | ---: |
| RGB data-record count | **21,045** | **21,112** |
| First DEVICE_TIME (ns) | 919390269025 | 825528737150 |
| Last DEVICE_TIME (ns) | 1620744706862 | 1529116146612 |
| Capture interval median (ns) | 33,327,337 | 33,327,337 |
| Capture interval P95 (ns) | 33,332,200 | 33,341,662.5 |
| Capture interval maximum (ns) | 33,752,050 | 33,765,750 |
| Decreasing / duplicate native timestamps | 0 / 0 | 0 / 0 |
| Native source frame-number range | 1927–22971 | 2197–23308 |
| Non-unit source frame-number steps | 0 | 0 |

The fractional P95 is a statistical interpolation, not a fractional stored timestamp. Every stored capture timestamp is the original int64 nanosecond value. Zero-based RGB record index, native source `frame_number`, MP4 ordinal, and MP4 PTS remain distinct fields. Metadata-only record reads cross-checked every native capture timestamp against the SDK array while recovering source frame numbers.

Evidence: [native VRS report](../../outputs/comind_vrs/43276420-701f-4731-b9ab-bebc7fd14994.json). Compact arrays and native/export calibration JSONs are in `data/processed/comind/<UUID>/vrs/`.

## Exact mapping model and confidence

The mapping is an explicit **monotonic per-frame lookup with repeated source indices and skipped native records**. It is not one constant offset and does not assume MP4 ordinal equals VRS ordinal. Each VERIFIED row has a directly compared image, a unique accepted local candidate, and the exact timestamp of that native record. All frames remain present in the map, including duplicates.

| Mapping observation | Helper | Leader |
| --- | ---: | ---: |
| MP4 rows | 21,109 | 21,109 |
| VERIFIED high-confidence direct matches | **21,109** | **21,109** |
| INFERRED / UNRESOLVED rows | **0 / 0** | **0 / 0** |
| First → last matched VRS index | 7 → 21044 | 0 → 21111 |
| Distinct VRS records matched | 21,032 | 21,108 |
| Adjacent repeated source-index transitions | 77 | 1 |
| Skipped native records between matches | 6 | 4 |
| Native records preceding first match | 7 | 0 |
| Decreasing assigned source indices | 0 | 0 |

Helper MP4 frames **21033–21108** all match VRS RGB record **21044**. These 76 output frames therefore preserve the same source DEVICE_TIME, **1620744706862 ns**, rather than fabricated increments. This accounts for 75 of the 77 repeated transitions; two occur elsewhere. Leader ends at its last native RGB record, 21111. Native capture counts differ from the MP4 counts without leaving unassigned MP4 images.

There is no separate bit-identical-image count: the images are re-encoded, and exact pixel equality was not the criterion. All 42,218 correspondences satisfy the direct robust matching gate. No interpolated capture timestamps are created. The matches explain the exports using individual source images; they do not identify every operation in the unpublished export implementation.

The efficient matching process was:

1. Decode each MP4 once into compact 32×32 grayscale fingerprints, retaining actual decoder ordinals and exact integer PTS/time base.
2. Test all four quarter-turn orientations at 61 sparse anchors per participant, spanning start/end, approximately 30-second intervals, and all six handover boundaries/context windows. All 122 anchors favor clockwise 90° rotation.
3. Use those anchors to bound local image searches, then verify every output frame against native RGB fingerprints. Interpolated anchor indices only restrict candidate searches; they never supply source identities or timestamps.
4. Reject ambiguous, high-error, search-edge or nonmonotonic candidates instead of assigning them. No such unresolved rows remain in this recording. Native fingerprints are cached once, and future matching uses the compact caches.

The configurable acceptance thresholds are grayscale RMSE ≤1.5, best/second error separation ≥0.08, and second/best error ratio ≥1.15; exact ties are always rejected. Worst observed RMSE is approximately 0.552 helper / 0.555 leader; minimum separation is 0.1515 / 0.1412, and minimum ratio is 1.360 / 1.342. Confidence is `1 - best_RMSE / second_RMSE`, an image-separation score, **not a probability**.

Source mappings are stored in `frame_maps/helper_frame_map.npz` and `leader_frame_map.npz`, with JSON provenance and SHA-256 validation. Fields include every MP4 ordinal, original PTS and rational time base, native RGB index, exact DEVICE_TIME, classification, confidence, residual image errors, repeat flags, skip counts, and mapping reason. The loader cross-checks assigned values against native SDK metadata. [Helper map evidence](../../data/processed/comind/43276420-701f-4731-b9ab-bebc7fd14994/frame_maps/helper_frame_map.json), [leader map evidence](../../data/processed/comind/43276420-701f-4731-b9ab-bebc7fd14994/frame_maps/leader_frame_map.json).

## Temporal validation and whole-video coverage

The authoritative cross-person timeline is **`comind_sync_frame_index`**, under CoMind's paired-video contract and the user's explicit instruction. At each sequence index, helper and leader independently select their own mapped native capture times, then their own Multi-SLAM poses and hands. Their numeric device-clock values are never subtracted or treated as one physical clock. A shared spatial graph alone is not used as synchronization evidence.

True nearest-pose matching uses all 721,702 helper and 724,987 leader trajectory rows, read once and cached. The older approximately 30 Hz trajectory subsets are not used to compute these residuals. Full existing hand caches are reused, including their source confidence/missing state and the residual of the original hand-to-world pose association.

| Stream | Accepted / 21,109 | Coverage | Median residual (ms) | P95 (ms) | Maximum accepted (ms) | Gap rejections |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Helper pose | 21,008 | 99.5215% | 0.095350 | 0.097925 | 0.363850 | 101 |
| Helper hands | 21,011 | 99.5357% | 0.095850 | 0.097994 | 0.519475 | 98 |
| Leader pose | 21,075 | 99.8389% | 0.094050 | 0.098475 | 0.459425 | 34 |
| Leader hands | 21,077 | 99.8484% | 0.094850 | 0.098888 | 0.531763 | 32 |

Residuals above are absolute source-sample-minus-video-capture differences among accepted matches. Full signed residual arrays and all rejected nearest candidates are retained. The limits are **2 ms for pose**, **20 ms for hands**, and **10 ms for the hand's secondary pose association**. These are sample-association residuals, not an independently measured bound on physical cross-person synchronization accuracy.

Rejected ranges, inclusive:

| Stream | Rejected sync-frame ranges | Largest candidate residual (ms) |
| --- | --- | ---: |
| Helper pose | 0–21; 21030–21108 | 733.311513 |
| Helper hands | 0–20; 21032–21108 | 699.983513 |
| Leader pose | 0–29; 21105–21108 | 966.608850 |
| Leader hands | 0–29; 21107–21108 | 966.608850 |

These missing endpoint observations are not extrapolated. A verified video timestamp does not imply an available pose or hand. Hand-row matching also does not imply both hands were tracked: whole-video high-confidence landmark coverage, with threshold 0.5, is helper left **96.2954%**, helper right **97.4655%**, leader left **96.8355%**, leader right **93.6425%**. Missing landmarks and low-confidence hands are cleared in playback.

Evidence: [temporal validation and all six handovers](../../outputs/comind_frame_validation/43276420-701f-4731-b9ab-bebc7fd14994.json). Every frame has derived pose/hand indices, signed residuals, acceptance masks and confidence under `frame_validation/<role>_aligned.npz`. Hashes and a generation ID prevent mixing stale arrays with newer frame maps.

## Annotation binding and selected handover

The six `skip == false` handover intervals are bound directly to `comind_sync_frame_index` under the explicit task contract. The earlier `frame / 30 == time` audit is repeated as a consistency check; that arithmetic does not create DEVICE_TIME. Source initiator/delivery labels remain uninterpreted, and no `left`/`right` source label is converted into helper/leader identity. Display and QC include both boundary indices as an explicit implementation policy; the dataset's endpoint-inclusion semantics remain unclaimed.

All six proposed context windows have complete VERIFIED mappings. Selection ranks: both-participant hand coverage, all-four-hand coverage, pose coverage, the lower participant median image-match confidence, hand confidence, temporal residuals, cross-person wrist distance, then motion outliers. The ranking is reproducible and uses wrist landmark index **5**, not thumb tip index 0.

| Handover | Event frames | All-four-hand high-confidence coverage | Lower participant median map confidence | Median nearest cross-person wrist distance |
| --- | --- | ---: | ---: | ---: |
| **018240** | **18240–18252** | **100%** | **0.8871** | **0.1792 m** |
| 015185 | 15185–15199 | 100% | 0.8646 | 0.0103 m |
| 015204 | 15204–15212 | 100% | 0.8230 | 0.0095 m |
| 018260 | 18260–18286 | 100% | 0.8155 | 0.1769 m |
| 002131 | 2131–2366 | 95.76% | 0.8724 | 0.3197 m |
| 018941 | 18941–18994 | 74.07% | 0.9095 | 0.0235 m |

Every event has 100% pose coverage. Selected **018240** is a **bowl / bowl / serveware** handover with `initiation_type=["verbal"]`, source labels `initiator="left"` and `delivering_flow="ltr"`. It wins the image-confidence tie-break among the four events with complete hand and pose coverage. Its combined median hand confidence is **0.9997775**, combined P95 temporal residual **0.0978 ms**, closest cross-person wrist separation **0.06445 m**, and wrist-speed outlier count **0** at the explicit 5 m/s threshold.

Calibrated wrist projections are reported for every handover in both ego cameras. During 018240, all four wrists project inside the leader image in all 13 event frames. In helper's image the corresponding counts are **12, 7, 13, 13**, ordered helper left/right, leader left/right. All eight projections are inside their images at midpoint frame 18246. This is **projection-domain coverage, not occlusion-aware visibility**; candidates 015204 and 018260 have better complete in-image coverage and are not claimed to lose on that metric. The selected event follows the broader stated ranking.

See the [projection report](../../outputs/comind_v0/43276420-701f-4731-b9ab-bebc7fd14994_projection_qc.json) and [paired-image wrist reprojection figure](../../outputs/comind_v0/43276420-701f-4731-b9ab-bebc7fd14994_wrist_projection.svg). The figure uses actual selected MP4 frames and projected 3D wrists, with no inferred participant labels from annotations.

## RGB calibration and scan status

**RGB orientation/calibration: VERIFIED for the tested export geometry.** Native RGB images match the MP4 after clockwise 90° rotation. Beginning/middle/end full-resolution matches for both roles favor zero translation over all eight ±1-pixel shifts. With a symmetric smoothing control for codec noise, unit scale beats 0.999 and 1.001 in all six pairs. These checks support the pure rotation model against the tested alternatives; they do not claim to exclude arbitrarily tiny image warps.

The exported projection preserves the native nonlinear Fisheye624 model and applies the exact pixel-center map `(u, v) -> (height - 1 - v, u)`. Its unit rays and `T_device_camera` retain the explicitly named native camera frame. Numeric checks agree with Project Aria's clockwise calibration utility and project/unproject round trips to below **1.2e-11 pixels**. This is mathematical implementation consistency, not physical calibration accuracy. The installed SDK supports the Fisheye624 rotation despite a stale Linear-only docstring; the real calls and equivalence tests are recorded.

Camera composition is `T_shared_world_camera = T_shared_world_device @ T_device_camera`. Frustums use 64 rays on the actual SDK-valid image contour; no pinhole approximation is introduced. Native source units are meters. Calibration caches include the anchor and full-resolution evidence needed by the playback gate.

**Scan registration: UNRESOLVED; scene disabled.** The previously measured leader standard-MPS→Multi-SLAM bridge is reused: held-out translation median/P95/max **0.0613 / 0.1387 / 6.2679 mm**, rotation **0.00557 / 0.02435 / 0.15523°**. Native VRS calibration does not establish which producer graph, units or already-applied transform belong to the entire `aria_semidense_points.ply` or `blk_scan_aria_aligned.ply`. No scene edge is inserted, and the nominal BLK transform is not applied again. No scan vertices or large point-cloud CSVs were reread. [Scan gate report](../../outputs/comind_v0/43276420-701f-4731-b9ab-bebc7fd14994_scan_gate.json).

## Reproduction

From the repository root, regenerate the focused RRD from the completed derived caches with one command:

```sh
.venv/bin/python scripts/visualize_comind.py \
  --recording-id 43276420-701f-4731-b9ab-bebc7fd14994 --synchronized
```

Open and verify it:

```sh
.venv/bin/rerun outputs/comind_visualization/43276420-701f-4731-b9ab-bebc7fd14994_synchronized_handover.rrd
.venv/bin/rerun rrd verify outputs/comind_visualization/43276420-701f-4731-b9ab-bebc7fd14994_synchronized_handover.rrd
```

Reproduce processing stages in dependency order when needed. The local SDK inventory command reads metadata; the matcher reuses fingerprint/pixel-audit caches when their source identities match. On first use it streams images and keeps only compact fingerprints. The temporal validator reuses full-rate pose and hand caches rather than rescanning CSVs.

```sh
.venv/bin/python scripts/inspect_comind_local_vrs.py --recording-id 43276420-701f-4731-b9ab-bebc7fd14994
.venv/bin/python scripts/build_comind_frame_map.py --recording-id 43276420-701f-4731-b9ab-bebc7fd14994 --audit-image-geometry
.venv/bin/python scripts/verify_comind_rgb_calibration.py --recording-id 43276420-701f-4731-b9ab-bebc7fd14994
.venv/bin/python scripts/validate_comind_frame_map.py --recording-id 43276420-701f-4731-b9ab-bebc7fd14994
.venv/bin/python scripts/validate_comind_projection.py --recording-id 43276420-701f-4731-b9ab-bebc7fd14994
.venv/bin/python scripts/visualize_comind.py --recording-id 43276420-701f-4731-b9ab-bebc7fd14994 --synchronized
```

The optional original source-frame-number audit reuses the earlier compact VRS index cache under `outputs/comind_sync_forensics/remote_vrs`; exact SDK capture timestamps do not depend on assuming source frame numbers. No full VRS export is made. The existing diagnostic mode is retained:

```sh
.venv/bin/python scripts/visualize_comind.py --recording-id 43276420-701f-4731-b9ab-bebc7fd14994 --diagnostic
```

## Files and verification

New reusable implementation:

- `src/duet/adapters/comind/vrs.py`: metadata-only exact RGB timestamps, native frame metadata and calibrated export projection.
- `src/duet/adapters/comind/frame_map.py`: robust image fingerprints, local candidate gates, immutable mappings and checksum validation.
- `src/duet/adapters/comind/temporal_validation.py`: full-rate local matching, annotation binding and quantitative handover ranking.
- `src/duet/adapters/comind/playback.py`: generation/hash-checked canonical episode assembly with independent device clocks.
- `src/duet/adapters/comind/indexed_video.py`: bounded selected-window decoding against actual cached PTS.
- `src/duet/qc/projection.py`: explicit-frame projection-domain coverage without visibility/occlusion claims.
- `src/duet/visualization/comind_handover.py`: paired-index RRD rendering with confidence/gap gates and stale-data clearing.

New CLIs: `inspect_comind_local_vrs.py`, `build_comind_frame_map.py`, `verify_comind_rgb_calibration.py`, `validate_comind_frame_map.py`, and `validate_comind_projection.py`. Existing `visualize_comind.py` gains `--synchronized` while retaining `--diagnostic` and no force option.

New synthetic tests: `test_comind_vrs.py`, `test_comind_frame_map.py`, `test_comind_temporal_validation.py`, `test_comind_playback.py`, `test_comind_indexed_video.py`, `test_comind_handover_viewer.py`, and `test_projection_qc.py`. Existing `test_visualize_comind_cli.py` adds dispatch/mode tests. This report and the current-status sections of `docs/datasets/comind_format.md` document the new evidence. Derived caches/reports and the RRD/figure remain outside raw data.

The full suite passes **648 tests**. Ruff check and formatting checks pass for `src/`, `tests/`, and all six changed/new CLIs. Repository-wide Ruff reports **33 pre-existing findings** in untouched files: 12 in `scripts/comind_download.py` and 21 in `scripts/inspect_comind.py`. The regenerated RRD passes `rerun rrd verify`, and playback verifies that current map, native timestamp, and aligned-cache hashes agree. No dependency was added, no existing raw file was modified, and no VRS file was copied. Original trajectory/hand adapters and the recovered helper trajectory were reused.

Remaining limits are explicit: whole-cloud scan registration, occlusion-aware visibility, annotation participant-label and box semantics, and dataset endpoint-inclusion semantics. Pose/hand endpoint gaps and low-confidence/missing hand samples remain data limitations. The MP4-to-DEVICE_TIME correspondence is no longer blocked.

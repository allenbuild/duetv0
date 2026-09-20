# CoMind per-frame timestamp and world-alignment investigation

Recording: `43276420-701f-4731-b9ab-bebc7fd14994`.

The official Project Aria extraction call succeeds, but neither selected MP4
contains a complete DEVICE_TIME array. Real synchronized video/3D playback
remains blocked. No VRS download was started and no raw data was modified.

## Common video index is supported; device bindings are missing

The [official CoMind page](https://comind.ethz.ch/) documents paired ego streams
as synchronized, frame-aligned, and hardware-timestamped. The
[paper](https://arxiv.org/html/2607.06691#S3.SS2) also describes synchronous
recording initiation. This supports a common `comind_sync_frame_index`; UTC
need not be its authority. The required path for 3D remains:

```text
common paired video frame i
  -> helper DEVICE_TIME[i] -> helper local pose/hand lookup
  -> leader DEVICE_TIME[i] -> leader local pose/hand lookup
```

Helper and leader device timestamps are never compared numerically to each
other. `PairedFrameTimeline` implements this contract with synthetic tests,
requiring two validated complete arrays and specifically scoped verified
paired-frame-alignment evidence. No real binding is constructed from the
incomplete metadata found here.

## Actual official extraction result

Project Aria Tools was absent. The authorized minimal installation added
`projectaria-tools==2.3.0` with `--no-deps` and its required MoviePy import
dependencies: `moviepy==2.2.1`, `decorator==5.3.1`, `ImageIO==2.37.4`,
`proglog==0.1.12`, `python-dotenv==1.2.3`, and `tqdm==4.70.1`. Existing NumPy
and Pillow were retained; `pyproject.toml` was not changed. An already installed
Kite application's `ffprobe` 6.0 supplied the required executable; no binary
download was needed. These packages are optional for the extraction probe;
the normal synthetic unit suite does not require their installation.

The [official documentation](https://facebookresearch.github.io/projectaria_tools/docs/data_utilities/advanced_code_snippets/vrs_to_mp4)
uses the historical `projectaria_tools.utils.vrs_to_mp4_utils` path. Version
2.3.0 exposes the actual function at
`projectaria_tools.tools.vrs_to_mp4.vrs_to_mp4_utils.get_timestamp_from_mp4`.
That unmodified official function was called on all four local ego MP4s.

| MP4 | Declared frames | Returned array length | First = last DEVICE_TIME (ns) |
| --- | ---: | ---: | ---: |
| `helper_trimmed_sync.mp4` | 21,109 | 1 | 919623571487 |
| `leader_trimmed_sync.mp4` | 21,109 | 1 | 825528737150 |
| `helper_sync.mp4` | 23,295 | 1 | 855400519487 |
| `leader_sync.mp4` | 23,295 | 1 | 752640406450 |

Every official return is a NumPy `int64` array with shape `(1,)`. The container
description is a bare scalar integer. The official parser accepts it as a
single-element array after its bracket stripping and comma splitting. Success
of the extraction call therefore does not mean per-frame metadata is complete.

Both selected arrays fail the required count of 21,109. Each has zero duplicate
values and zero timestamp intervals; monotonicity and interval median/P95/max
are unassessable. The helper scalar is 733,311,513 ns before its first standard
MPS/Multi-SLAM timestamp. The leader scalar is 966,608,850 ns before its first
standard MPS/Multi-SLAM timestamp. Neither lies inside the corresponding
trimmed trajectory range. Their relationship to trimming remains unverified.
No start offset, extrapolation, PTS replacement, or repeated scalar is used.

The strict wrapper preserves original integer nanoseconds, validates array
count/order, preserves legitimate duplicate captures, and rejects incomplete
arrays. Probe results are saved under `outputs/comind_timestamp_probe/`.

Reproduce the header-only official extraction on this machine:

```bash
.venv/bin/python scripts/probe_comind_mp4_timestamps.py \
  --recording-id 43276420-701f-4731-b9ab-bebc7fd14994 \
  --validation-report outputs/comind_validation/43276420-701f-4731-b9ab-bebc7fd14994.json \
  --ffmpeg-directory /Applications/Kite.app/Contents/Resources/binaries/ffmpeg
```

On another machine, supply the directory containing an existing `ffprobe`, or
omit that option if it is on `PATH`. The optional packages and executable are
needed only for the real official extraction call. The script does not install
anything or decode full videos.

## Annotation and calibration evidence

For all six usable handovers, the twelve source boundaries agree with
`frame / 30`:

| Absolute residual (seconds) | Median | P95 | Maximum |
| --- | ---: | ---: | ---: |
| Zero-origin formula | 2.0e-14 | 4.833333333333333e-14 | 6.666666666666667e-14 |
| One-origin formula `(frame - 1) / 30` | 0.03333333333332833 | 0.03333333333336667 | 0.03333333333336667 |

There is no consistent off-by-one discrepancy relative to the stored annotation
seconds. The zero-origin arithmetic is supported; the exact annotated video,
trim origin, and endpoint inclusion are still unverified. This does not install
an active-annotation mapping or interpret boxes/participant labels.

The official Aria upright helper rotates pixels clockwise 90 degrees and updates
both calibration intrinsics and extrinsics. Its documented implementation uses
`np.rot90(image, k=3)` and `calibration.rotate_camera_calib_cw90deg`, supporting
pinhole and Fisheye624. See the
[pinned helper](https://github.com/facebookresearch/projectaria_tools/blob/99e68c8aeb26f270933c88cd6f77f8fc1d137c13/projectaria_tools/utils/calibration_utils.py#L54-L65).
No evidence establishes that this exact export chain produced the CoMind MP4s.
The native calibration is therefore not applied directly to upright pixels;
frustums and numerical MP4 reprojection remain gated.

## Standard leader MPS world to shared Multi-SLAM world

The study reuses 21,077 cached Multi-SLAM poses and reads the standard leader
trajectory once. All timestamp matches are accepted within 1 ms; absolute
residual median/P95/max is 250/475/500 microseconds in the same leader clock.
The fit uses both translation and orientation with Huber weighting and fixed
scale one, training on 16,861 pairs and holding out 4,216 pairs before fitting.

| Held-out residual | Median | P95 | Maximum |
| --- | ---: | ---: | ---: |
| Translation (m) | 0.0000613283 | 0.000138671 | 0.006267885 |
| Rotation (degrees) | 0.0055663 | 0.0243545 | 0.1552268 |

Ten time bins agree closely: maximum local-fit versus global-fit discrepancy is
0.0003248 m and 0.019625 degrees. A separate scale diagnostic gives 1.000592
(0.0592% from one); it is not applied. The output is strictly SE(3), not Sim(3).
This strongly supports a constant numerical world bridge over the tested poses.
It is an estimated transform with reported errors, not a declaration that the
entire PLY belongs to that source frame. Neither scene registration nor a scan
frame-graph edge is enabled. The already-aligned BLK matrix is not applied again.

Report: `outputs/comind_world_alignment/43276420-701f-4731-b9ab-bebc7fd14994.json`.
Matched poses are cached under `data/processed/comind/<UUID>/alignment/`.
The study does not reread Multi-SLAM CSVs, scan points, semidense observations,
videos, or VRS.

## Small metadata fallback and optional VRS assets

A metadata-only solution remains possible: obtain the exact original
DEVICE_TIME for every frame of each actual trimmed MP4, with export/trim
provenance and duplicate/dropped-frame handling. The two raw int64 timestamp
arrays alone would total 337,744 bytes. Export calibration and annotation
contracts can also be supplied as small metadata. No such time sidecar appears
in the inspected recording manifest.

It is **not established that full VRS files are necessary**. VRS would provide
native capture/time-sync records, but would not by itself prove correspondence
to every frame of a later trimmed/re-encoded MP4. No large download is justified
automatically by this probe.

For a future explicitly chosen VRS fallback, the local manifest lists:

- `trimmed_vrs/helper_trimmed.vrs`: 12,572,405,231 bytes.
- `trimmed_vrs/leader_trimmed.vrs`: 11,901,082,187 bytes.
- Total: **24,473,487,418 bytes = 22.792711 GiB**.

The prepared one-recording ID file is
`outputs/comind_timestamp_probe/selected_recording.txt`. The following command
was **not run**. Its separate staging target preserves existing raw files:

```bash
.venv/bin/python scripts/comind_download.py \
  outputs/comind_timestamp_probe/selected_recording.txt \
  --target data/downloads/comind_vrs --parts vrs --no-meshes --no-annotations
```

No handover is selected for synchronized 3D playback. The existing static
Rerun diagnostic remains available; the real synchronized viewer upgrade is
stopped at the missing per-frame timestamp gate.

Final validation: **508 tests passed**. Ruff lint and formatting pass for all
implementation/tests and the five validation/probe/viewer scripts. The
pre-existing 33 downloader/inspector lint findings were left untouched. New
tests cover official extraction/count gates, participant-local frame lookup,
annotation arithmetic, rigid pose alignment, and prevention of unverified
assembly. No real rotated-calibration, scan, or synchronized-viewer claim is
made where the required evidence is still missing.

# CoMind Dataset Format

This document records the CoMind v1 structures used by Duet.

Local inspection and official documentation are distinguished below. Observed
filenames, numeric values, and mathematical validity do not independently prove
coordinate registration or clock synchronization. Unresolved semantics remain
explicit; no learned perception is used by the adapter.

**Current V0 status:** full local VRS inspection and direct image matching now
verify every one of the 21,109 synchronized MP4 frames per participant against
an exact native DEVICE_TIME value. The paired-index Rerun handover viewer is
generated, including calibrated clockwise-rotated RGB geometry. Scan registration
remains gated. The [current consolidated report](../design/comind_synchronized_v0.md)
contains mapping counts, residuals, handover selection and reproduction commands;
the earlier audits below retain their original measurement scope.

## Dataset version

CoMind v1.

Dataset source:

[CoMind project](https://comind.ethz.ch/) and
[paper, data processing section](https://arxiv.org/html/2607.06691#S3.SS3).

Downloader:

`scripts/comind_download.py`

Selected Duet development recording:

`43276420-701f-4731-b9ab-bebc7fd14994`

## Local dataset root

The dataset is stored under:

`data/raw/comind/`

Raw dataset files must remain immutable.

High-level organization:

- `recordings/<RECORDING_ID>/`
- `annotations/`
- `meshes/`
- `split/`

## Split files

The downloader generated:

- `data/raw/comind/split/train.txt`
- `data/raw/comind/split/test.txt`

The selected development recording is from the training split.

## Global annotations

Downloaded annotation files:

- `data/raw/comind/annotations/dataset_handover_consolidated.json`
- `data/raw/comind/annotations/dataset_joint_attention_consolidated.json`
- `data/raw/comind/annotations/dataset_scoia_consolidated.json`

These are shared dataset-level assets.

## Handover annotation structure

Verified from:

`dataset_handover_consolidated.json`

Top-level fields:

- `dataset_version`
- `file_timestamp`
- `data`

`data` is a dictionary keyed by recording UUID.

The inspected release has an object at `data[recording_uuid]`, keyed by
opaque source segment strings such as `002131`. The adapter accepts this native
layout by default, preserving key order and retaining keys as annotation IDs.
Keys are not converted into frame numbers. Each value contains fields including:

- `start_frame`
- `end_frame`
- `start_time`
- `end_time`
- `bbox`
- `delivering_flow`
- `initiation_type`
- `initiator`
- `object_category_level_1`
- `object_category_level_2`
- `object_category_level_3`
- `transcript_10s`
- `numeric_id`
- `description`
- `annot_type`
- `skip`

For Duet V0, only entries where:

`skip == false`

should be treated as usable annotations.

## Verified handover fields

### `start_frame`

Beginning frame index of the annotated handover interval.

### `end_frame`

Ending frame index of the annotated handover interval.

### `start_time`

Start time of the handover annotation in seconds.

### `end_time`

End time of the handover annotation in seconds.

### `delivering_flow`

Observed values:

- `ltr`
- `rtl`

The exact mapping to physical participant identity must still be verified.

### `initiator`

Observed values:

- `left`
- `right`

The exact mapping to the participant or camera streams must still be verified.

### `initiation_type`

Observed values include:

- `verbal`
- `gestural`
- `implicit`

Multiple initiation types may occur for one handover.

### Object categories

Three hierarchy levels are available:

- `object_category_level_1`
- `object_category_level_2`
- `object_category_level_3`

### `bbox`

A four-value bounding box is present for at least some handovers.

The associated image stream, pixel convention, and coordinate ordering remain unresolved.

### `transcript_10s`

Text associated with the surrounding interaction.

This is not required for the first geometry milestone.

## Selected recording handovers

Selected recording:

`43276420-701f-4731-b9ab-bebc7fd14994`

Verified usable handovers:

6

Verified unique level-1 object categories:

3

Examples:

- bowl
- carrot peel
- spoon

Median usable handover duration:

approximately 0.67 seconds.

## Local inventory and integrity

The original pre-VRS inspection contained 45 files, totaling 15,777,200,618 bytes
(14.69 GiB), with no filenames recognized as partial downloads by
`scripts/inspect_comind.py`. This inventory is **not** proof that every archive
is complete. The requested components are `mp4`, `mps`, `multislam`, and `scan`;
raw VRS files were absent at that stage. The six transcript files and both full
trimmed VRS files are now local. The user reported official downloader checksum
verification for both VRS files; this implementation did not copy or modify them.

The helper `multislam_output/0/slam.zip` is truncated: its size is
1,237,340,583 bytes and it has no ZIP end-of-central-directory record. One later
member extends beyond the available file. Python's standard ZIP reader rejects
the archive. Its local `closed_loop_trajectory.csv` member is fully present:
local header offset 39,396,361, compressed size 40,807,069 bytes, declared
uncompressed size 228,136,335 bytes. A complete member may be recovered into
`data/processed/` only after verifying its declared size and CRC. Such recovery
does not make the raw archive complete or validate its other members.

The inspection report also records a ZIP-reader failure for
`multislam_output/summary.json.zip`; this file is not required by the adapter.
Raw files are opened for reading only. No archive is extracted into `data/raw/`.

## Verified recording paths and identities

All paths below are relative to
`data/raw/comind/recordings/43276420-701f-4731-b9ab-bebc7fd14994/`.
The participant identities are explicitly supplied for this integration:
`helper` maps to `participant/helper`, and `leader` to `participant/leader`.
They are independent of annotation `left`/`right` labels.

```text
mp4s/
  helper_trimmed_sync.mp4
  leader_trimmed_sync.mp4
  helper_sync.mp4
  leader_sync.mp4
  gopro_front_sync.mp4
  gopro_back_sync.mp4
  merge_gaze_sync.mp4
mps_helper_trimmed_vrs/             # equivalent leader directory also present
  hand_tracking/hand_tracking_results.csv
  hand_tracking/summary.json
  eye_gaze/general_eye_gaze.csv
  eye_gaze/summary.json
  slam/closed_loop_trajectory.csv
  slam/online_calibration.jsonl
  slam/open_loop_trajectory.csv
  slam/semidense_points.csv.gz
  slam/semidense_observations.csv.gz
  slam/summary.json
multislam_output/
  vrs_to_multi_slam.json
  0/slam.zip                      # helper; truncated container
  1/slam.zip                      # leader
  1/slam/closed_loop_trajectory.csv
  1/slam/online_calibration.jsonl
  1/slam/open_loop_trajectory.csv
  1/slam/semidense_points.csv.gz
  1/slam/semidense_observations.csv.gz
  1/slam/summary.json
scan/
  T_ariaWorld_from_blkWorld.txt
  aria_semidense_points.ply
  blk_scan_aria_aligned.ply
```

The local `vrs_to_multi_slam.json` maps the recording's
`trimmed_vrs/helper_trimmed.vrs` to folder `0` and
`trimmed_vrs/leader_trimmed.vrs` to folder `1`. These exact role basenames and
the mapping establish discovery; arbitrary filename guesses are not used.

## MP4 video: verified metadata and reconstructed device-time mapping

Both selected ego MP4s contain one H.264 video stream and one audio stream,
1408 × 1408 pixels,
average rate 30 fps, 21,109 declared frames, time base `1/15360` seconds, and
stream start PTS `0`. Their declared video duration is
`21109/30` seconds, approximately 703.633333 s. Container duration is a separate
quantity and can include audio: both report 703,660,998 microseconds. Observed
first frame PTS values are `0,512,1024`; the final frame PTS is `10807296`
(703.6 s). Stream/container metadata dictionaries contain no rotation entry;
that absence does not prove the exported pixels were never rotated.
Frame access preserves original integer PTS
and rational time base; `PTS * time_base` is video time, not device time.

The GoPro front/back streams are exocentric. The untrimmed `helper_sync.mp4`
and `leader_sync.mp4` report 776.5 s; `merge_gaze_sync.mp4` is a 2816 × 1408
composite. These are not selected as the two ego sources.

The [official project page](https://comind.ethz.ch/) explicitly documents the
paired ego streams as synchronized, frame-aligned, and hardware-timestamped.
This supports `comind_sync_frame_index` as a shared **video** sequence key.
UTC is not required as its authority. Attaching each participant's poses and
hands requires that video's complete frame-to-DEVICE_TIME correspondence. That
correspondence is now verified from native RGB images: helper uses VRS indices
7 through 21044 with 77 repeated-source transitions and six skipped native
records between matches; leader uses 0 through 21111 with one repeat and four
skips. All 42,218 output frames have direct high-confidence matches, with no
inferred/unresolved timestamp rows. Helper's final 76 MP4 images preserve the
same native capture time. Original MP4 PTS remain separate from DEVICE_TIME.

The [CoMind paper](https://arxiv.org/html/2607.06691#S3.SS3) describes TICSync
for the Aria pair, audio-based SyncSink alignment for GoPros, and manual
trimming to a common start. This verifies the production process, but does
not itself provide this recording's MP4 PTS-to-device-time mapping. Equal frame
counts, duration, or start PTS do not establish that mapping. Direct matching
and explicit source indices supply it. Native-to-MP4 clockwise 90-degree rotation
and the Fisheye624 export projection are verified by sparse orientation anchors,
bounded full-resolution pixel-geometry checks and official calibration utilities.
The shared timeline is a paired frame index, never an affine merge of device clocks.

## Standard MPS trajectories

Both `mps_<role>_trimmed_vrs/slam/closed_loop_trajectory.csv` files contain
`graph_uid`, `tracking_timestamp_us`, `utc_timestamp_ns`,
`{tx,ty,tz,qx,qy,qz,qw}_world_device`, and `quality_score`.
The [Project Aria trajectory format](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/slam/mps_trajectory)
defines these poses as **T_world_device**, with translation in meters and
Hamilton quaternion components in the named CSV order `x,y,z,w`.
`tracking_timestamp_us` is device time in microseconds. UTC is an additional
nanosecond wall-clock value; `-1` denotes unavailable UTC. Quality scores have
range `[0,1]`; they are retained rather than treated as geometric accuracy.
The parser checks quaternion norm and rigid-matrix validity using an explicit
`1e-7` tolerance for these MPS exports. It does not normalize quaternions or
repair invalid rotations.

The standard trajectories are individual spatial solutions, represented in
`helper/mps_world/<graph_uid>` and `leader/mps_world/<graph_uid>` so different
graph islands remain distinct. Local helper graph ID is
`e7a29a45-6188-8c38-5041-0f9a26bcdd79` (700,322 rows); local leader graph ID is
`aa198227-fa0f-b328-06e1-a9a4b16e7565` (702,555 rows).
They are not edges connecting the two participants. The shared-world path
uses Multi-SLAM device trajectories instead; it never composes a standard
MPS world trajectory with a Multi-SLAM world trajectory.

## Hand representation and missing data

Each `mps_<role>_trimmed_vrs/hand_tracking/hand_tracking_results.csv` contains
21 landmarks per hand, indexed `0..20`, in columns
`{tx,ty,tz}_{left,right}_landmark_<index>_device`. Wrist transforms and wrist/palm
normals are also present. The
[official hand format](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/hand_tracking)
defines landmark coordinates in the device frame in meters, and supplies the
landmark ordering. Hand timestamps use the same participant device clock as
MPS trajectories.

`left_tracking_confidence` and `right_tracking_confidence` in `[0,1]` represent
available results; **exactly `-1` means that hand is missing** and its positions,
wrist pose, and normals must not be used. Local missing rows contain zero
coordinate placeholders, while zero-confidence rows can contain nonzero
coordinates. Zero confidence must therefore not be treated as the missing
sentinel. Missing hands use an explicit missing state, with no spatial payload.

For each participant independently, nearest matching selects that participant's
Multi-SLAM pose in the same device-time domain, applies an explicit maximum gap,
and reports the signed residual. Available points transform as
`p_shared = T_shared_world_device @ p_device`. Helper device timestamps are
never directly matched to leader device timestamps.

## Online camera calibration

The two standard `slam/online_calibration.jsonl` files have one calibration
record per line with `tracking_timestamp_us`, `utc_timestamp_ns`, and
`CameraCalibrations`. The observed labels are `camera-slam-left`,
`camera-slam-right`, and `camera-rgb`; their parallel `ImageSizes` values are
`[640,480]`, `[640,480]`, and `[1408,1408]`.
[Project Aria's sensor table](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/aria_vrs/aria_vrs_format)
identifies `camera-rgb` as the RGB camera. Selection requires that label in
the actual calibration rather than assuming an array index.

Verified JSON encoding from the official reader:

- `T_Device_Camera.Translation`: `[x,y,z]`.
- `T_Device_Camera.UnitQuaternion`: `[w,[x,y,z]]`, distinct from CSV order.
- `Projection.Name` and `Projection.Params`: retained together; the observed
  `FisheyeRadTanThinPrism` name corresponds to Fisheye624, not a pinhole model.
- `ImageSizes`: indexed consistently with `CameraCalibrations`.
- `ReadoutTimesSec`: pairs of camera index and readout seconds when present.
- `TimeOffsetSec_Device_Camera`: preserved without inventing an MP4 correction.

Evidence is the official
[sensor reader](https://github.com/facebookresearch/projectaria_tools/blob/99e68c8aeb26f270933c88cd6f77f8fc1d137c13/core/calibration/loader/SensorCalibrationJson.cpp),
[SE(3) JSON reader](https://github.com/facebookresearch/projectaria_tools/blob/99e68c8aeb26f270933c88cd6f77f8fc1d137c13/core/data_provider/json_io/JsonHelpers.h),
and [online calibration reader](https://github.com/facebookresearch/projectaria_tools/blob/99e68c8aeb26f270933c88cd6f77f8fc1d137c13/core/mps/OnlineCalibrationsReader.cpp).
Initial local calibration records contain UTC `-1`; this is retained in
`utc_timestamp_ns_raw`, while the associated `Timestamp.raw_value` is `None`.
It is never a usable cross-participant synchronization anchor.

With same-participant timestamp matching, the camera pose is
`T_shared_world_camera = T_shared_world_device @ T_device_camera`.
The [Project Aria transform convention](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/coordinate_convention/3d_coordinate_frame_convention)
and [MPS camera example](https://facebookresearch.github.io/projectaria_tools/docs/data_utilities/core_code_snippets/mps)
establish this direction and multiplication order. Camera extrinsics are not
the same quantity as a device trajectory.

## Multi-SLAM: shared space does not establish shared time

[Project Aria Multi-SLAM documentation](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/slam/mps_multi_slam)
defines the numbered recording folders and `vrs_to_multi_slam.json`. Its
trajectories have the standard SLAM schema and align multiple recordings in
a shared spatial frame. A matching `graph_uid` identifies the same world;
outputs may also contain multiple islands or consecutive graph IDs.

The adapter therefore checks the trajectories' graph IDs before assigning a
shared-world frame. A mere intersection of two multi-graph sets would not
justify treating every sample as one world. A directory or intact ZIP can
supply a trajectory. The damaged helper container requires the separately
reported member recovery described above. No standard single-recording MPS
world is substituted if Multi-SLAM cannot be read.

## Scan: verified matrix, unverified frame binding

The local transform text contains:

```text
-0.9928017634 -0.1197083691 -0.0038164150 -0.6142937981
 0.1196865468 -0.9927964403  0.0055098685  0.5734620340
-0.0044485006  0.0050134337  0.9999775379 -0.1923995649
 0.0000000000  0.0000000000  0.0000000000  1.0000000000
```

It passes rigid-matrix checks without repair at absolute tolerance `1e-8`:
maximum orthonormality residual `6.988856341851568e-11`, determinant
`1.0000000000269647`, and exact homogeneous final row. Both PLY files use
`binary_little_endian 1.0`, the comment `Created by Open3D`, and double-precision
`x,y,z` properties. `aria_semidense_points.ply` declares 304,681 vertices;
`blk_scan_aria_aligned.ply` declares 10,000,000 vertices and also has unsigned
byte RGB properties. Neither header establishes distance units or a Multi-SLAM
graph identity.

The [CoMind paper](https://arxiv.org/html/2607.06691#S3.SS3) describes
BLK2GO-to-Aria registration through image localization, robust initial
alignment, and point-to-plane ICP with manual verification. This supports
the intended registration process. It does not prove which local graph
`ariaWorld` refers to, the text file's precise direction/unit contract, or
whether applying it to an already aligned PLY would double-transform it.
The adapter records the matrix and PLY metadata without inserting an
unverified scan edge into the shared frame graph.

## Frame table

`<role>` is explicitly `helper` or `leader`. Transform notation maps source
column-vector coordinates into the destination; translations are canonical
meters only when the source unit is verified.

| Source frame | Destination frame | Transform or operation | Units | Clock | Evidence |
| --- | --- | --- | --- | --- | --- |
| `<role>/device` | `<role>/mps_world/<graph_uid>` | Standard CSV `T_world_device` | m | That participant's device time | Local named columns; Aria trajectory format |
| `<role>/device` | `comind/<recording_id>/multislam/<graph_uid>` | Multi-SLAM CSV `T_world_device` | m | That participant's device time | Mapping file, trajectory graph verification, Aria Multi-SLAM format |
| `<role>/camera-rgb` | `<role>/device` | Calibration `T_Device_Camera` | m | That participant's calibration device time | Observed label/JSON; official sensor and SE(3) readers |
| `<role>/camera-rgb` | Verified Multi-SLAM shared world | `T_shared_world_device @ T_device_camera` | m | Same-participant pose/calibration matches | Aria chaining convention; explicit matching residuals |
| Hand landmarks in `<role>/device` | Verified Multi-SLAM shared world | Apply matched `T_shared_world_device` | m | Same participant's hand/trajectory device time | Aria hand format and trajectory format |
| Filename-implied BLK scan frame | Filename-implied Aria scan frame | Parsed matrix only; no graph edge | Unresolved | Static registration metadata | Valid matrix; paper describes process, exact binding unverified |

## Clock table

| Source value | Unit | Domain and origin | Permitted comparison |
| --- | --- | --- | --- |
| MP4 PTS and `time_base` | Integer ticks and exact seconds/tick | Separate video stream clock; observed start PTS 0 | Within that stream |
| Trajectory `tracking_timestamp_us` | µs | Recording- and participant-specific device clock | That participant's MPS, Multi-SLAM, and hands |
| Hand `tracking_timestamp_us` | µs | Same participant's device clock | That participant's trajectory |
| Calibration `tracking_timestamp_us` | µs | Same participant's device clock | That participant's trajectory |
| `utc_timestamp_ns` | ns | Source wall clock; `-1` missing | Preserved as auxiliary evidence; no automatic clock mapping |
| Annotation `start_time`/`end_time` | s | Isolated annotation clock; origin unresolved | Within the annotation domain |

The adapter names the device clock
`comind/<recording_id>/<role>/device_time` and the auxiliary UTC clock
`comind/<recording_id>/<role>/utc_unverified`. The latter is deliberately
participant-scoped until its cross-device alignment accuracy is established.

[Project Aria timestamp documentation](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/aria_vrs/timestamps_in_aria_vrs)
states that sensors on one pair of glasses share device time. Multiple devices
require a timecode or TICSync mapping. A shared spatial graph does not supply
that mapping. UTC values alone do not establish adequate alignment precision;
the [official time-domain definition](https://github.com/facebookresearch/projectaria_tools/blob/99e68c8aeb26f270933c88cd6f77f8fc1d137c13/core/data_provider/TimeTypes.h)
describes UTC as only seconds-accurate. No offset is fitted from array indices
or the apparent start/end correspondence of streams.

## Reproducible real-data validation

Use the existing virtual environment from the repository root:

```bash
.venv/bin/python scripts/validate_comind_recording.py \
  --root data/raw/comind \
  --recording-id 43276420-701f-4731-b9ab-bebc7fd14994 \
  --recover-incomplete-zip \
  --output outputs/comind_validation/43276420-701f-4731-b9ab-bebc7fd14994.json
```

`--recover-incomplete-zip` explicitly permits recovery of the complete helper
trajectory member after the archive failure is surfaced. Recovery writes only
under `data/processed/comind/` by default, verifies uncompressed length and CRC,
and leaves the source archive unchanged. Omit the flag when the needed sources
are valid directories or intact ZIPs. Without the flag, an unreadable required
archive is an error, not a reason to substitute standard MPS trajectories.

The command prints its JSON report and optionally writes it to the specified
path. `--processed-root` and `--output` must remain outside raw storage.
The script compares raw file names, byte sizes, and modification timestamps
before and after validation; this detects inventory changes but is not a
cryptographic hash of every raw payload.

The report distinguishes these checks:

- Video metadata comes from container/stream headers; the validator does not
  decode entire videos into memory.
- Trajectory statistics audit every standard MPS and Multi-SLAM trajectory row,
  including timestamp extents, graph IDs, and transform validity.
- Every camera transform in every online calibration record is validated.
  Shared-world RGB camera **composition is exercised at the first and last
  calibration records only**; this is not a full camera-trajectory or image
  reprojection validation.
- Hand coverage for each side is the number of present hand samples divided
  by the number of rows in that participant's hand CSV. It is not a fraction
  of all MP4 frames or recording duration. Missing `-1` rows remain in the
  denominator, and zero-confidence present hands are counted separately.
- Every hand CSV timestamp is matched to the same participant's Multi-SLAM
  trajectory. Residual statistics contain one observation per source hand row,
  rather than double-counting its two hands. Available hand landmarks are
  transformed only for accepted matches. The default maximum absolute gap
  is `0.01` seconds and can be changed with `--max-hand-gap-seconds`.
- Scan checks cover the rigid matrix and PLY headers. They do not establish
  scan registration semantics or load the entire scan for spatial comparison.

The report does not prove cross-device synchronization, MP4 reprojection
correctness, or simultaneous two-person spatial plausibility. No viewer is
launched.

## Validated recording results

The command above completed successfully. Full machine-readable evidence is in
`outputs/comind_validation/43276420-701f-4731-b9ab-bebc7fd14994.json`.
The raw inventory check found 53 files across the dataset root unchanged in name,
size, and modification time; the selected recording itself has 45 files.

The helper trajectory was recovered to
`data/processed/comind/43276420-701f-4731-b9ab-bebc7fd14994/0/slam/closed_loop_trajectory.csv`:
228,136,335 uncompressed bytes, CRC32 `529bb29f`, from 40,807,069 compressed
bytes. The leader trajectory came from its existing extracted directory.
The helper raw archive remains incomplete.

Both Multi-SLAM trajectories contain exactly one graph:
`8652bfb3-cb32-09d6-5685-b5e6f271b6ee`. Their verified spatial frame is
`comind/43276420-701f-4731-b9ab-bebc7fd14994/multislam/8652bfb3-cb32-09d6-5685-b5e6f271b6ee`.
**Cross-participant time synchronization remains unverified.**

The two selected video headers match the metadata above: H.264, 1408 × 1408,
30 fps, 21,109 frames, PTS time base `1/15360`, start PTS `0`, duration
`10807808` ticks (`21109/30` seconds).

The following raw microsecond ranges are in each row's **own participant device
clock**. Subtracting helper values from leader values is not a synchronization
operation.

| Source | Rows | First tracking timestamp (µs) | Last tracking timestamp (µs) |
| --- | ---: | ---: | ---: |
| Helper standard MPS | 700,322 | 920,356,883 | 1,620,677,883 |
| Leader standard MPS | 702,555 | 826,495,346 | 1,529,049,346 |
| Helper hands | 21,015 | 920,323,555 | 1,620,678,146 |
| Leader hands | 21,081 | 826,495,346 | 1,529,049,586 |
| Helper Multi-SLAM | 721,702 | 920,356,883 | 1,620,641,622 |
| Leader Multi-SLAM | 724,987 | 826,495,346 | 1,529,013,190 |
| Helper calibration | 21,014 | 920,356,883 | 1,620,678,146 |
| Leader calibration | 21,081 | 826,495,346 | 1,529,049,586 |

All 2,849,566 trajectory transforms passed the explicit `1e-7` tolerance:
helper/leader standard counts 700,322/702,555 and Multi-SLAM counts
721,702/724,987, with zero invalid transforms, missing tracking timestamps,
or duplicate timestamps. All four trajectory quality-score ranges are
`[0.5,1.0]`. The largest observed quaternion-norm error is
`9.227062447436651e-10`; largest rotation orthonormality and determinant errors
are `6.099755367472426e-9` and `6.8981358403163995e-9` respectively.

All 63,042 helper and 63,243 leader camera-calibration transforms passed.
Both label sets are `camera-rgb`, `camera-slam-left`, `camera-slam-right`.
Every calibration row has UTC sentinel `-1`; trajectory UTC fields are present
but retained only as unverified auxiliary clocks.

| Participant / hand | Present | Missing (`-1`) | Coverage of hand CSV rows | Present with confidence 0 |
| --- | ---: | ---: | ---: | ---: |
| Helper left | 20,334 | 681 | 96.7595% | 3 |
| Helper right | 20,596 | 419 | 98.0062% | 17 |
| Leader left | 20,463 | 618 | 97.0685% | 17 |
| Leader right | 19,798 | 1,283 | 93.9140% | 26 |

Residuals are candidate pose time minus hand time, including rejected candidates.
At a maximum allowed absolute gap of 10 ms:

| Participant | Hand timestamps | Accepted | Gap rejected | Signed range (ms) | Median / 95th-percentile absolute (ms) | Maximum absolute (ms) |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| Helper | 21,015 | 21,013 | 2 | −36.524 to +33.328 | 0 / 0 | 36.524 |
| Leader | 21,081 | 21,080 | 1 | −36.396 to 0 | 0 / 0 | 36.396 |

After gap enforcement, 40,929 present helper hands and 40,259 present leader
hands were transformed. Missing and rejected observations were not filled with
invented positions.

| Participant | First shared-world device position (m) | Last shared-world device position (m) |
| --- | --- | --- |
| Helper | `(0.075845, −0.109612, 0.000583)` | `(−0.610469, −0.268030, −0.055343)` |
| Leader | `(−0.535009, −0.539805, −0.115942)` | `(−0.133578, −0.102705, −0.125671)` |

The first RGB camera composition for each participant has zero matching
residual. Its shared-world position is approximately
`(0.065782, −0.102188, 0.005437)` m for helper and
`(−0.522371, −0.542605, −0.111410)` m for leader. The final calibration timestamps
extend beyond their respective Multi-SLAM endpoints: residuals are −36.524 ms
and −36.396 ms. Both final camera outputs are explicitly `unknown`, with no
world pose, under the 10 ms threshold. These are first/last checks, not a claim
that every camera pose was composed or visually validated.

Scan matrix validation passed with the determinant and orthonormality residual
reported above. The 304,681-vertex Aria PLY and 10,000,000-vertex BLK PLY headers
were readable. Scan distance units and exact registration to this shared graph
remain unverified.

## Semantic evidence table

This follow-up reuses the completed adapter and its validation report. Evidence
classifications apply to the exact claim in each row, not neighboring claims.
`VERIFIED` does not mean that an entire asset has been semantically calibrated.

| Relationship | Classification | Evidence and permitted use |
| --- | --- | --- |
| Helper/leader Multi-SLAM device poses share graph `8652bfb3-…` | VERIFIED | All trajectory graph IDs agree; poses may occupy one spatial frame in meters |
| Hands and online extrinsics to their own device | VERIFIED | Named Aria fields, units, quaternion conventions, and same-device timestamps; explicit nearest-gap checks |
| Device timestamp to its paired reported UTC value | VERIFIED | Exact source-row correspondence, preserved without interpolation; participant-scoped UTC |
| Affine fit of device time to reported UTC | INFERRED | Descriptive fit of staircase UTC values; never installed as a verified clock mapping |
| Reported UTC as precise cross-device capture time | UNRESOLVED | Rough phone RTC origin and no record-specific TICSync/timecode mapping or accuracy bound |
| MP4 PTS/time base and first/last presentation times | VERIFIED | Exact decoded endpoint PTS and container metadata |
| Paired ego videos are frame-aligned | VERIFIED (published contract) | Official CoMind project page; supports a common video frame index, without supplying pose/hand device-time bindings |
| Complete embedded MP4 frame-to-DEVICE_TIME arrays | UNRESOLVED / unavailable | Official Project Aria extractor returns one timestamp per trimmed video, versus 21,109 frames; strict count gate rejects both |
| MP4 PTS to device, TICSync, or common physical time | UNRESOLVED | Only one description integer per video, no complete image correspondence or trimming provenance |
| 90° clockwise export rotation | INFERRED | Official Aria exporter performs it for Gen1, but its applicability to these CoMind exports is unproven |
| Native RGB calibration to exported pixels/frustum | UNRESOLVED | Missing full rotation/crop/resize/distortion chain; native extrinsics alone are insufficient |
| Native keyed annotation layout and six non-skipped records | VERIFIED | Actual release inspected; source keys, labels, times, and categories retained |
| Annotation time numerically follows frame/30 | VERIFIED | All 14 boundaries agree within `6.67e-14` seconds; internal consistency only |
| Annotation video, origin, interval inclusion, bbox schema, and left/right identity | UNRESOLVED | No binding to exact video/composite dimensions and participant ordering |
| Scan text matrix is rigid | VERIFIED | Determinant, orthonormality and final-row tests; no repair |
| Thirty sampled Aria PLY coordinates equal standard leader MPS points | VERIFIED | Exact XYZ equality in bounded samples, standard graph `aa198227-…` |
| Whole Aria PLY uses standard leader MPS meters/world | INFERRED | Sample correspondence supports this; not an exhaustive file/export contract |
| BLK PLY is already aligned to stored Aria PLY | INFERRED | As-stored sampled distances are smaller than reapplying the matrix |
| Constant standard-leader-to-Multi world fit has small pose residuals | VERIFIED (measured test) | 21,077 local-time matches; held-out P95 translation 0.139 mm and rotation 0.0244 degrees, with explicit maxima and time-bin checks |
| Fitted standard-leader-to-Multi world transform | INFERRED, numerically validated | Joint orientation/position SE(3) estimate, fixed scale 1; preserves uncertainty and does not attach assets automatically |
| Either PLY and nominal scan transform bind to shared Multi-SLAM graph | UNRESOLVED | Numerically validated world estimate is available, but full PLY unit/frame/export contracts remain unresolved |

The detailed video/annotation investigation, primary source links, raw labels,
and missing evidence are in
[`comind_semantic_evidence.md`](../design/comind_semantic_evidence.md).
Machine-readable reports are in `outputs/comind_semantics/` and
`outputs/comind_scan_semantics/` for this recording.
The subsequent official API probe, common frame-index contract, annotation
origin audit, and matched-pose world-alignment results are documented in
[`comind_frame_timestamp_probe.md`](../design/comind_frame_timestamp_probe.md).
That probe found one embedded timestamp per MP4 and stopped the synchronized
viewer upgrade without downloading VRS. UTC is not required as the authority
if complete paired-video DEVICE_TIME arrays become available.

## Reported clock statistics and spatial QC

The semantic validator made one sequential read of each Multi-SLAM trajectory
and compact hand stream. It rechecked helper trajectory CRC `529bb29f` during
that same read. No raw archive was repaired or decompressed again. The report
is `outputs/comind_semantics/43276420-701f-4731-b9ab-bebc7fd14994.json`.

| Measurement | Helper | Leader |
| --- | ---: | ---: |
| Reported UTC first (ns) | 1774714644554113156 | 1774714644837450464 |
| Reported UTC last (ns) | 1774715344808718746 | 1774715347325033648 |
| Repeated UTC steps / consecutive pose steps | 700690 / 721701 | 703908 / 724986 |
| Maximum device timestamp gap (ms) | 1.077 | 1.075 |
| Reported UTC minus device time minimum (ns) | 1774713724163721142 | 1774713818308388794 |
| Reported UTC minus device time maximum (ns) | 1774713724197324936 | 1774713818342201518 |
| Net offset change (ms) | −30.133410 | −30.260816 |
| Descriptive affine slope drift (ppm) | −0.003738 | +0.003208 |
| Affine paired-value residual RMS (ms) | 9.884997 | 9.884655 |
| Affine paired-value residual p95 / max (ms) | 16.177026 / 17.331100 | 16.177772 / 17.536921 |

The reported UTC extent overlaps for **699.971268282 seconds**. This is an
INFERRED coarse wall-clock extent, not verified simultaneous capture. Device
times are strictly increasing, while reported UTC is nondecreasing with held
values advancing at roughly 30 Hz. The offset changes include that staircase
quantization; they are not evidence of individual clock resets. Slope fits
describe the source values and do not measure physical clock accuracy. The
official [MPS overview](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/mps_summary)
describes phone-derived RTC as approximate and potentially adjusted. Neither
fits nor nominal nanosecond units establish the missing physical alignment.

`ReportedUtcPairs` provides exact/nearest original source-row lookup with
deterministic ties, explicit maximum sampling gaps, and residuals. Its UTC
domain remains participant-scoped, uncertainty remains unknown, and the viewer
rejects it as a replacement for `VerifiedClockMapping`.

| Per-participant motion metric | Helper | Leader |
| --- | ---: | ---: |
| Device speed p50 / p95 / max (m/s) | 0.0418 / 0.2797 / 1.7634 | 0.0522 / 0.3407 / 1.3981 |
| Largest consecutive displacement (m) | 0.001874 | 0.001468 |
| Net displacement (m) | 0.7066 | 0.5935 |
| Speed > 2 m/s / step > 0.05 m / gap > 0.1 s | 0 / 0 / 0 | 0 / 0 / 0 |

| Hand | Landmark-device distance p50 / p95 / max (m) | Maximum device-relative wrist speed (m/s) | Speed intervals > 5 m/s |
| --- | --- | ---: | ---: |
| Helper left | 0.5159 / 0.6539 / 0.8935 | 3.3238 | 0 |
| Helper right | 0.5255 / 0.7202 / 1.1420 | 5.1992 | 1 |
| Leader left | 0.4545 / 0.5920 / 0.8434 | 4.2888 | 0 |
| Leader right | 0.4919 / 0.7344 / 0.9440 | 3.5144 | 0 |

No landmark exceeds the explicit 2 m device-distance limit. Wrist means Aria
landmark index 5; velocity is relative to the moving device, not an absolute
world-hand velocity. Missing hands and gaps over 0.1 s break continuity. These
are configurable diagnostic thresholds, not proof of anatomical validity.
The helper-right event remains a QC failure. Shared-world hands retain the
existing 10 ms own-device pose gap limit and missing/zero-confidence distinction.
World z is not interpreted as height above a floor. Cross-person distance,
simultaneous region occupancy, and handover-local QC remain unavailable.

## Sampled scan correspondence

Bounded PLY sampling reads at most 16,384 vertices in fixed-size blocks by
default and retains original indices. The observed sample extents are
`[14.109141, 9.803877, 3.493063]` for Aria and
`[16.795571, 10.810853, 2.473042]` for BLK, in unverified source units.
Thirty exact XYZ matches against a bounded standard-leader point prefix
support a standard leader MPS origin, distinct from the shared Multi-SLAM
graph. This does not establish a whole-file graph/unit contract.

Sampled median nearest distance from stored BLK to Aria is approximately
`0.210` source units, versus `0.950` after applying the nominal matrix and
`0.693` after applying its inverse. This supports the INFERRED already-aligned
interpretation; it does not validate registration. A trial standard/Multi
point-ID correspondence also did not establish a rigid bridge, and point IDs
are only documented as unique within a map. No fitted bridge is installed.
Scene rendering remains disabled. Full scope and numerical evidence are in
`outputs/comind_scan_semantics/43276420-701f-4731-b9ab-bebc7fd14994.json`.

## Resolved and remaining V0 semantics

1. Recover/replace the incomplete helper archive outside raw storage as needed;
   successful trajectory-member recovery is not complete dataset integrity.
2. Complete frame-index-to-DEVICE_TIME arrays are now verified for both MP4s.
   The documented paired frame alignment is authoritative; UTC is not involved.
3. Clockwise native RGB export orientation, camera composition and nonlinear
   projection are verified; no unverified rolling-shutter correction is invented.
4. Bind scan coordinate units and registration direction to the exact shared
   graph, and establish whether each PLY is already transformed.
5. Source annotation frame bounds now bind directly to the paired sync index
   under the explicit task contract. Left/right-to-participant label semantics,
   bounding-box encoding and source endpoint inclusion remain unresolved.
6. Cross-person wrist distances, local coverage and projection-domain coverage
   are reported for all six usable handovers. Spatial proximity is a QC result,
   never the basis for the synchronization mapping; occlusion is not evaluated.

The guarded Rerun renderer now supports the real paired-index pipeline. Bowl
handover `018240` is selected with context frames `18120–18372`; all 253 frames
have verified image mappings and independently matched participant samples.
The scene remains disabled. Static trajectory diagnostic mode is also retained.

## Reproducing semantic validation and Rerun diagnostics

The following full audit was run once for this investigation. It reuses the
previously recovered helper member, verifies its CRC during the read, and
creates sampled trajectory/all-hand caches under `data/processed/`:

```bash
.venv/bin/python scripts/validate_comind_semantics.py \
  --recording-id 43276420-701f-4731-b9ab-bebc7fd14994
```

For subsequent integration checks, validate those derived caches without
rescanning original trajectories or decoding videos:

```bash
.venv/bin/python scripts/validate_comind_semantics.py \
  --recording-id 43276420-701f-4731-b9ab-bebc7fd14994 --verify-cache
```

The cache check verifies source timestamp representations, graph agreement,
sample dimensions, rotations, missing points, and same-device gap enforcement.
Full-resolution clock/geometry statistics and source CRCs remain historical
evidence from generation; the cache check does not refresh source integrity.

Save the supported real-data diagnostic without launching a GUI:

```bash
.venv/bin/python scripts/visualize_comind.py \
  --recording-id 43276420-701f-4731-b9ab-bebc7fd14994 --diagnostic
.venv/bin/rerun rrd verify \
  outputs/comind_visualization/43276420-701f-4731-b9ab-bebc7fd14994_spatial_diagnostic.rrd
```

This reads only derived trajectory positions and existing evidence reports.
The recording contains static `world/helper/trajectory` and
`world/leader/trajectory` points plus visible evidence/QC status. It has no
simultaneous poses or fabricated common timeline. Scan, synchronized hands,
ego video, camera frustums, active annotations, and boxes are disabled.
Omitting `--diagnostic` fails clearly with the missing mapping requirements;
there is no force flag.

The reusable `duet.visualization.rerun_episode` API can render synthetic
episodes with explicit `VerifiedClockMapping` objects and exact integer
nanosecond timeline values. It clears missing/gap-rejected entities, preserves
source timestamps and confidence, and uses lazy nearest-frame video decoding.
Each optional layer has its own evidence requirement. Camera frustums require
verified model-unprojected perimeter rays in the exported camera frame, not a
pinhole approximation to the native fisheye. Scene points require verified
shared-frame registration and meters. Tests write and verify small RRD files
without a GUI.

Final integration validation: **413 tests passed** (the existing 316-test
baseline is preserved). Ruff lint and formatting pass across `src/duet`,
`tests`, and the three validation/visualization entry points. The pre-existing
downloader/inspector lint issues were not changed. The real diagnostic was
saved successfully and `rerun rrd verify` reported one file verified without
error. Cache validation passed for 21,010 helper and 21,077 leader sampled
trajectory poses, plus all 21,015 and 21,081 respective hand rows. These checks
do not change any UNRESOLVED semantic classification above.

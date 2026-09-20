# CoMind Dataset Format

This document records the CoMind v1 structures used by Duet.

Only verified facts should be stated as facts. Anything not yet confirmed from downloaded files or official documentation must remain marked as unresolved.

## Dataset version

CoMind v1.

Dataset source:

`https://comind.ethz.ch/dataset`

Downloader:

`scripts/comind_download.py`

Selected Duet development recording:

`43276420-701f-4731-b9ab-bebc7fd14994`

## Local dataset root

The dataset is stored under:

`data/raw/comind/`

Raw dataset files must remain immutable.

Expected high-level organization:

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

Each recording contains handover segments with fields including:

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

## Requested recording modalities

Duet is downloading:

- `mp4`
- `mps`
- `multislam`
- `scan`

Raw VRS files are intentionally excluded from V0.

## Approximate selected-recording sizes

From the downloader manifest:

- MP4: 6.23 GiB
- MPS: 3.72 GiB
- Multi-SLAM: 4.48 GiB
- 3D scan: 264.47 MiB

## MP4 video

Status:

`PENDING FILE INSPECTION`

Need to determine:

- exact filenames
- number of video streams
- which streams correspond to participant A and participant B
- whether other streams are exocentric or auxiliary
- codec
- resolution
- frame rate
- duration
- embedded timestamps
- relationship between MP4 timestamps and annotation times

Do not assume MP4 timestamps equal annotation timestamps until verified.

## MPS

Status:

`PENDING FILE INSPECTION`

The downloader describes this component as:

`MPS (eye-gaze/hand/SLAM)`

Need to determine:

- hand-tracking files
- eye-gaze files
- camera trajectories
- calibration files
- timestamps
- coordinate frames
- units

Need to verify whether trajectories represent:

`T_world_camera`

or:

`T_camera_world`

Do not guess.

## Multi-SLAM

Status:

`PENDING FILE INSPECTION`

Need to determine:

- exact filenames
- file formats
- purpose of each file
- participant-to-participant alignment
- shared-world definition
- coordinate convention
- relationship to individual MPS trajectories
- units
- timestamps
- whether transforms are static or time-varying

No transform should be used until its direction is explicitly verified.

## 3D scan

Status:

`PENDING FILE INSPECTION`

The downloader describes scan assets as:

`3D scan (.b2g + .pcd)`

Need to determine:

- exact filenames
- which scan corresponds to the environment
- point-cloud coordinate system
- units
- relationship to the Multi-SLAM world frame
- whether an additional registration transform is required

## Camera calibration

Status:

`PENDING FILE INSPECTION`

Need to locate and document:

- camera model
- intrinsics
- distortion model
- image dimensions
- camera-to-device extrinsics
- device-to-world trajectory
- units
- axis convention

If multiple cameras exist on one device, preserve their explicit names.

## Hand representation

Status:

`PENDING FILE INSPECTION`

Need to determine:

- representation type
- number of joints or landmarks
- left/right hand identification
- confidence fields
- units
- coordinate frame
- timestamps
- relationship to the device trajectory

All usable hands should eventually be transformed into the shared Duet world frame.

## Timestamp systems

Status:

`PARTIALLY VERIFIED`

Handover annotations contain times in seconds.

Still need to identify timestamp systems used by:

- MP4
- MPS camera trajectory
- MPS hand tracking
- Multi-SLAM
- scan metadata

For every modality document:

- timestamp field
- timestamp unit
- time origin
- clock domain
- synchronization relationship

Do not align modalities by array index if explicit timestamps exist.

## Coordinate systems

Status:

`UNRESOLVED`

Need to determine relationships among:

- participant A device frame
- participant B device frame
- participant A camera frame
- participant B camera frame
- participant A MPS world
- participant B MPS world
- Multi-SLAM shared frame
- scene-scan frame

For every transform record:

    source frame:
    destination frame:
    matrix convention:
    units:
    timestamp dependence:
    verified from:

Do not infer transform direction solely from filenames.

## Canonical Duet conventions

Target internal conventions:

- distances in meters
- explicit timestamps
- homogeneous transforms as 4x4 matrices
- transform names use `T_destination_source`
- both participants represented in one shared world frame

Example:

`T_world_camera`

means a point in camera coordinates can be transformed into world coordinates.

## File tree

Status:

`PENDING RECORDING DOWNLOAD`

After the recording finishes downloading, inspect it with:

    find data/raw/comind/recordings/43276420-701f-4731-b9ab-bebc7fd14994 \
      -maxdepth 4 \
      -type f \
      -print | sort

Do not invent filenames before inspection.

## Unresolved questions

1. Which MP4 files correspond to the two egocentric participants?
2. What additional MP4 streams are included?
3. Which MPS files contain hand tracking?
4. Which MPS files contain camera trajectories?
5. Where are camera intrinsics and calibration stored?
6. What are the MPS coordinate conventions?
7. How does Multi-SLAM align the participants?
8. What is the exact shared-world frame?
9. How is the scene scan registered to that frame?
10. What timestamp domain is shared across modalities?
11. What does `left` / `right` mean in the handover annotations?
12. Which video stream is used for handover bounding boxes?
13. Are hand coordinates in world, device, or camera space?
14. Are all spatial quantities in meters?
15. Are hand samples missing or low confidence during handovers?

These questions must be answered from actual files or official documentation before the first milestone is considered complete.

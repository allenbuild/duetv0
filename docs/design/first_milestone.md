# Duet First Milestone

## Goal

Build a correct shared spatial and temporal representation of one CoMind human-human handover recording.

The first milestone is not to estimate everything from raw video.

The first milestone is to prove that CoMind's provided ground-truth and processed metadata can be loaded, synchronized, transformed, and visualized correctly.

## Selected recording

CoMind recording:

`43276420-701f-4731-b9ab-bebc7fd14994`

This recording was selected because it contains multiple usable handovers while remaining relatively small compared with other candidate recordings.

Relevant downloaded modalities:

- MP4 video
- MPS output
- Multi-SLAM output
- 3D scan
- global CoMind handover annotations

Do not use raw VRS files for the first milestone unless they become strictly necessary.

## Target pipeline

The first working pipeline should be:

1. Load both egocentric video streams.
2. Preserve original timestamps.
3. Load camera calibration and camera trajectories.
4. Load provided hand tracking or hand-pose data.
5. Load Multi-SLAM alignment between participants.
6. Determine the coordinate-frame conventions.
7. Transform both participants into one shared world coordinate system.
8. Load the available 3D scene representation.
9. Visualize the shared scene in Rerun.
10. Validate synchronization and geometry around a labeled handover.

## Shared-world representation

At any timestamp t, the system should aim to represent:

- timestamp
- camera A pose
- camera B pose
- participant A left hand
- participant A right hand
- participant B left hand
- participant B right hand
- relevant object or handover annotation
- scene geometry

All spatial quantities should ultimately be expressed in one canonical world frame.

Distances should use meters internally.

## Coordinate-frame rules

Do not assume coordinate conventions.

For every transform, determine and document:

- source frame
- destination frame
- whether the transform is camera-to-world or world-to-camera
- translation units
- rotation representation
- axis convention
- handedness
- timestamp basis

Use explicit names such as:

`T_world_camera`

rather than ambiguous names such as:

`camera_pose`

Whenever a transform convention is uncertain, surface the uncertainty instead of guessing.

## Temporal alignment

Preserve source timestamps whenever possible.

Do not assume frame number alone is the canonical time reference.

The pipeline should determine:

- timestamps used by each ego video
- timestamps used by MPS
- timestamps used by Multi-SLAM
- timestamps used by handover annotations
- how the two participants are synchronized

Provide a reusable function for finding the nearest sample from each modality for a requested timestamp.

Record synchronization residuals where possible.

## Handover annotations

The global annotation file is:

`data/raw/comind/annotations/dataset_handover_consolidated.json`

For each usable handover, the annotation includes information such as:

- start frame
- end frame
- start time
- end time
- initiator
- delivery direction
- initiation type
- object category
- skip flag

Use only annotations where:

`skip == false`

For the first visualization, choose one clear usable handover from the selected recording.

Include context before and after the labeled interval rather than visualizing only the exact annotated frames.

## Rerun visualization

The first successful Rerun visualization should show:

- the 3D scene or point cloud
- participant A camera trajectory
- participant B camera trajectory
- both camera frustums at the current timestamp
- participant A hands
- participant B hands
- synchronized ego video for participant A
- synchronized ego video for participant B
- current timestamp
- current handover annotation when applicable

Use clearly separated entity paths such as:

`world/participant_a/camera`

`world/participant_a/hands/left`

`world/participant_a/hands/right`

`world/participant_b/camera`

`world/participant_b/hands/left`

`world/participant_b/hands/right`

`world/scene`

## Geometry validation

Do not treat a visualization that merely runs without crashing as success.

Validate the geometry.

At minimum check:

- transform matrices have valid dimensions
- homogeneous transforms have a valid final row
- rotation matrices are approximately orthonormal
- rotation determinants are approximately +1
- translation magnitudes are physically plausible
- both participants occupy the same physical environment
- camera trajectories are continuous
- hands remain physically near their associated participant
- the two participants' hands approach each other during a labeled handover

Create automated tests for reusable geometry functions.

## Synchronization validation

At minimum report:

- video duration for participant A
- video duration for participant B
- timestamp range for each trajectory
- timestamp range for each hand stream
- overlap interval
- nearest-sample residuals during the selected handover

Flag large synchronization gaps rather than silently interpolating through them.

## Canonical internal schema

The implementation should move toward a reusable episode representation rather than exposing CoMind-specific structures everywhere.

A minimal conceptual frame could contain:

- timestamp_ns
- participant_a_camera
- participant_b_camera
- participant_a_left_hand
- participant_a_right_hand
- participant_b_left_hand
- participant_b_right_hand
- handover_annotation
- QC metadata

Dataset-specific parsing belongs in:

`src/duet/adapters/`

General transform mathematics belongs in:

`src/duet/geometry/`

Synchronization logic belongs in:

`src/duet/synchronization/`

Visualization code belongs in:

`src/duet/visualization/`

Quality-control code belongs in:

`src/duet/qc/`

## Required documentation

Before implementing assumptions about CoMind, document the actual downloaded format in:

`docs/datasets/comind_format.md`

That document should include:

- file tree
- purpose of each relevant file
- timestamp format
- camera calibration format
- hand-pose format
- Multi-SLAM format
- coordinate conventions
- units
- unresolved ambiguities

The format document should be based on actual files and official documentation, not guesses.

## First milestone acceptance criteria

The milestone is successful when all of the following are true:

1. One command loads the selected CoMind recording.
2. Both egocentric views are synchronized on one timeline.
3. Both participants' camera poses are represented in the same world frame.
4. Both participants' available hand poses are represented in that same world frame.
5. A labeled handover can be selected by timestamp.
6. Rerun displays both participants and the scene simultaneously.
7. Camera and hand geometry appears physically consistent.
8. Synchronization and transform QC checks pass or explicitly report known limitations.
9. Relevant unit tests pass.
10. The implementation does not require any learned perception model.

## Explicitly out of scope for V0

Do not add these yet:

- WiLoR
- HaMeR
- SAM
- Grounded SAM
- depth estimation
- VGGT
- COLMAP
- object-pose estimation
- task segmentation models
- VLA training
- robot-policy training

Those belong to later milestones.

## What comes after V0

Once the ground-truth pipeline is correct:

1. Replace provided hand pose with learned hand tracking and compare against ground truth.
2. Add 3D object localization.
3. Replace provided camera trajectories with estimated trajectories.
4. Add automatic interaction-stage segmentation.
5. Build single-view versus paired-view prediction benchmarks.

The ground-truth implementation remains the reference system used to evaluate all later learned components.

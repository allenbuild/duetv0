# Duet Research

Research infrastructure for processing multimodal human-human interaction data into structured representations for robot learning.

## Initial objective

Convert synchronized recordings of two collaborating humans into a shared spatial and temporal representation containing:

- synchronized video
- camera and head trajectories
- hand trajectories
- object state
- shared world coordinates
- task stages
- annotations
- quality-control metrics

## First milestone

Using one existing public dataset episode:

1. Load both egocentric streams.
2. Preserve dataset timestamps.
3. Load camera calibration and trajectories.
4. Load provided hand poses.
5. Transform both participants into one world coordinate system.
6. Visualize everything in Rerun.
7. Validate synchronization and geometry.

No learned perception models are required for the first milestone.

## Planned development sequence

1. Ground-truth dataset to shared-world visualization
2. Replace ground-truth hand pose with learned hand tracking
3. Add object segmentation and 3D tracking
4. Replace provided camera trajectories with estimated trajectories
5. Add automatic task-stage segmentation
6. Build single-view vs paired-view prediction benchmarks

## Initial dataset

We will begin with one CoMind interaction episode and use its provided metadata wherever possible before introducing learned perception models.

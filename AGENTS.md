# Duet Research

This repository develops a processing and evaluation pipeline for multimodal human-human interaction data for robot learning.

## Core engineering rules

- Never modify original files under `data/raw/`.
- Never commit raw datasets, videos, credentials, or model weights.
- Treat dataset coordinate conventions as unknown until verified from documentation or tests.
- Never silently guess whether a transform is camera-to-world or world-to-camera.
- Use meters as the canonical internal distance unit.
- Preserve original timestamps whenever available.
- Prefer dataset-provided ground truth over inferred values during initial pipeline development.
- Write unit tests for coordinate transformations and synchronization logic.
- Keep dataset-specific parsing inside `src/duet/adapters/`.
- Keep general geometry code independent of any specific dataset.
- Add assertions for dimensions, units, timestamps, and transform validity.
- Surface ambiguities rather than hiding them.
- Do not add large ML models or dependencies unless required for the current task.

## Development philosophy

Build and validate the pipeline incrementally.

First use provided ground truth to establish a correct shared-world representation.

Only then replace individual ground-truth components with learned perception models.

Do not try to solve synchronization, SLAM, hand pose, object pose, depth, and task segmentation simultaneously.

## First milestone

Given one paired-egocentric human-human interaction, visualize both cameras and both participants' hands correctly in one synchronized shared 3D coordinate system.

For the first implementation, use dataset-provided:

- timestamps
- synchronization
- camera intrinsics
- camera extrinsics
- camera trajectories
- hand annotations
- scene geometry

whenever available.

## Repository structure

- `data/raw/`: immutable original dataset files
- `data/processed/`: generated intermediate representations
- `src/duet/adapters/`: dataset-specific loaders
- `src/duet/synchronization/`: timestamp alignment
- `src/duet/geometry/`: transforms, projection, coordinate systems
- `src/duet/tracking/`: hand and object tracking
- `src/duet/visualization/`: Rerun and debugging visualization
- `src/duet/qc/`: automated quality control
- `src/duet/schemas/`: canonical Duet episode representation
- `scripts/`: runnable entry points
- `tests/`: automated tests
- `docs/datasets/`: dataset format documentation
- `docs/design/`: architecture and design notes
- `outputs/`: generated visualizations, metrics, and reports

## Coding expectations

- Use Python type hints for public functions.
- Write small reusable functions rather than giant scripts.
- Include docstrings whenever coordinate frames or units matter.
- Run relevant tests after modifying core functionality.
- Prefer reproducible CLI commands over notebook-only workflows.
- Document assumptions explicitly.
- Do not hard-code one dataset episode when a reusable loader is practical.

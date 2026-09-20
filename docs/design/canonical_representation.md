# Duet V0 canonical representation

This foundation is independent of CoMind recording files. It implements explicit
spatial and temporal contracts, a limited verified handover adapter, and synthetic
tests. It does not establish that real CoMind streams are synchronized, aligned,
or ready for visualization. No raw files are required by the tests.

## Timestamps and clock domains

`schemas/time.py` defines `Timestamp(raw_value, unit, clock_domain, provenance)`.
The raw value is preserved as an integer, float, or `Decimal`; `None` means missing.
Booleans, NaN, infinity, and numeric strings are rejected. Negative timestamps are
allowed because a verified source clock may precede its chosen origin.

`TimeUnit` explicitly identifies seconds, milliseconds, microseconds, or
nanoseconds. `Timestamp.seconds` converts units using exact `Fraction` arithmetic.
Integer nanoseconds remain distinguishable at large epochs. Floats retain their
original Python value and are interpreted through their decimal string for
arithmetic; precision already lost upstream cannot be recovered. The JSON
annotation loader uses `Decimal` for decimal numeric tokens.

A `ClockDomain` identifies both a clock and its origin. Names must distinguish
recordings, devices, and modalities unless their clock relationship has been
verified. Equal numeric values or units do not establish a shared clock. Naming
two clocks identically is an explicit caller assertion that they are the same
domain. No ordering operators are defined for `Timestamp`.

Cross-clock operations require one direct `VerifiedClockMapping` with explicit
source and destination clocks, positive scale, an offset in seconds, provenance,
and nonempty verification evidence:

```text
destination_seconds = scale * source_seconds + offset_seconds
```

The caller must verify that this relation holds over the processed data. Merely
constructing this object does not measure synchronization. There is no automatic
fitting, inverse mapping, chaining, index alignment, or inferred frame-rate
conversion. Ambiguous duplicate mappings are rejected. Original timestamps are
kept even when a normalized matching value is calculated.

CoMind annotation clocks are isolated as
`comind:<recording_uuid>:handover_annotations`. Their values are verified seconds;
their origin and relation to video, MPS, and Multi-SLAM clocks are unresolved.

## Identities and frame names

`ParticipantId` is an explicit episode-local identity. `FrameId` identifies a
coordinate system. Names such as `synthetic:world` and
`synthetic:alice:camera` avoid collisions and explain intended ownership. A name
alone does not establish axes, handedness, extrinsics, or physical identity.
Adapters must verify those conventions before creating transforms.

`FrameIdentifier(stream_id, index, source_id)` identifies an original image or
sample frame. At least an index or opaque source ID is required. Source indices
are preserved; they are neither spatial frame IDs nor substitutes for timestamps.

CoMind source labels `left`, `right`, `ltr`, and `rtl` remain uninterpreted labels.
They do not become canonical participants, cameras, hand sides, or graph edges.

## Geometry and units

`RigidTransform` uses column-vector `T_destination_source` semantics:

```text
p_destination = R @ p_source + t
T_world_hand = T_world_camera @ T_camera_hand
```

The API expresses the second line as
`T_world_camera.compose(T_camera_hand)`. Composition requires the inner
destination to equal the outer source; incompatible frames raise an error.
`inverse()` exchanges source and destination. `apply(points, unit=...)` accepts
points of shape `(..., 3)` in the named source frame and returns destination-frame
points. Callers must explicitly declare that point coordinates are in meters.

Transforms validate dimensions `(4, 4)`, real finite entries, orthonormality of
the rotation, determinant approximately `+1`, and homogeneous final row
`[0, 0, 0, 1]`. A transform within the same named frame must be identity. Validation
uses an explicit absolute tolerance, default `1e-8`, with zero relative tolerance.
Invalid matrices are rejected without normalization, projection onto SO(3),
reflection correction, or other repair. Matrices are copied into immutable
storage to prevent later mutation invalidating those checks.

The direct constructor accepts only `DistanceUnit.METERS`. The explicit
`RigidTransform.from_matrix(..., unit=...)` factory converts translation from
verified meters, centimeters, or millimeters into meters and records the
conversion provenance. Units must never be inferred from numeric magnitude.
Canonical hand points also require meters; no adapter-specific conversion is
hidden in the sample schema.

`FrameGraph` stores verified static transforms or one explicitly selected temporal
snapshot. It supports inverse and composed paths. It rejects every cycle and
redundant edge, even numerically consistent ones, so each pair has at most one
path. Unknown or disconnected frames fail explicitly. It has no interpolation,
time-varying edge lookup, calibration selection, or trajectory synchronization.
Build separate graphs for different snapshots when transforms vary with time.

## Samples, episodes, and provenance

`CameraSample` associates a participant, camera coordinate frame, original
timestamp, optional source frame identifier, and `T_destination_camera`. The
transform source must match the declared camera frame.

`HandSample` contains a participant, caller-defined hand ID, timestamp, coordinate
frame, explicit meters, and finite `(N, 3)` landmarks. Optional unique landmark
names must match the point count. No skeleton topology, landmark order, or
left/right interpretation is invented. The point array is copied and immutable.

`Episode` declares participant IDs and a shared coordinate frame. Available
camera transforms must target that frame, and hand samples must declare that
frame. Undeclared participants are rejected. The episode preserves separate
camera and hand sequences and their source timestamps. Episode membership
establishes a spatial contract, not a shared clock or same-index correspondence.

`Provenance` records a nonempty source reference, optional explanatory detail, and
parent provenance records. Derived transforms retain their parents; conversions
name the operation and source units. Timestamps, observations, confidence,
annotations, and QC can retain provenance as well. A source reference is evidence
to trace, not automatic proof that its semantics have been verified.

## Missing data and confidence

`SampleMetadata` distinguishes `PRESENT`, `MISSING`, and `UNKNOWN` payload states.
Missing/unknown camera or hand payloads use `None` and require an explanatory
reason. Present spatial payloads cannot be marked absent, and absent payloads
cannot be marked present. Missing timestamps remain independently expressible as
`Timestamp(raw_value=None, ...)`. No zero pose, zero timestamp, or invented hand
substitutes for unavailable data.

Confidence is optional. `None` means unknown; zero is a supplied zero score.
`Confidence(value, provenance)` requires a finite value in `[0, 1]`, without
claiming a calibrated probability. Source-specific scores must only be normalized
after verifying their scale and documenting that conversion. No confidence is
inferred from sample presence or from a handover's `skip` flag.

A `HandoverAnnotation` represents an existing record, so its record state is
present even when individual boundaries or labels are missing. Missing boundaries
use `None`. No annotation object is created for an absent record. Initiation
types, categories, and documented opaque JSON fields retain list/scalar structure
and are copied away from source parsing buffers; their nested JSON remains mutable
for compatibility and should be treated as source data by consumers.

## Synchronization and QC

`order_timestamps(..., clock_domain=..., mappings=...)` creates a reusable ordered
timeline with original source indices. Present timestamps sort by exact time,
then original index; duplicates are retained. Missing values retain their original
indices separately. A missing-valued timestamp from another clock still requires
a verified mapping. Bare `None` entries have no domain and only count as missing.

`nearest_timestamp` and `nearest_in_timeline` require `max_gap_seconds`. Equal
distances choose the earlier time, and duplicate candidates choose the first
original index. A residual exactly equal to the limit is accepted. The signed
residual is **candidate minus query** in the target clock's seconds.

Results distinguish `MATCHED`, `GAP_EXCEEDED`, `NO_SAMPLES`, and `MISSING_QUERY`.
A rejected nearest candidate keeps its index, raw timestamp, and residual for QC;
consumers must check `accepted` or use `matched_index`, which is `None` for rejected
matches. No samples are interpolated through missing intervals.

`timestamp_overlap` intersects inclusive stream extents. A common single instant
has zero duration. Empty, entirely missing, and disjoint streams have no overlap;
the result retains per-stream ranges and missing/duplicate counts. Extent overlap
does not prove continuous sample coverage within the interval.

`QCResult` carries a check name, `PASS`, `FAIL`, or `INSUFFICIENT_DATA`, structured
metrics, explicit applied thresholds, message, and provenance. Samples and
episodes can retain QC records. The foundation provides:

| Check | Explicit threshold | Evidence |
| --- | --- | --- |
| Transform validity | Absolute matrix tolerance | Validation failures and valid rotation metrics |
| Trajectory continuity | Translation speed in m/s; angular speed in rad/s | Consecutive displacement/rotation rates and invalid intervals |
| Timestamp overlap | Minimum overlap seconds | Stream ranges, intersection, missing and duplicate counts |
| Nearest residuals | Maximum absolute residual seconds | Signed residuals, missing matches, rejected candidates |
| Synchronization gaps | Maximum consecutive gap seconds | Ordered gaps and excessive source-index pairs |

Trajectory QC retains source order so duplicate/decreasing timestamps are visible.
It does not bridge missing poses or times. It requires matching frame endpoints
and a common clock for each evaluated pair, reporting incompatible pairs as
failures. Angular speed uses the shortest relative rotation and is not a model
of acceleration or motion between observations. Synchronization configuration
errors and incompatible clocks raise; missing evidence yields insufficient data
where the check cannot be completed. Thresholds express caller policy, not
unverified CoMind physical limits.

## Verified CoMind annotation adapter

`adapters/comind/annotations.py` provides:

- `parse_handover_segments(segments, recording_uuid=..., provenance=...)` for an
  explicitly selected sequence of segment dictionaries.
- `load_handover_annotations(path, recording_uuid=..., segment_selector=...)` for
  the global `dataset_handover_consolidated.json` file at a caller-supplied path.

The file loader uses the verified root `data` dictionary keyed by exact
recording UUID. Subsequent real-data inspection established the native
per-recording object keyed by opaque segment IDs. This is now the default;
an explicit selector remains available for another independently verified
layout. Source keys and insertion order are preserved. No default raw path is
accessed, and importing the adapter performs no I/O.

Only entries whose `skip` value is literally `False` are retained; null, missing,
zero, strings, and `True` are excluded. The adapter preserves original frame
indices, second-based time values, initiator/delivery labels, initiation types,
and all three object-category levels. It preserves the documented `bbox`,
`transcript_10s`, `numeric_id`, `description`, `annot_type`, and `skip` fields as
opaque source values. It does not interpret boxes, turn `numeric_id` into a
participant or annotation identity, or parse undocumented fields. Malformed
values and reversed intervals raise rather than being silently repaired.

## What remains unresolved for CoMind

The subsequent real-data adapter now verifies the helper/leader recording layout,
MPS `T_world_device` trajectories in meters, device-frame hand landmarks, the
`-1` missing-hand confidence sentinel, online calibration encoding and RGB label,
and Multi-SLAM graph-based shared-space semantics. See
[CoMind format evidence](../datasets/comind_format.md) for actual file observations,
official sources, frame/clock tables, and the archive-integrity limitation.
The reusable canonical contracts above remain dataset-independent.

The full first milestone still requires:

- Verified mappings between helper and leader device clocks, trimmed MP4 PTS,
  and annotation seconds. Equal world graph IDs establish space, not time.
- Validation of exported MP4 pixels against native RGB calibration, including
  rotation, distortion/image processing, and capture-time offsets.
- Scan units and registration direction bound to the exact shared graph;
  matrix validity and filename wording are insufficient evidence.
- Annotation `left`/`right` and delivery labels mapped to physical participants,
  bounding-box conventions, and interval inclusion semantics.
- Complete archive integrity: helper Multi-SLAM ZIP member recovery can provide
  a CRC-checked trajectory while the containing raw archive remains truncated.
- Shared-time spatial plausibility and handover-local coverage checks before a
  synchronized Rerun visualization can be considered correct.

No learned perception is introduced. The Rerun integration accepts explicit
verified bindings for synthetic/common-clock episodes. Current CoMind evidence
enables only static trajectory diagnostics, without claiming that independently
timestamped participant samples are simultaneous. Exact paired reported UTC is
preserved in participant-scoped clocks; an empirical fit never upgrades it to
a verified common clock. See the current evidence table and reproduction
commands in [CoMind format evidence](../datasets/comind_format.md).

## Reproducible validation

Use the existing environment; no additional dependencies are needed:

```bash
.venv/bin/pytest
.venv/bin/ruff check src/duet tests
.venv/bin/ruff format --check src/duet tests
```

`tests/test_synthetic_pipeline.py` is the executable two-participant example:

```bash
.venv/bin/pytest tests/test_synthetic_pipeline.py -q
```

It defines fake clocks, mismatched stream ordering/cadence, two camera transforms,
four hand samples, and a shared synthetic world. It selects samples by timestamp,
checks residuals, composes frame paths, and verifies that selected hand landmarks
coincide in the shared frame. It uses no CoMind recording files.

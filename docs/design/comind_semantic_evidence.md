# CoMind video and annotation evidence

This review concerns recording `43276420-701f-4731-b9ab-bebc7fd14994`.
The machine-readable record is
`outputs/comind_semantics/43276420-701f-4731-b9ab-bebc7fd14994_video_annotations.json`.
Inspection read the recording manifest, its handover annotations, MP4 headers,
and first/last frames. It did not decode complete videos or rescan trajectories.
All source files remain unchanged. Frame probes are in the adjacent UUID directory.

## Evidence and enabled layers

| Relationship | Finding | Consequence |
| --- | --- | --- |
| Multi-SLAM shared space | Established separately by graph UID verification | Individual samples and static paths can occupy a shared frame |
| Native RGB camera pose | `T_world_device @ T_device_camera-rgb`, using each participant's device time | Physical native camera axes are meaningful when the local pose match passes its gap limit |
| Trimmed MP4 PTS to device time | Unresolved | No synchronized video overlay |
| Device time to common physical time | Missing record-specific TICSync correspondence; MPS UTC is insufficient | No claim of simultaneous participant poses |
| Native RGB to exported image pixels | Unresolved export rotation/crop/resize/distortion chain | No image-aligned frustum or projection |
| Annotation times/frames to a video | Internal 30 Hz numerical relationship only | No active handover on the shared timeline |
| Annotation labels to helper/leader | Unresolved composite ordering | Preserve `left`, `right`, `ltr`, `rtl` without assigning participants |
| Raw annotation bounding boxes | Unresolved exact video, size, order, and scaling | No box overlay |

`MappingEvidence` in `duet.adapters.comind.semantics` records `verified`,
`inferred`, or `unresolved` claims with scoped statements and source references.
Only `verified` passes its capability check. An evidence record is not a
numerical clock mapping or a camera transform.

## Video timestamps and pixels

Both inspected trimmed ego MP4s contain 21,109 frames at nominal 30 Hz. Their
sampled first/last PTS are `0` and `10807296`, with time base `1/15360` seconds.
Thus the last sampled presentation time is exactly `703.6` seconds in each
file's own clock. This equality does not establish corresponding physical
instants. Their raw container `description` values are:

| File | Raw description |
| --- | --- |
| `helper_trimmed_sync.mp4` | `919623571487` |
| `leader_trimmed_sync.mp4` | `825528737150` |

The inspected [official Project Aria exporter, pinned source revision](https://github.com/facebookresearch/projectaria_tools/blob/99e68c8aeb26f270933c88cd6f77f8fc1d137c13/projectaria_tools/tools/vrs_to_mp4/vrs_to_mp4_utils.py)
stores an array of source device capture timestamps in `description` and
writes `mp4_to_vrs_time_ns.csv` with explicit MP4/device pairs. It can duplicate
or skip source images. One integer for a 21,109-frame CoMind video cannot be
treated as that full correspondence. No code here assumes it is a start offset.
The inspected exporter also rotates Gen1 images 90 degrees clockwise and may
resize them. Its applicability to these particular CoMind exports is unverified.

Neither inspected stream/container metadata nor sampled frame side data supplied
a display rotation matrix. The first decoded images look upright and exhibit
wide-angle distortion. These are observations, not evidence for an exact
native-to-export pixel transform or distortion model. The physical native RGB
calibration alone cannot justify attaching its rays to these exported pixels.

The [CoMind paper, processing section](https://arxiv.org/html/2607.06691#S3.SS3)
describes TICSync synchronization and common trimming, but supplies no numerical
mapping for this recording. [Aria's MPS overview](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/mps_summary)
explains the approximate phone RTC origin of MPS UTC. Record-specific
TICSync/time-domain metadata must establish a common timeline before the viewer
may claim synchronization.

The separate one-pass clock audit,
`outputs/comind_semantics/43276420-701f-4731-b9ab-bebc7fd14994.json`, also found
that reported UTC repeats across the roughly 1 kHz Multi-SLAM samples and
advances at approximately 30 Hz: 700,690 of 721,701 helper steps and 703,908 of
724,986 leader steps repeat the previous UTC value. An empirical affine fit
has about 9.885 ms RMS residual against those reported pairs. This quantized
relationship does not establish cross-device physical-time accuracy.

## Annotation observations

The inspected `dataset_handover_consolidated.json` release is version `1.0`,
with raw file timestamp `2026-06-26 15:47:24+00:00`. Its `data[UUID]` value is
an object keyed by source segment keys. `load_handover_annotations` now accepts
this native layout by default. It retains those keys as opaque annotation IDs,
preserves original ordering and numeric precision, and continues to accept an
explicit selector for alternative verified layouts. It does not derive frame
numbers from keys.

There are seven source entries and six with literal `skip == false`. All 14
source time/frame boundaries satisfy `time ≈ frame / 30`, with maximum absolute
difference `6.666666666666667e-14` seconds. This is an internal consistency
measurement, not a verified annotation-to-video mapping. It does not settle
the video asset, frame-number origin, boundary inclusivity, or trim origin.

| Retained source key | Original frames | Original seconds | Initiator | Delivery flow | Object |
| --- | --- | --- | --- | --- | --- |
| `002131` | 2131–2366 | 71.03333333333333–78.86666666666666 | right | rtl | carrot peel |
| `015185` | 15185–15199 | 506.1666666666667–506.6333333333333 | right | rtl | bowl |
| `015204` | 15204–15212 | 506.8–507.06666666666666 | right | rtl | spoon |
| `018260` | 18260–18286 | 608.6666666666666–609.5333333333333 | left | ltr | spoon |
| `018240` | 18240–18252 | 608.0–608.4 | left | ltr | bowl |
| `018941` | 18941–18994 | 631.3666666666667–633.1333333333333 | right | rtl | bowl |

The [paper's task definition](https://arxiv.org/html/2607.06691#S3.SS1)
places a handover box in the sender view of a horizontal composite. Its
[model prompts](https://arxiv.org/html/2607.06691#S7) use normalized output boxes;
that does not establish the raw annotation JSON box schema. The inspected
raw boxes include decimal values greater than 1000. No box ordering, image
scale, source stream, or participant-half association is inferred from them.
No handover has been selected for synchronized playback while these mappings
remain unresolved.

## Exact missing evidence

The recording manifest names two VRS assets that are absent locally:

- `recordings/43276420-701f-4731-b9ab-bebc7fd14994/trimmed_vrs/helper_trimmed.vrs`
- `recordings/43276420-701f-4731-b9ab-bebc7fd14994/trimmed_vrs/leader_trimmed.vrs`

Their time-sync records and native RGB metadata may resolve device-to-TICSync
and stream identity. They alone do not prove the CoMind MP4 export/trim chain.
The additional required evidence is:

- Per-export `mp4_to_vrs_time_ns.csv` or equivalent exact image correspondence,
  with `vrs_to_mp4_log.json` and subsequent CoMind trimming/export provenance.
- Exporter command/version and full native-to-MP4 pixel transform chain,
  including source stream, rotation, crop, resize, and any undistortion.
- Annotation export/UI schema binding times, frame numbers, raw boxes,
  composite ordering, and image dimensions to exact video assets.

Those exporter and annotation artifacts are not listed in the inspected
manifest. They are requirements for verification, not claims that additional
published files with those names exist. No large download or dependency
installation was performed for this review.

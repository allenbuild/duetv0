# CoMind synchronization forensics and download decision

Recording: `43276420-701f-4731-b9ab-bebc7fd14994`.

**Recommendation: B — request the MP4 frame/device-time mapping from the CoMind authors first.** The missing small transcript component has now been checked and does not supply that mapping. The published `other` component is empty. Bounded remote VRS inspection recovered native RGB counts and endpoint timestamps, but the MP4 exports have different counts and sampled frame-index irregularities. Downloading both VRS files would supply native images and complete capture timestamps; it would not automatically establish their correspondence to the MP4s.

No full VRS download was started, no existing raw file was modified, and no viewer functionality was added. Six previously missing official transcript files were added under raw after hash verification. The remote VRS probe read 18,273,269 response-body bytes, below the 20,000,000-byte limit. This report distinguishes observed image matches, native timestamp facts, and mappings that remain unverified.

## 1. Missing components and small-metadata check

Fresh official manifests list 52 recording payloads, three shared annotations and nine shared meshes. The recording manifest equals the existing local copy. Exact URLs, local destinations, published SHA-256 hashes and sizes for all 64 payloads are retained in the [inventory JSON](../../outputs/comind_sync_forensics/43276420-701f-4731-b9ab-bebc7fd14994_inventory.json) and [CSV](../../outputs/comind_sync_forensics/43276420-701f-4731-b9ab-bebc7fd14994_inventory.csv). Large existing assets were not rehashed or copied.

| Part | Published files | Published total bytes | Missing before this pass | Missing now |
| --- | ---: | ---: | ---: | ---: |
| `vrs` | 2 | 24,473,487,418 | 2 | 2 |
| `mp4` | 7 | 6,690,575,825 | 0 | 0 |
| `gopro` | 0 | 0 | 0 | 0 |
| `mps` | 22 | 3,994,480,083 | 0 | 0 |
| `multislam` | 12 | 4,814,822,301 | 0 | 0 |
| `transcripts` | 6 | 363,702 | 6 | 0 |
| `scan` | 3 | 277,312,919 | 0 | 0 |
| `other` | **0** | **0** | **0** | **0** |
| shared `annotations` | 3 | 5,275,342 | 0 | 0 |
| shared `meshes` | 9 | 127,550,688 | 9 | 9 |

The separate `gopro` part is empty because the official classifier includes the `mp4s/gopro_*` files in `mp4`. No additional non-trimmed VRS entry is hidden by the downloader's trimmed-file filter in this manifest. All remaining missing published payloads total **24,601,038,106 bytes**: the two VRS files and nine object meshes.

The only missing recording component small enough to inspect was `transcripts`. The official downloader fetched just that component for this UUID into fresh staging under `outputs/comind_sync_forensics/`. Each file passed its published size and SHA-256 check before exclusive creation of its missing raw destination. All 45 pre-existing recording files retained their sizes, modification times and inodes; existing raw healthcheck/split metadata was not replaced. The [download receipt](../../outputs/comind_sync_forensics/43276420-701f-4731-b9ab-bebc7fd14994_transcripts_receipt.json) records the operation.

The following names are relative to both the recording URL `https://comind.ethz.ch/dataset/43276420-701f-4731-b9ab-bebc7fd14994/` and the local recording root `data/raw/comind/recordings/43276420-701f-4731-b9ab-bebc7fd14994/`:

| Newly downloaded file | Bytes |
| --- | ---: |
| `transcripts/helper_trimmed_sync_transcript.ass` | 9,221 |
| `transcripts/helper_trimmed_sync_transcript.json` | 180,364 |
| `transcripts/helper_trimmed_sync_transcripts_squared.ass` | 9,221 |
| `transcripts/leader_trimmed_sync_transcript.ass` | 6,340 |
| `transcripts/leader_trimmed_sync_transcript.json` | 152,216 |
| `transcripts/leader_trimmed_sync_transcripts_squared.ass` | 6,340 |
| **Total** | **363,702** |

Both JSON files contain `segments` and `word_segments`; recursive fields are `start`, `end`, `text`, `words`, `word`, `score`, and `speaker`. These describe speech/word intervals. ASS files contain subtitle intervals, styles and layout. **Neither format contains per-frame device timestamps, VRS indices, trim/export maps, clock-domain fields or synchronization bindings.** Speaker labels remain uninterpreted. Filename words such as `trimmed_sync` do not prove a timing convention.

No remaining small published component is identified as a synchronization/export sidecar. The remaining shared `meshes` manifest contains only these nine PLY objects; it was not downloaded. URLs are explicit below; local destinations are `data/raw/comind/meshes/<filename>`.

| Missing shared mesh | Bytes |
| --- | ---: |
| [board.ply](https://comind.ethz.ch/dataset/meshes/board.ply) | 26,182,981 |
| [bowl.ply](https://comind.ethz.ch/dataset/meshes/bowl.ply) | 7,668,751 |
| [knife.ply](https://comind.ethz.ch/dataset/meshes/knife.ply) | 26,065,497 |
| [ladle_1.ply](https://comind.ethz.ch/dataset/meshes/ladle_1.ply) | 986,697 |
| [ladle_2.ply](https://comind.ethz.ch/dataset/meshes/ladle_2.ply) | 1,136,374 |
| [pan_1.ply](https://comind.ethz.ch/dataset/meshes/pan_1.ply) | 11,085,655 |
| [pan_2.ply](https://comind.ethz.ch/dataset/meshes/pan_2.ply) | 17,727,288 |
| [pot_1.ply](https://comind.ethz.ch/dataset/meshes/pot_1.ply) | 11,360,391 |
| [pot_2.ply](https://comind.ethz.ch/dataset/meshes/pot_2.ply) | 25,337,054 |
| **Total** | **127,550,688** |

Auxiliary files outside the payload manifests are selection lists or manifests: `candidate_ids.txt` (111 bytes), `selected_recording.txt` (37), `annotations/healthcheck.json` (725), the recording's `healthcheck.json` (9,490), `split/test.txt` (925), and `split/train.txt` (2,035). No additional per-recording timing payload was exposed by the official downloader or these manifests.

The fresh [official downloader](https://comind.ethz.ch/scripts/comind_download.py), version `20260626-01`, is byte-identical to the local script: 49,867 bytes, SHA-256 `ef9fa39c5f0b39c47b8790278df211e6e354b1cf8026bdbd5e1f442a06f1ea49`. Requested source searches found zero occurrences of timestamp(s), sync/synchron, frame, metadata, export, device_time, timecode, aria, mapping or sidecar. Nine lines match trim/trimmed, covering filters and labels; one matches offset, for HTTP resume. Exact line evidence is retained in the inventory JSON.

The previously audited helper `multislam_output/0/slam.zip` and `multislam_output/summary.json.zip` are present but structurally invalid locally. They are not missing assets. Published-size agreement does not establish hash integrity or prove that the remote objects are defective. This pass did not reread or repeat recovery of the 1.24 GB helper archive.

## 2. Exact VRS files and Multi-SLAM correspondence

**VERIFIED:** the two downloadable object keys exactly match the keys in the existing `vrs_to_multi_slam.json`.

| Missing VRS file, relative to recording | Bytes | Multi-SLAM output |
| --- | ---: | --- |
| [trimmed_vrs/helper_trimmed.vrs](https://comind.ethz.ch/dataset/43276420-701f-4731-b9ab-bebc7fd14994/trimmed_vrs/helper_trimmed.vrs) | 12,572,405,231 | `0` |
| [trimmed_vrs/leader_trimmed.vrs](https://comind.ethz.ch/dataset/43276420-701f-4731-b9ab-bebc7fd14994/trimmed_vrs/leader_trimmed.vrs) | 11,901,082,187 | `1` |
| **Total** | **24,473,487,418** | **22.79271131 GiB** |

Local destinations would be under `data/raw/comind/recordings/43276420-701f-4731-b9ab-bebc7fd14994/trimmed_vrs/`. This proves file identity in the manifest/Multi-SLAM mapping; it does not make helper and leader device-clock values directly comparable.

## 3. Trimmed versus untrimmed MP4 image correspondence

Both trimmed MP4s have 21,109 frames; both untrimmed MP4s have 23,295. Indices below are zero-based MP4 presentation-grid indices. Each MP4 has start PTS 0 and step 512 in time base 1/15360. All decoded consecutive PTS increments were checked, the initial 2,220 untrimmed decoder ordinals were checked against that grid, and decoding to EOF confirmed last indices 21,108 and 23,294. The grid in uninspected intervals is supported by headers, not an exhaustive decode. PTS was used only to locate frames within each file; **cross-video correspondence was selected from images**, not timestamp offsets.

The probe compared 64×64 grayscale fingerprints on a 0–255 intensity scale, accepting a candidate only with RMSE ≤2.0 and a best-versus-second margin ≥0.1. It searched 2,217 initial offset candidates using consecutive frames, then checked **26 well-spaced anchors and consecutive neighborhoods: 88 trimmed samples per participant**, including the first and last eight. The largest retained fingerprint window was 9,089,024 bytes; full videos were never loaded into memory. Detailed candidates, margins and sample indices are in the [video report](../../outputs/comind_sync_forensics/43276420-701f-4731-b9ab-bebc7fd14994_video_trim.json).

| Observation | Helper | Leader |
| --- | --- | --- |
| Trimmed frame 0 best visual match | Untrimmed **2186** | Untrimmed **2187** |
| Frame 0 RMSE; second/best ratio | 0.493; 3.76 | 0.470; 4.51 |
| Confident sampled offset counts | 59 at **2186**, 21 at **2187** | 64 at **2186**, 23 at **2187** |
| Ambiguous/high-error samples | 8 | 1 |
| Confident match RMSE median / P95 / max | 0.463 / 0.506 / 0.522 | 0.473 / 0.520 / 0.545 |
| Ending untrimmed index | **Not uniquely determined** | **No accepted local match for final trimmed frame 21108** |
| Exact global `untrimmed = trimmed + K` | **Rejected by sampled visual correspondences** | **Rejected by sampled visual correspondences** |

Examples show the changing correspondence directly:

| Participant | Trimmed indices | Best untrimmed indices | Offset |
| --- | --- | --- | ---: |
| helper | 0, 1, 2 | 2186, 2187, 2188 | 2186 |
| helper | 843, 844, 845 | 3030, 3031, 3032 | 2187 |
| helper | 10131, 10132 | 12317, 12318 | 2186 |
| helper | 20262, 20263, 20264 | 22449, 22450, 22451 | 2187 |
| leader | 0, 1 | 2187, 2187 | 2187, 2186 |
| leader | 843, 844, 845 | 3029, 3030, 3031 | 2186 |
| leader | 21101 through 21107 | 23288 through 23294 | 2187 |

Leader's final frame 21108 has best local candidate 23294 at RMSE 2.222, above the acceptance threshold and well above typical accepted errors. Continuing its late offset 2187 would require index 23295, beyond EOF. This is an unmatched endpoint in the inspected neighborhood, not a claim that an exhaustive whole-video search was performed. The initial three-frame aggregate search favors 2186, while the first individual frame favors 2187; the two fields are kept separate.

A bounded 120-frame helper tail check found **exactly eight identical decoded RGB images at trimmed indices 21101–21108**, with 21100 different. Its final 27 downsampled fingerprints are identical. The untrimmed helper ends with four identical decoded RGB images at 23291–23294. Multiple tail candidates tie, so assigning a unique ending index would be false precision. These are verified repetitions of decoded images; they do not establish duplicate capture timestamps or explain the 64-frame MP4/VRS count difference.

The evidence rejects a simple contiguous, one-to-one frame slice for both participants. It does not identify the exporter operation that caused every irregularity, distinguish all dropped versus inserted/resampled frames, or provide an exact full-recording map. Re-encoding and static/repeated images can obscure individual correspondences. No DEVICE_TIME mapping was created from these image matches.

## 4. Remote VRS inspection succeeded below 20 MB

Both objects support byte ranges. The probe made two HEAD and ten bounded GET requests. Every GET returned 206 with the requested `Content-Range`, identity encoding, and stable object ETag. The utility rejects a 200/full-body fallback before reading its body, caps response reads, persists its budget before requests, and does not follow redirects or retry automatically.

Total response bodies read: **18,273,269 bytes**. Conservative reservation including 65,536 bytes per request: **19,059,701 bytes**, below 20,000,000. Network probing is stopped. Small fragments are stored under `outputs/comind_sync_forensics/remote_vrs/`; no sparse imitation of a full VRS was created and no full-file SDK reader was invoked on fragments.

The VRS headers point to front description records and compressed indexes. Reading those complete indexes was sufficient to enumerate streams and RGB record counts; four small endpoint-record reads provided the original integer capture timestamps. Parsing follows the official [VRS file format](https://github.com/facebookresearch/vrs/blob/main/vrs/FileFormat.h), [description format](https://github.com/facebookresearch/vrs/blob/main/vrs/DescriptionRecord.cpp), and [index implementation](https://github.com/facebookresearch/vrs/blob/main/vrs/IndexRecord.cpp). Indexed record extents sum to each exact remote object size.

| Native RGB observation | Helper | Leader |
| --- | ---: | ---: |
| Stream | `214-1` (`ArianeRgb`, camera ID 2) | `214-1` (`ArianeRgb`, camera ID 2) |
| RGB data records | **21,045** | **21,112** |
| Trimmed MP4 frames | 21,109 | 21,109 |
| First `capture_timestamp_ns` | **919390269025** | **825528737150** |
| Last `capture_timestamp_ns` | **1620744706862** | **1529116146612** |
| First / last native source frame number | 1927 / 22971 | 2197 / 23308 |
| Retained scalar MP4 timestamp | 919623571487 | 825528737150 |
| Candidate equal index-record time | RGB index 7 | RGB index 0 |

All endpoint integers are native device times in nanoseconds in their respective participant clocks. The complete index arrays preserve original double-precision RECORD_TIME seconds; they were **not** promoted to exact per-frame DEVICE_TIME integers. Equality of an MP4 scalar and one index timestamp is candidate timing evidence, not image correspondence. Project Aria documents the [distinction between record and capture timestamps](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/aria_vrs/timestamps_in_aria_vrs).

The unequal counts rule out a complete one-to-one correspondence of the two whole streams. Helper has fewer VRS records than MP4 frames, so every MP4 frame cannot have a distinct native record. Leader has three more VRS records, so a prefix/subset identity mapping remains possible in principle; counts alone cannot prove or reject it. Neither export map has been verified.

Both descriptions contain parsed native calibration JSON with `camera-slam-left`, `camera-slam-right`, `camera-et-left`, `camera-et-right`, and `camera-rgb`. Both report the same shared session ID, with TICSync enabled: helper client, leader server. The indexes contain helper SyncData and TimeDomainMapping streams and a leader TimeDomainMapping stream. Those facts do not establish a numeric cross-clock mapping: mapping-record modes/values were not decoded or installed. Native calibration has not been bound to the MP4's rotation/crop/resize chain. Full inventories and evidence are in the [remote VRS report](../../outputs/comind_sync_forensics/43276420-701f-4731-b9ab-bebc7fd14994_remote_vrs.json).

## 5. What a full VRS download would and would not solve

For these exact files, a complete, integrity-verified download would provide native RGB image content and every RGB capture timestamp, plus local access to the stream counts, calibration and synchronization records. Counts, native calibration metadata and endpoint timestamps have already been obtained remotely. Official [data-provider APIs](https://facebookresearch.github.io/projectaria_tools/docs/data_utilities/core_code_snippets/data_provider) expose streams, image records, counts and DEVICE_TIME; [calibration APIs](https://facebookresearch.github.io/projectaria_tools/docs/data_utilities/core_code_snippets/calibration) expose the stored device/camera calibration.

For each participant, native RGB capture times can be compared with that participant's MPS tracking times using the documented nanosecond/microsecond unit relation: `tracking_timestamp_us` is Aria device time. This supplies a shared **within-device time domain**, not equality of image and trajectory sample times or a mapping between participant clocks. Retain the originals and report nearest-sample residuals, maximum gaps and interpolation policy. [Official MPS trajectory schema](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/slam/mps_trajectory).

Downloading does **not** automatically give `MP4 frame i == VRS RGB record i`. The official Aria exporter can repeat the previous image when capture frames are dropped and applies image rotation/calibration updates. This documents why correspondence needs checking; it does not prove CoMind used that exporter or those exact operations. [Official VRS-to-MP4 documentation](https://facebookresearch.github.io/projectaria_tools/docs/data_utilities/advanced_code_snippets/vrs_to_mp4).

Minimum validation after any future VRS download:

1. Verify published size and SHA-256; identify the actual RGB stream and extract original integer capture timestamps with participant-specific clock provenance.
2. Compare native RGB images with the MP4, accounting only for verified pixel rotation/crop/resize transformations. Check start, end, spaced anchors and consecutive neighborhoods to expose repeats, skipped records and drift.
3. Establish each exported frame's correspondence, or obtain an authoritative export map/procedure accounting for trim, resampling, duplicates and omissions. Equal counts alone would still be insufficient. For a limited handover clip, verify every frame in that clip and its context; whole-recording synchronization requires coverage of the whole recording. Leave static/ambiguous images unresolved unless independent evidence disambiguates them.
4. Validate each participant's mapped RGB times against hands and MPS/Multi-SLAM samples with explicit units, residuals and gaps. If cross-participant matching uses TICSync, inspect and verify its mapping modes/values; a shared session ID is not a clock transform. Alternatively, CoMind's documented paired-video alignment plus verified per-participant export-to-device maps can support a common paired-frame timeline while retaining separate device clocks and explicit alignment provenance.
5. Bind native intrinsics/extrinsics to the verified MP4 pixel geometry. Keep scene registration, camera extrinsics and time synchronization as separate claims.

The unresolved blocker is the **per-frame MP4 export/device-time correspondence**, plus verification of any cross-device mapping used by the synchronized representation. Native geometry and matching utilities remain usable; synchronized MP4/hand playback is not enabled from an assumed frame rate, scalar timestamp, or shared-world frame.

## 6. Publisher fallback and decision

The official [CoMind site](https://comind.ethz.ch/) describes aligned streams and lists `agavryushin@ethz.ch` as contact. Its GitHub anchor currently points to `#`. The [paper's synchronization section](https://arxiv.org/html/2607.06691#S3.SS3) describes TICSync, GoPro audio alignment and common-start trimming, but the inspected public material does not provide this recording's export map or a timestamp-sidecar path. The [coauthor institution's publication page](https://www.mcml.ai/publications/gzh%2B26/) labels a link GitHub but points back to the project site. The official downloader and all relevant manifests exposed no separate sidecar. This is a scoped search result, not a claim that unpublished metadata cannot exist.

**Choose B.** A small metadata download has been investigated and the available transcripts do not solve synchronization. Ask for the existing mapping before spending 22.79 GiB on files that would still require image/export validation. If the authors cannot supply it, the exact trimmed VRS files are a justified fallback for native-image forensics, not an automatic synchronization fix.

Draft email — **not sent**:

> To: Alexey Gavryushin <agavryushin@ethz.ch>
>
> Subject: CoMind MP4-to-Aria timestamp map for recording 43276420-701f-4731-b9ab-bebc7fd14994
>
> Hi Alexey,
>
> For recording 43276420-701f-4731-b9ab-bebc7fd14994, could you point us to the `helper_trimmed_sync.mp4` and `leader_trimmed_sync.mp4` frame-index → Aria DEVICE_TIME timestamp arrays, or their MP4/VRS export and trim mapping? Project Aria's official extractor returns one timestamp per MP4 rather than 21,109. A sidecar path or confirmation of the exact RGB-record correspondence (including repeated/dropped frames) would let us align the provided hands and Multi-SLAM poses without downloading both VRS files.
>
> Thanks,
>
> Allen

The standalone [email draft](../../outputs/comind_sync_forensics/email_draft.md) is ready for review. No message was sent.

For completeness, the official fallback command below has **not been run**. It downloads only the two VRS files into a new staging target, avoiding changes to existing raw healthcheck/split files:

```sh
.venv/bin/python scripts/comind_download.py \
  outputs/comind_sync_forensics/43276420-701f-4731-b9ab-bebc7fd14994_only.txt \
  --parts vrs --no-meshes --no-annotations \
  --target data/staging/comind_vrs_only --yes
```

Additional payload would be 24,473,487,418 bytes (22.79271131 GiB). Current raw CoMind file payload is 15,782,843,495 bytes; after installing only those VRS files it would be 40,256,330,913 bytes. Keeping a complete staging copy as well would add another 24,473,487,418 bytes. Filesystem overhead, caches and other generated outputs are excluded. Installation would require hash verification and creation of missing destinations only; it is not part of this pass.

## 7. Changes and verification

This pass adds three forensic scripts and two focused test files:

- `scripts/probe_comind_remote_vrs.py`: persistent bounded HTTP-range budget, response validation and captured evidence.
- `scripts/inspect_comind_vrs_probe.py`: offline parsing of the downloaded VRS metadata/index fragments; no network access.
- `scripts/probe_comind_video_trim.py`: bounded decoded-image comparison, candidate margins, explicit uncertainty and tail checks.
- `tests/test_remote_vrs_probe.py`: nineteen synthetic tests for HTTP safety, concurrent budget locking, cached-artifact integrity and VRS metadata parsing.
- `tests/test_comind_video_trim.py`: nine synthetic tests for frame matching, ambiguous matches, offsets and repeated tails.
- This decision report, generated inventory/probe reports, manifests, download receipt, small probe fragments, and the unsent email draft under `outputs/comind_sync_forensics/`.
- The six previously missing transcript payloads listed above; no existing raw file was overwritten.

No adapter/viewer implementation was redone, no large asset was copied/decompressed, and no dependency was added in this pass.

Validation: **536 tests passed**. Ruff passes for `src`, `tests` and all three new scripts; all five new Python files pass Ruff's format check. Repository-wide `ruff check .` still reports the same **33 pre-existing findings** confined to `scripts/comind_download.py` and `scripts/inspect_comind.py`; these existing scripts were left unchanged. The official downloader's byte identity is preserved.

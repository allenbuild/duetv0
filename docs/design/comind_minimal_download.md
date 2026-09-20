# Minimal CoMind download and remote VRS processing

The synchronized Duet handover demo now has an individual-file acquisition path.
For recording `2fc0aa53-9070-4c86-81c2-41450253c74d`, the local recording inputs total
**4,253,520,380 bytes** (4.254 GB). The original complete recording transfer from
the official manifest totals **81,372,146,897 bytes** (81.372 GB), after the
official exclusion of non-trimmed VRS. This comparison excludes dataset-wide
meshes and annotations; adding those to the original command would cost more.

This work plans acquisition only. No recording asset download, remote VRS range
read, or new recording processing was authorized or executed. The cached
`data/processed/comind/2fc0aa53-9070-4c86-81c2-41450253c74d/download_plan/healthcheck.json`
provides exact individual sizes and SHA-256 hashes. The planned files have not
yet supplied target-recording calibration, synchronization, or graph evidence.

## Exact local inputs

Paths below are relative to
`data/raw/comind/recordings/2fc0aa53-9070-4c86-81c2-41450253c74d/`.
The downloader preserves these official paths.

| Selected file | Exact bytes | Pipeline use |
| --- | ---: | --- |
| `mp4s/helper_trimmed_sync.mp4` | 1,729,299,362 | Helper synchronized imagery and exact MP4 output PTS |
| `mp4s/leader_trimmed_sync.mp4` | 1,490,091,792 | Leader synchronized imagery and exact MP4 output PTS |
| `mps_helper_trimmed_vrs/hand_tracking/hand_tracking_results.csv` | 55,115,758 | Helper hand annotations in its device frame |
| `mps_leader_trimmed_vrs/hand_tracking/hand_tracking_results.csv` | 56,196,755 | Leader hand annotations in its device frame |
| `multislam_output/0/slam/closed_loop_trajectory.csv` | 460,682,842 | Complete shared-world trajectory and graph IDs |
| `multislam_output/1/slam/closed_loop_trajectory.csv` | 462,133,713 | Complete shared-world trajectory and graph IDs |
| `multislam_output/vrs_to_multi_slam.json` | 158 | Explicit VRS participant to Multi-SLAM output mapping |
| **Total new local recording data** | **4,253,520,380** | |

The existing common file
`data/raw/comind/annotations/dataset_handover_consolidated.json` is **259,041 bytes**
and is reused after size and hash verification against the existing
`annotations/healthcheck.json`. It provides the target's 11 handover intervals.
No additional annotation transfer is needed here. On another installation,
provide a saved annotation healthcheck with `--annotations-manifest`; only the
handover JSON is selected if it is missing, never the complete annotation part.

Selecting output directories `0` and `1` does **not** assign either to helper or
leader. With exactly two complete trajectory files, both can be selected before
the tiny mapping file is available. Processing then parses the exact recording
and VRS keys in `vrs_to_multi_slam.json`. A recording with additional trajectory
directories requires an already local, manifest-verified mapping before the
minimal planner selects its two trajectories.

The semantic stage reads every trajectory row and requires one nonempty
`graph_uid` per trajectory and the same graph for both participants. Consequently,
`summary.json`, `summary.json.zip`, and `vrs_health_check.json` add no graph
verification input. Shared coordinates do not establish a shared device clock;
existing synchronization evidence and device-local timestamps remain separate.

## Calibration and bounded remote VRS evidence

No local full VRS, standard MPS SLAM trajectory, `online_calibration.jsonl`,
open-loop trajectory, semidense points, observations, GoPro imagery, scan, mesh,
or transcript is needed by this path.

`src/duet/adapters/comind/remote_vrs.py` reads the remote VRS header, description,
configuration, and classic compressed index using explicit HTTP ranges. It
preserves each indexed RGB record's original `RECORD_TIME` double and extent.
For requested RGB records it obtains the original integer
`capture_timestamp_ns` (`DEVICE_TIME`) and source frame number from the record
metadata. It never manufactures capture times by rounding index doubles,
subtracting an assumed offset, or fitting a device-clock conversion. Unknown
capture timestamps remain `-1` with `metadata_known=False`.

The VRS description's native `calib_json` supplies the RGB camera calibration.
The recorded RGB configuration supplies the actual image dimensions, and the
Project Aria calibration API rescales the native camera model to those recorded
dimensions. This avoids downloading the large MPS online calibration logs.
The existing visual orientation and full-resolution pixel-geometry checks must
then establish the exported MP4 geometry before camera calibration is accepted.
No new target calibration or orientation is assumed from a previous recording.

The remote mode of `scripts/build_comind_frame_map.py` reuses the existing MP4
fingerprint cache, sparse anchors, image matcher, frame-map schema, and geometry
audit. It fetches RGB candidate records for all 11 handover windows, **120 MP4
frames of context on each side**, and sparse global anchors. Candidate selection
may interpolate a search center, but a verified mapping requires direct visual
evidence plus exact record metadata. Other MP4 rows remain `UNRESOLVED`; the
result is intentionally a partial recording frame map. Ambiguous matches and
missing records also remain unresolved. The normal downstream validation gates
determine which handovers can be exported.

Only compact derived metadata is cached beneath `data/processed/`: record
indexes, exact observed timestamps, native calibration, 32 × 32 fingerprints,
validity masks, frame maps, and request ledgers. JPEG records and other VRS byte
ranges are decoded in memory and discarded. No sparse file with VRS-sized holes,
full VRS file, or cached RGB payload is produced.

HTTP reads require status 206, the exact requested `Content-Range`, manifest file
length, identity encoding, and a stable strong ETag. Each request is reserved
against a persistent byte budget before it starts; redirects, whole-file GET
fallback, and automatic retries are disabled. The default conservative budget
is **4,000,000,000 bytes per role**, including a per-request transport allowance.
Exhausting it stops the run; it does not trigger a larger transfer automatically.
The budget is a limit, not an estimate or a completion guarantee.

A partial VRS read cannot verify the official hash of the entire VRS file.
The saved manifest hash records source provenance; strong ETag consistency,
SHA-256 of each received range, record headers/extents, exact timestamps, and
derived-cache hashes provide the available partial-read evidence. Reports must
not describe a full VRS hash as verified.

## Planning and manual commands

All commands below run from `/Users/allenxu/Documents/duet-research` using the
existing environment, including its installed Project Aria tools. The commands
are documented for the user; the transfer and processing commands were not run
as part of this implementation.

This fully offline dry-run reads the saved manifest and existing annotation
metadata, prints each selected path, byte size, and official hash, and optionally
writes a derived report. It does not request recording assets or VRS ranges:

```sh
.venv/bin/python scripts/comind_download_minimal.py \
  2fc0aa53-9070-4c86-81c2-41450253c74d \
  --dry-run \
  --manifest data/processed/comind/2fc0aa53-9070-4c86-81c2-41450253c74d/download_plan/healthcheck.json \
  --remote-vrs-estimate-bytes 4134659295 \
  --report data/processed/comind/2fc0aa53-9070-4c86-81c2-41450253c74d/download_plan/minimal_plan.json
```

Without `--manifest`, planning fetches only the official recording
`healthcheck.json`, using `fetch_manifest` from the unchanged official downloader.
The official `build_url`, byte formatting, and non-trimmed VRS exclusion logic
are reused. Dry-run is the default even when no mode flag is supplied.

The exact manual download command is:

```sh
cd /Users/allenxu/Documents/duet-research
.venv/bin/python scripts/comind_download_minimal.py \
  2fc0aa53-9070-4c86-81c2-41450253c74d \
  --download \
  --manifest data/processed/comind/2fc0aa53-9070-4c86-81c2-41450253c74d/download_plan/healthcheck.json
```

`--download` only fetches the seven selected assets and any explicitly planned
missing handover annotation file. It does not start remote VRS processing.
Existing files are read and hash-verified before transfer: verified files are
skipped and mismatches fail without replacement. New assets are streamed to an
exclusively created `.part`, checked against the official byte size and digest,
then published using an atomic OS no-overwrite rename. The downloader rejects
unsafe paths, symlink destinations, and pre-existing partial files. It does not
modify original raw assets, silently replace a mismatching file, or resume an
unverified partial file.

After the manual download succeeds, run these stages in order, stopping if a
stage fails. The first stage uses only the downloaded trajectories and hands:

```sh
.venv/bin/python scripts/validate_comind_semantics.py \
  --recording-id 2fc0aa53-9070-4c86-81c2-41450253c74d
```

The next command explicitly permits bounded remote VRS reads and builds the
existing frame maps and pixel-geometry evidence. It is a separate user-initiated
network step:

```sh
.venv/bin/python scripts/build_comind_frame_map.py \
  --recording-id 2fc0aa53-9070-4c86-81c2-41450253c74d \
  --remote-vrs --fetch \
  --manifest data/processed/comind/2fc0aa53-9070-4c86-81c2-41450253c74d/download_plan/healthcheck.json \
  --range-budget-bytes 4000000000 \
  --audit-image-geometry
```

The remaining stages reuse the derived remote metadata and existing demo code;
they need no full VRS or additional dataset assets:

```sh
.venv/bin/python scripts/verify_comind_rgb_calibration.py \
  --recording-id 2fc0aa53-9070-4c86-81c2-41450253c74d

.venv/bin/python scripts/validate_comind_frame_map.py \
  --recording-id 2fc0aa53-9070-4c86-81c2-41450253c74d

.venv/bin/python scripts/validate_comind_projection.py \
  --recording-id 2fc0aa53-9070-4c86-81c2-41450253c74d

.venv/bin/python scripts/export_comind_v0_examples.py \
  --recording-id 2fc0aa53-9070-4c86-81c2-41450253c74d \
  --output-dir outputs/v0_demo/2fc0aa53-9070-4c86-81c2-41450253c74d
```

## Transfer estimate and verification status

The completed offline dry-run reports:

| Transfer | Bytes | Decimal GB |
| --- | ---: | ---: |
| Old official full recording selection | 81,372,146,897 | 81.372 |
| Seven new local files | 4,253,520,380 | 4.254 |
| Estimated remote VRS ranges, both participants | 4,134,659,295 | 4.135 |
| **Expected total** | **8,388,179,675** | **8.388** |

The range estimate covers all 11 handovers with context and global calibration
anchors: approximately 5,066 distinct RGB records per role and 10,148 HTTP
requests total. Roughly 35.7 MB is index/description/configuration data; the
remaining range traffic is sparse RGB imagery needed for direct visual matching.
Using the prior recording's 95th-percentile RGB record size for every requested
image instead gives 4,346,991,539 bytes of range bodies (8.601 GB including local
files). This is a size scenario, not a statistical confidence interval.
The request ledger's separate conservative reservation adds 65,536 bytes per
request; its 4 GB per-role cap is deliberately above these estimates.

Machine-readable evidence is in the sibling derived `download_plan/minimal_plan.json`
and `download_plan/remote_vrs_estimate.json`, including the reference record sizes,
source manifest SHA-256, and all selected file digests. No target raw directory
was created by the dry-run.
No target VRS probe is required to produce the planning estimate. The estimate
uses the existing recording's compact RGB record-size statistics and the
target's annotation windows, and must be labeled as an estimate. Actual JPEG
sizes, anchor candidates, and request latency can differ. The final dry-run can
include it through `--remote-vrs-estimate-bytes`; this option never authorizes or
performs a VRS request.

For a three-hour transfer target, the required effective throughput is
**6.214 Mbps** including request overhead, or about **1 hour 52 minutes** of
payload transfer at 10 Mbps before request latency. MP4 decoding, integrity
checks, and rendering add elapsed time. No live throughput test was performed.
The hard per-role range budget remains separate from this estimate.

The downloader's 27 synthetic tests cover exact selection, missing assets,
ambiguous output directories, offline dry-run, mode gating, hash and length
failures, truncated/oversized HTTP bodies, immutable existing data, symlinks,
exclusive partial creation, and concurrent no-clobber publication. The full
suite passes: **860 tests**, with Ruff clean for the changed code. Offline reads of
the existing recording additionally matched the SDK's exact capture timestamps
and RGB pixels at first/middle/last samples for both participants; rescaled
native calibration parameters also matched exactly. No target VRS bytes were
read for those checks. Synthetic verification does not establish the
unread target recording's real-world matching or graph/calibration correctness.

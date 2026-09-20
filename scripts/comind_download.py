#!/usr/bin/env python3
"""
CoMind v1 dataset download script -- CVG Group, ETH Zurich
Version: 20260626-01
Reference: https://comind.ethz.ch

Downloads CoMind v1 recordings into a target directory (default
``./dataset/v1/``).  The recording IDs to download are read from a text file,
one UUID per line; if no file is given, a built-in default list (the test split
first, then the train split) is used.  Passing the literal ``train`` or ``test``
in place of a file downloads only that built-in split.  Each recording is placed
in its own sub-folder under a ``recordings/`` parent
(``<target>/recordings/<RECORDING_ID>/``).

When a download starts, the built-in train and test splits are also written to
``<target>/split/train.txt`` and ``<target>/split/test.txt``.

The dataset's *meshes* and *annotations* common assets are downloaded too, into
``<target>/meshes/`` and ``<target>/annotations/`` from ``<base-url>/meshes/``
and ``<base-url>/annotations/``.  They are offered as their own entries in the
parts selection (so they can be picked or dropped like any recording part, or
skipped outright with ``--no-meshes`` / ``--no-annotations``) and, when selected,
are downloaded *before* any recordings.  Like recordings, each is published with
a ``healthcheck.json`` manifest, used to know which files to fetch and to verify
them after download.  (The annotations common asset contains JSON files; its own
``healthcheck.json`` manifest is the only file there parsed as a manifest -- it
is never listed among the annotation files it describes.)

Every recording on the server is published together with a ``healthcheck.json``
manifest at:

    https://comind.ethz.ch/dataset/<RECORDING_ID>/healthcheck.json

The manifest lists each file belonging to the recording (path, byte size and
content hash) plus the hash algorithm used.  This script first fetches all of
the requested manifests and uses them to:

  * know exactly which files to download for each recording,
  * compute and display a total download-size estimate, and
  * (where possible) compare that estimate against the free space available in
    the target directory before asking for confirmation.

By default you are asked which *parts* of each recording to fetch (trimmed VRS,
MP4 videos, GoPro, MPS, multi-SLAM, transcripts, ...); this can also be given
non-interactively with ``--parts`` (e.g. ``--parts vrs,mp4`` or
``--parts all,-gopro``).  Raw, non-trimmed VRS files are never downloaded.  The
meshes / annotations common assets appear in the same selection as the keys
``meshes`` and ``annotations`` and are fetched whole when selected.

Recordings are then downloaded one after another.  While downloading, a live
display shows, for the current file, the current recording and the whole job:
a progress bar, the transferred / total size and an ETA.  After each recording
folder finishes downloading, every file in it is re-hashed and checked against
the manifest (unless ``--skip-verification`` is given).

Partial files are resumed on re-runs when the server supports HTTP range
requests, and already-complete files are skipped.

Usage
-----
    python3 comind_download.py                 # built-in default recording list
    python3 comind_download.py train           # built-in train split only
    python3 comind_download.py test            # built-in test split only
    python3 comind_download.py recordings.txt
    python3 comind_download.py recordings.txt --target /data/comind/v1
    python3 comind_download.py recordings.txt --skip-verification --yes
    python3 comind_download.py recordings.txt --parts vrs,mp4
    python3 comind_download.py recordings.txt --no-meshes --no-annotations
    python3 comind_download.py recordings.txt --list-parts
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

DEFAULT_BASE_URL = "https://comind.ethz.ch/dataset"
DEFAULT_TARGET = "./dataset/v1/"
MANIFEST_NAME = "healthcheck.json"
# Within the target directory recordings live under recordings/<RECORDING_ID>/
# and the common assets (meshes, annotations) under meshes/ and annotations/
# (mirroring the server layout under <base-url>/<subdir>/).
RECORDINGS_SUBDIR = "recordings"
MESHES_SUBDIR = "meshes"
ANNOTATIONS_SUBDIR = "annotations"
USER_AGENT = "comind-v1-download/1.0 (+https://comind.ethz.ch)"
DOWNLOAD_CHUNK = 4 * 1024 * 1024  # 4 MiB
READ_CHUNK = 8 * 1024 * 1024      # 8 MiB (verification)
DEFAULT_RETRIES = 4
DEFAULT_TIMEOUT = 60.0            # seconds, per network read

UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# Matches a "Version: <tag>" line in the script's own header (see read_version_tag).
_VERSION_RE = re.compile(r"^\s*Version:\s*(\S.*?)\s*$")


# --------------------------------------------------------------------------- #
# Built-in default recording splits                                           #
# --------------------------------------------------------------------------- #
# Baked-in train / test recording-ID splits.  When no ids file is passed the
# union of these (test split first, then train) is the default download set, and
# both lists are written to <target>/split/{train,test}.txt when a download starts.

TRAIN_IDS = [
    "01d63ce5-f734-4647-93c8-ea4af07b8c7f",
    "0542e1ea-8c26-4f86-9ff1-b66adacb7639",
    "0ea30478-5d19-410b-9b96-ee8f4fb4dc15",
    "0eb5f8d2-5ed2-4a8d-927c-cadda0dfae9e",
    "13175f01-e72c-42d5-87e8-737633ecfe27",
    "1326e688-9e84-4d1a-aef7-a588ca2cc47d",
    "14aaebe8-0a7f-43f0-b7f3-78ac6127df75",
    "16b9e645-0f3b-4caf-bbdd-f89d4e3387b3",
    "199d1b52-2f0b-44bd-806c-d8c61e851433",
    "1df816f6-d987-4c00-bc32-22ab6f08aa18",
    "222861ac-b5d7-4f92-b682-d6c639ede7eb",
    "2fc0aa53-9070-4c86-81c2-41450253c74d",
    "3c254eb1-5a00-4e73-b72f-1fc1f1cd8167",
    "3ef7356e-3b75-4eab-add9-363b0cdc2b99",
    "43276420-701f-4731-b9ab-bebc7fd14994",
    "451bf296-ed93-4f29-9ced-81f3a4f8c0cf",
    "5533b1d7-838f-4b1d-9b54-a5bf6e8e3128",
    "571b195e-95dc-4686-a355-0b26a6ece873",
    "585b52a0-a019-44a0-b70f-531244971bcf",
    "5f8dc440-e2df-41a7-b144-14de3877b8fb",
    "62531cc1-b19e-452c-96b4-d1e60f077e3f",
    "63a789f1-32bf-4cfc-815b-0f4e93dcbfb3",
    "6473136b-6037-46be-a8b0-fb86b5997376",
    "665e4989-28c4-4d1a-84fe-a425ea193ebe",
    "68b97376-0dce-4dc3-93ae-01f3b3977b57",
    "69fed21a-8fb3-4757-a728-8b0dc3dce828",
    "73e68fb0-de34-4982-b0a3-6c5ae70442ba",
    "806c599c-ebb7-4bd8-9372-84582bdf0a4c",
    "833eac29-25e8-476d-b81c-f311cac0fd28",
    "845323e6-ca6d-4eeb-87f0-590b5ce08aa5",
    "8ee6c694-38be-4a90-ad5a-75ef4449e972",
    "968a90d3-2fba-49ca-a34c-ccfe6939b3c7",
    "9db1ea2e-9677-41f3-8bb6-3dc7d61bd962",
    "9fcdca26-9acd-4dec-a940-bb425080bc3a",
    "a20c8ae1-147d-4996-8cf8-69c599a710f5",
    "a268a350-c55f-4f75-832b-52750ca30bce",
    "a5081f80-f7cb-475e-9c8d-3b9567d69a02",
    "a9e48896-8f2f-431b-9ad9-ad4c6b37b58e",
    "ae5b5499-2913-4e10-85f0-1fec63a3791f",
    "b1446007-e7d8-4463-83cd-5a6c187ef027",
    "b4f68f2b-2460-48e4-bd6d-863a25a796f9",
    "bb657948-9db6-4f64-807b-8fd36d6bb00f",
    "c5fdc20b-d865-43d8-a5e8-52af272ac17e",
    "c807cf0e-c32e-4c57-9215-603047e8f471",
    "c8776f57-6f73-41a9-974e-1890cc2bb7c1",
    "cdd2114f-4bef-401e-803f-a8da2327bb37",
    "d053e943-5dd5-4833-a0c8-813de6c93a24",
    "d8d72aca-ecd2-485f-a725-f391bcc4ef87",
    "d90d7d88-24a0-48f1-8406-bae0d5ceac27",
    "da2e013e-e431-4788-9dd7-7bc436e9376b",
    "e1868637-8cfa-41d2-b6e1-8b2754383dbe",
    "e30c47bf-c150-4a1b-aa04-76a5c3a95c54",
    "eb784196-2ece-4c95-b430-91a20b985df2",
    "f867da74-9841-4261-8ae9-3193029e0568",
    "f87737d4-bf15-4242-8082-e6cb055867c4",
]

TEST_IDS = [
    "f0b49334-565a-4d30-a010-15d92e511c9a",
    "d1c6ae3a-3f7b-4193-9923-2e029d9513fd",
    "7b04b3ef-43ed-4c46-ab3d-0d3baa25352e",
    "58dd70af-0c19-486e-9bdc-24b5c22b49e6",
    "63f25fa8-4573-48da-84e4-4cd5ea1c2701",
    "f6968635-3d7c-4784-8b45-1354f7a1d68f",
    "0b9598ec-b2a0-4211-86c3-d77d2c82cb8f",
    "d7f489e1-227a-4de7-aa07-7b7d37afd139",
    "4621cc65-f05d-46fb-aea2-7245d513e959",
    "21c13149-ca54-45dc-94a1-bd74a1c8a27e",
    "4a93a8f7-305e-489b-b4e7-c31253507ade",
    "b19c41f5-719d-4a93-92e2-0555cad8a607",
    "87f944b2-ff9d-47fe-bf59-6c374f139356",
    "313483e1-e51a-44f6-8e73-3d80d56fe08e",
    "8f8b5bf7-a8ab-41f2-a912-d42139e603e1",
    "d8fcc3dd-e9bb-42e2-b8fd-b2ec5c5e54f1",
    "482e5b50-8c69-4d03-b6d8-e4932a981711",
    "c4944bc8-994d-4ed2-9d99-9812c3275bf0",
    "11bd710a-d8f3-4e3c-88e4-d97d7bc8b3f5",
    "b1ade277-a777-43cf-92a6-ede187252a5c",
    "9a667685-1cdd-4097-9138-9138faf388b2",
    "4fbe9724-b960-4d6c-a016-6b545557b7de",
    "f794039a-f9c7-49f1-82f4-39f93c561523",
    "39097cb5-de7a-49d4-8217-b2c30e19a3ae",
    "e5a6754a-5149-4cc7-821a-bb967d8857bf",
]


# --------------------------------------------------------------------------- #
# Formatting helpers                                                          #
# --------------------------------------------------------------------------- #

def human_bytes(n):
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(n) < 1024.0 or unit == "PiB":
            return f"{int(n)} B" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024.0


def fmt_duration(secs):
    if secs is None or secs != secs or secs in (float("inf"),) or secs < 0:
        return "--:--"
    secs = int(secs)
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def progress_bar(frac, width=26):
    frac = 0.0 if frac < 0 else 1.0 if frac > 1 else frac
    full = int(round(frac * width))
    return "[" + "#" * full + "-" * (width - full) + "]"


def shorten(text, width):
    return text if len(text) <= width else "..." + text[-(width - 3):]


# --------------------------------------------------------------------------- #
# Input parsing                                                               #
# --------------------------------------------------------------------------- #

def read_id_list(path):
    ids, seen = [], set()
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not UUID_RE.match(line):
                print(f"  ! line {lineno}: '{line}' is not a UUID -- ignored",
                      file=sys.stderr)
                continue
            low = line.lower()
            if low not in seen:
                seen.add(low)
                ids.append(line)
    return ids


def build_url(base_url, recording_id, relpath):
    """Join base / recording / relpath, URL-quoting each path segment."""
    quoted = "/".join(urllib.parse.quote(seg) for seg in relpath.split("/"))
    return f"{base_url.rstrip('/')}/{recording_id}/{quoted}"


def baked_default_ids():
    """The built-in default recording set: test split first, then train."""
    ids, seen = [], set()
    for rid in list(TEST_IDS) + list(TRAIN_IDS):
        low = rid.lower()
        if low not in seen:
            seen.add(low)
            ids.append(rid)
    return ids


def write_split_files(target):
    """Write the baked-in train/test ID lists to <target>/split/{train,test}.txt.

    Returns the list of paths written.  These come straight from the constants
    compiled into this script -- nothing is read from disk to produce them.
    """
    split_dir = os.path.join(target, "split")
    os.makedirs(split_dir, exist_ok=True)
    written = []
    for name, ids in (("train.txt", TRAIN_IDS), ("test.txt", TEST_IDS)):
        path = os.path.join(split_dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(ids) + "\n")
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# File categories ("parts")                                                   #
# --------------------------------------------------------------------------- #

def is_non_trimmed_vrs(relpath):
    """True for VRS files that are not the trimmed VRS (excluded everywhere).

    Only files under ``trimmed_vrs/`` are kept; raw / eye-calibration / gap-fill
    VRS files (e.g. ``eye_calibration_helper.vrs``) are never downloaded.
    """
    return relpath.lower().endswith(".vrs") and not relpath.startswith("trimmed_vrs/")


# Ordered (key, label, predicate). First match wins; "other" is the catch-all
# and must stay last so every file maps to exactly one part.
PART_DEFS = [
    ("vrs",         "Trimmed VRS",              lambda p: p.startswith("trimmed_vrs/")),
    ("mp4",         "MP4 videos",               lambda p: p.startswith("mp4s/")),
    ("gopro",       "GoPro videos",             lambda p: p.startswith("gopro_")),
    ("mps",         "MPS (eye-gaze/hand/SLAM)", lambda p: p.startswith("mps_")),
    ("multislam",   "Multi-SLAM output",        lambda p: p.startswith("multislam_output/")),
    ("transcripts", "Transcripts",              lambda p: p.startswith("transcripts/")),
    ("scan",        "3D scan (.b2g + .pcd)",    lambda p: p.startswith("scan/") or p.lower().endswith((".b2g", ".pcd"))),
    ("other",       "Other",                    lambda p: True),
]
PART_KEYS = [k for k, _, _ in PART_DEFS]
PART_LABELS = {k: lbl for k, lbl, _ in PART_DEFS}

# Common assets: dataset-wide, non-recording data.  These are offered as their
# own entries in the parts selection (alongside vrs/mp4/...) and, when selected,
# are downloaded before any recordings.
AUX_DEFS = [
    (MESHES_SUBDIR,      "Meshes (common asset)"),
    (ANNOTATIONS_SUBDIR, "Annotations (common asset)"),
]
AUX_KEYS = [k for k, _ in AUX_DEFS]
AUX_LABELS = {k: lbl for k, lbl in AUX_DEFS}


def aux_rows(aux_mans):
    """Parts-table rows describing the fetched meshes / annotations common assets.

    Each aux manifest's recording_id is its key ('meshes' / 'annotations'), so
    it slots into the same (key, label, count, size) shape as recording parts.
    """
    return [(m["recording_id"], AUX_LABELS.get(m["recording_id"], m["recording_id"]),
             len(m["files"]), m["total_size_bytes"]) for m in aux_mans]


def categorize(relpath):
    """Return the part key for a manifest file path."""
    for key, _label, pred in PART_DEFS:
        if pred(relpath):
            return key
    return "other"


def part_breakdown(manifests):
    """Return ordered [(key, label, count, size), ...] for the parts present."""
    counts = {k: 0 for k in PART_KEYS}
    sizes = {k: 0 for k in PART_KEYS}
    for man in manifests:
        for e in man["files"]:
            k = categorize(e["path"])
            counts[k] += 1
            sizes[k] += e["size"]
    return [(k, PART_LABELS[k], counts[k], sizes[k]) for k in PART_KEYS if counts[k]]


def parse_part_selection(value, present_keys):
    """Parse a parts spec into a set of keys.

    Accepts keys or 1-based numbers, comma/semicolon separated, with ``-``/``!``
    prefixes to subtract: ``vrs,mp4``, ``all,-gopro``, ``-gopro`` (= all but
    GoPro), ``all``, ``none``.  Empty selects all.  Raises ValueError on an
    unknown token.
    """
    idx_to_key = {str(i + 1): k for i, k in enumerate(present_keys)}
    tokens = [t.strip().lower() for t in value.replace(";", ",").split(",") if t.strip()]
    if not tokens:
        return set(present_keys)
    has_include = any(not t.startswith(("-", "!")) for t in tokens)
    selected = set() if has_include else set(present_keys)
    for tok in tokens:
        neg = tok.startswith(("-", "!"))
        body = tok[1:] if neg else tok
        if body == "all":
            group = set(present_keys)
        elif body in ("none", ""):
            group = set()
        elif body in present_keys:
            group = {body}
        elif body in idx_to_key:
            group = {idx_to_key[body]}
        else:
            raise ValueError(f"unknown part '{tok}'")
        selected = (selected - group) if neg else (selected | group)
    return selected


def _files_word(n):
    return f"{n:>5} {'file ' if n == 1 else 'files'}"


def print_part_table(rows):
    print("\nAvailable to download (recording parts + common assets):")
    for i, (key, label, cnt, size) in enumerate(rows, 1):
        print(f"  [{i}] {key:11} {label:26} {_files_word(cnt)}  {human_bytes(size):>11}")
    tot_c = sum(r[2] for r in rows)
    tot_s = sum(r[3] for r in rows)
    print(f"      {'':11} {'TOTAL':26} {_files_word(tot_c)}  {human_bytes(tot_s):>11}")


def resolve_part_selection(args, rows):
    """Pick which part keys to download: --parts, interactive prompt, or all.

    Returns a set of keys, or None on an invalid --parts value.
    """
    present_keys = [r[0] for r in rows]
    if args.parts is not None:
        try:
            return parse_part_selection(args.parts, present_keys)
        except ValueError as exc:
            print(f"error: {exc}; available parts: {', '.join(present_keys)}",
                  file=sys.stderr)
            return None
    if args.yes or not sys.stdin.isatty():
        return set(present_keys)
    print_part_table(rows)
    while True:
        try:
            raw = input("Select parts to download "
                        "(keys/numbers, e.g. 'vrs,mp4' or 'all,-gopro') [all]: ")
        except EOFError:
            return set(present_keys)
        try:
            sel = parse_part_selection(raw, present_keys)
        except ValueError as exc:
            print(f"  {exc}; try again.")
            continue
        if not sel:
            print("  nothing selected; type some parts or Ctrl-C to abort.")
            continue
        return sel


def apply_part_selection(manifests, selected):
    """Filter each manifest's files to *selected* parts (in place).

    Recomputes total_size_bytes and drops recordings left with no files.
    The saved-to-disk manifest (``raw``) is left untouched. Returns
    (kept_manifests, n_dropped).
    """
    kept, dropped = [], 0
    for man in manifests:
        files = [e for e in man["files"] if categorize(e["path"]) in selected]
        if not files:
            dropped += 1
            continue
        man["files"] = files
        man["total_size_bytes"] = sum(e["size"] for e in files)
        kept.append(man)
    return kept, dropped


# --------------------------------------------------------------------------- #
# Manifest fetching                                                           #
# --------------------------------------------------------------------------- #

def fetch_manifest(base_url, url_id, timeout, dest_subpath=None, label=None):
    """Fetch and parse a healthcheck.json. Returns a dict.

    *url_id* is the path segment under *base_url* (a recording UUID, or the
    literal ``meshes``).  *dest_subpath* is where the files are written relative
    to the download target (defaults to ``recordings/<url_id>``); *label* is the
    human-readable name shown in logs (defaults to *url_id*).
    """
    url = build_url(base_url, url_id, MANIFEST_NAME)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    data = json.loads(raw.decode("utf-8"))
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest contains no file list")
    algo = data.get("hash_algorithm", "sha256")
    if algo not in hashlib.algorithms_available:
        raise ValueError(f"unsupported hash algorithm '{algo}'")
    clean = []
    for entry in files:
        path = entry["path"]
        size = int(entry["size"])
        digest = entry.get("hash")
        if path.startswith("/") or ".." in path.split("/"):
            raise ValueError(f"unsafe path in manifest: {path!r}")
        if is_non_trimmed_vrs(path):
            continue  # raw / non-trimmed VRS are excluded everywhere
        clean.append({"path": path, "size": size, "hash": digest})
    if not clean:
        raise ValueError("manifest has no downloadable files after exclusions")
    return {
        "recording_id": label or url_id,
        "url_id": url_id,
        "dest_subpath": dest_subpath or os.path.join(RECORDINGS_SUBDIR, url_id),
        "hash_algorithm": algo,
        "files": clean,
        "total_size_bytes": sum(e["size"] for e in clean),
        "raw": raw,
    }


def fetch_all_manifests(base_url, ids, timeout):
    """Fetch every manifest sequentially. Returns (ok_list, failures)."""
    print(f"Fetching manifests ({MANIFEST_NAME}) for {len(ids)} "
          f"recording(s), one at a time ...")
    ok, failures = [], []
    for i, rid in enumerate(ids, 1):
        prefix = f"  [{i}/{len(ids)}] {rid} "
        try:
            man = fetch_manifest(base_url, rid, timeout)
        except urllib.error.HTTPError as exc:
            failures.append((rid, f"HTTP {exc.code}"))
            print(prefix + f"UNAVAILABLE (HTTP {exc.code}) -- will skip")
        except (urllib.error.URLError, socket.timeout, ssl.SSLError) as exc:
            failures.append((rid, f"network: {exc}"))
            print(prefix + f"UNAVAILABLE (network error) -- will skip")
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            failures.append((rid, f"bad manifest: {exc}"))
            print(prefix + f"BAD MANIFEST ({exc}) -- will skip")
        else:
            ok.append(man)
            print(prefix + f"ok ({len(man['files'])} files, "
                  f"{human_bytes(man['total_size_bytes'])})")
    return ok, failures


def fetch_aux_manifest(base_url, subdir, timeout):
    """Fetch a common asset's manifest (<base>/<subdir>/healthcheck.json).

    Used for the ``meshes`` and ``annotations`` common assets: each is a single
    directory alongside the recordings, downloaded whole into
    ``<target>/<subdir>/`` and offered as one entry in the "parts" selection.
    (For annotations the
    manifest is itself a .json file living next to the annotation JSONs; it is
    excluded from its own file list by the healthcheck generator, so there is no
    confusion here -- we only ever parse <subdir>/healthcheck.json as a
    manifest.)  Returns the manifest dict, or None (with a message) if it is
    unavailable.
    """
    print(f"\nFetching {subdir} manifest ({subdir}/{MANIFEST_NAME}) ...")
    try:
        man = fetch_manifest(base_url, subdir, timeout,
                             dest_subpath=subdir, label=subdir)
    except urllib.error.HTTPError as exc:
        print(f"  {subdir} UNAVAILABLE (HTTP {exc.code}) -- will skip")
        return None
    except (urllib.error.URLError, socket.timeout, ssl.SSLError):
        print(f"  {subdir} UNAVAILABLE (network error) -- will skip")
        return None
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"  {subdir} BAD MANIFEST ({exc}) -- will skip")
        return None
    print(f"  {subdir} ok ({len(man['files'])} files, "
          f"{human_bytes(man['total_size_bytes'])})")
    return man


# --------------------------------------------------------------------------- #
# Disk-space check                                                            #
# --------------------------------------------------------------------------- #

def free_space_bytes(path):
    """Free bytes on the filesystem holding *path* (nearest existing parent).

    Returns None if it cannot be determined (e.g. permissions, unusual mount).
    Never requires elevated privileges.
    """
    try:
        probe = os.path.abspath(path)
        while not os.path.exists(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                return None
            probe = parent
        return shutil.disk_usage(probe).free
    except (OSError, ValueError):
        return None


def bytes_already_present(target, manifests):
    """Sum of bytes of manifest files already fully downloaded on disk."""
    present = 0
    for man in manifests:
        rec_dir = os.path.join(target, man["dest_subpath"])
        for entry in man["files"]:
            dest = os.path.join(rec_dir, entry["path"].replace("/", os.sep))
            try:
                if os.path.isfile(dest) and os.path.getsize(dest) == entry["size"]:
                    present += entry["size"]
            except OSError:
                pass
    return present


# --------------------------------------------------------------------------- #
# Live multi-level progress display                                           #
# --------------------------------------------------------------------------- #

class Display:
    """Renders file / recording / overall progress with ETAs.

    Counters are absolute byte totals; ``xfer`` accumulators track bytes really
    moved over the network this session and drive the speed/ETA estimates.
    """

    def __init__(self, total_bytes, to_transfer, enabled):
        self.tty = enabled and sys.stdout.isatty()
        self.enabled = enabled
        # Overall
        self.total_bytes = total_bytes          # full size of all recordings
        self.to_transfer = max(to_transfer, 1)  # bytes expected over network
        self.done_base = 0                      # bytes of fully-finished files
        self.session_xfer = 0                   # bytes moved this session
        self.session_start = time.time()
        # Recording
        self.rec_label = ""
        self.rec_idx = self.rec_count = 0
        self.rec_total = 0
        self.rec_done_base = 0
        self.rec_xfer = 0
        self.rec_start = time.time()
        # File
        self.file_name = ""
        self.file_idx = self.file_count = 0
        self.file_total = 0
        self.file_done = 0
        self.file_xfer = 0
        self.file_start = time.time()
        self._lines = 0
        self._last = 0.0

    # -- recording / file lifecycle ------------------------------------- #
    def start_recording(self, label, idx, count, rec_total):
        self.rec_label, self.rec_idx, self.rec_count = label, idx, count
        self.rec_total = rec_total
        self.rec_done_base = 0
        self.rec_xfer = 0
        self.rec_start = time.time()

    def start_file(self, name, idx, count, file_total, file_done):
        self.file_name, self.file_idx, self.file_count = name, idx, count
        self.file_total = file_total
        self.file_done = file_done   # already on disk (resume offset)
        self.file_xfer = 0
        self.file_start = time.time()
        self.render(force=True)

    def add_bytes(self, delta, file_done_abs):
        """Account *delta* freshly transferred bytes; file is at *file_done_abs*."""
        self.file_done = file_done_abs
        self.session_xfer += delta
        self.rec_xfer += delta
        self.file_xfer += delta
        self.render()

    def finish_file(self, file_total):
        self.file_done = file_total
        self.render(force=True)        # show the finished file at 100%
        self.done_base += file_total
        self.rec_done_base += file_total
        self.file_done = 0             # rolled into the base; avoid double count

    def skip_file(self, file_total):
        """A file already complete on disk -- counts as done, no transfer."""
        self.done_base += file_total
        self.rec_done_base += file_total

    # -- rendering ------------------------------------------------------- #
    def render(self, force=False):
        if not self.enabled:
            return
        now = time.time()
        if not force and now - self._last < 0.1:
            return
        self._last = now

        sess_speed = self.session_xfer / max(now - self.session_start, 1e-6)
        rec_speed = self.rec_xfer / max(now - self.rec_start, 1e-6)
        file_speed = self.file_xfer / max(now - self.file_start, 1e-6)

        total_done = self.done_base + self.file_done
        rec_done = self.rec_done_base + self.file_done

        file_eta = ((self.file_total - self.file_done) / file_speed
                    if file_speed > 0 else None)
        rec_eta = ((self.rec_total - rec_done) / rec_speed
                   if rec_speed > 0 else None)
        all_remaining = self.to_transfer - self.session_xfer
        all_eta = all_remaining / sess_speed if sess_speed > 0 else None

        file_frac = self.file_done / self.file_total if self.file_total else 1.0
        lines = [
            f"Recording {self.rec_idx}/{self.rec_count}  {self.rec_label}  "
            f"(file {self.file_idx}/{self.file_count})",
            f"  {shorten(self.file_name, 60)}",
            f"  {progress_bar(file_frac)} {file_frac*100:5.1f}%  "
            f"{human_bytes(self.file_done)}/{human_bytes(self.file_total)}  "
            f"{human_bytes(file_speed)}/s  ETA {fmt_duration(file_eta)}",
            f"  recording : {human_bytes(rec_done)}/{human_bytes(self.rec_total)}"
            f"  ETA {fmt_duration(rec_eta)}",
            f"  all {self.rec_count} recs: {human_bytes(total_done)}/"
            f"{human_bytes(self.total_bytes)}  {human_bytes(sess_speed)}/s"
            f"  ETA {fmt_duration(all_eta)}",
        ]
        self._emit(lines)

    def _emit(self, lines):
        if self.tty:
            out = []
            if self._lines:
                out.append(f"\033[{self._lines}A")
            for ln in lines:
                out.append("\033[2K" + ln + "\n")
            sys.stdout.write("".join(out))
            sys.stdout.flush()
            self._lines = len(lines)
        else:
            # Non-interactive: occasional one-line summaries only.
            now = time.time()
            if now - getattr(self, "_plast", 0) >= 5.0:
                self._plast = now
                total_done = self.done_base + self.file_done
                print(f"  ... {self.rec_label} file {self.file_idx}/"
                      f"{self.file_count}; overall "
                      f"{human_bytes(total_done)}/{human_bytes(self.total_bytes)}")

    def stop(self):
        """Leave the current block on screen and resume normal printing."""
        if self.tty and self._lines:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self._lines = 0


# --------------------------------------------------------------------------- #
# Downloading                                                                 #
# --------------------------------------------------------------------------- #

class DownloadError(Exception):
    pass


def download_file(url, dest, expected_size, on_progress, retries, timeout):
    """Download *url* to *dest* with resume + retries.

    ``on_progress(delta, file_done_abs)`` is called as bytes arrive.  Returns
    'exists' if already complete, otherwise 'downloaded'.  Raises DownloadError
    on permanent failure.
    """
    part = dest + ".part"
    if os.path.isfile(dest) and os.path.getsize(dest) == expected_size:
        return "exists"
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    resume = os.path.getsize(part) if os.path.isfile(part) else 0
    if resume > expected_size:
        os.remove(part)
        resume = 0

    attempt = 0
    while True:
        attempt += 1
        try:
            headers = {"User-Agent": USER_AGENT}
            if resume:
                headers["Range"] = f"bytes={resume}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                ranged = resp.status == 206
                if resume and not ranged:
                    resume = 0  # server ignored the range -> restart cleanly
                mode = "ab" if resume else "wb"
                on_progress(0, resume)
                written = resume
                with open(part, mode) as fh:
                    while True:
                        chunk = resp.read(DOWNLOAD_CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
                        written += len(chunk)
                        on_progress(len(chunk), written)
            actual = os.path.getsize(part)
            if actual != expected_size:
                # Could be a truncated transfer; retry/resume.
                raise DownloadError(
                    f"size mismatch (got {actual}, expected {expected_size})")
            os.replace(part, dest)
            return "downloaded"
        except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout,
                ssl.SSLError, ConnectionError, DownloadError, OSError) as exc:
            if attempt > retries:
                raise DownloadError(f"{type(exc).__name__}: {exc}")
            resume = os.path.getsize(part) if os.path.isfile(part) else 0
            wait = min(2 ** attempt, 30)
            # Drop below the live block (if any) to print the warning.
            sys.stdout.write("\n")
            print(f"    ! {os.path.basename(dest)}: {exc}; retry "
                  f"{attempt}/{retries} in {wait}s", file=sys.stderr)
            time.sleep(wait)


def verify_recording(rec_dir, manifest, display_enabled):
    """Re-hash every file against the manifest. Returns list of failures."""
    algo = manifest["hash_algorithm"]
    files = manifest["files"]
    failures = []
    tty = display_enabled and sys.stdout.isatty()
    total = sum(e["size"] for e in files)
    done = 0
    start = time.time()
    for i, entry in enumerate(files, 1):
        relpath = entry["path"]
        dest = os.path.join(rec_dir, relpath.replace("/", os.sep))
        line = f"  [verify {i}/{len(files)}] {shorten(relpath, 50)}"
        if tty:
            sys.stdout.write("\r\033[2K" + line)
            sys.stdout.flush()
        if not os.path.isfile(dest):
            failures.append((relpath, "missing"))
            done += entry["size"]
            continue
        if os.path.getsize(dest) != entry["size"]:
            failures.append((relpath, "wrong size"))
            done += entry["size"]
            continue
        if entry["hash"] is None:
            done += entry["size"]
            continue
        h = hashlib.new(algo)
        try:
            with open(dest, "rb") as fh:
                while True:
                    chunk = fh.read(READ_CHUNK)
                    if not chunk:
                        break
                    h.update(chunk)
                    done += len(chunk)
                    if tty and time.time() - start > 0.1:
                        speed = done / max(time.time() - start, 1e-6)
                        sys.stdout.write(
                            "\r\033[2K" + line +
                            f"  {human_bytes(done)}/{human_bytes(total)}"
                            f"  {human_bytes(speed)}/s")
                        sys.stdout.flush()
        except OSError as exc:
            failures.append((relpath, f"read error: {exc}"))
            continue
        if h.hexdigest() != entry["hash"]:
            failures.append((relpath, "hash mismatch"))
    if tty:
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()
    return failures


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #

def confirm(prompt):
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def download_recording(man, target, display, args):
    rid = man["recording_id"]
    rec_dir = os.path.join(target, man["dest_subpath"])
    os.makedirs(rec_dir, exist_ok=True)

    # Persist the manifest locally for reference / later verification.
    try:
        with open(os.path.join(rec_dir, MANIFEST_NAME), "wb") as fh:
            fh.write(man["raw"])
    except OSError as exc:
        print(f"  ! could not save manifest for {rid}: {exc}", file=sys.stderr)

    files = man["files"]
    rec_total = man["total_size_bytes"]
    display.start_recording(rid, display.rec_idx + 1, display.rec_count,
                            rec_total)

    for fidx, entry in enumerate(files, 1):
        relpath = entry["path"]
        size = entry["size"]
        dest = os.path.join(rec_dir, relpath.replace("/", os.sep))
        url = build_url(args.base_url, man["url_id"], relpath)

        if os.path.isfile(dest) and os.path.getsize(dest) == size:
            display.skip_file(size)
            continue

        resume0 = os.path.getsize(dest + ".part") if os.path.isfile(dest + ".part") else 0
        display.start_file(relpath, fidx, len(files), size, min(resume0, size))

        def cb(delta, file_done_abs):
            display.add_bytes(delta, file_done_abs)

        try:
            download_file(url, dest, size, cb, args.retries, args.timeout)
        except DownloadError as exc:
            display.stop()
            print(f"  ! FAILED {rid}/{relpath}: {exc}", file=sys.stderr)
            return False
        display.finish_file(size)

    display.stop()
    return True


def run(args):
    target = os.path.abspath(args.target)
    if args.ids_file is None:
        ids = baked_default_ids()
        ids_source = "built-in default list (test split first, then train)"
    elif args.ids_file.lower() == "train":
        ids = list(TRAIN_IDS)
        ids_source = "built-in train split"
    elif args.ids_file.lower() == "test":
        ids = list(TEST_IDS)
        ids_source = "built-in test split"
    else:
        ids = read_id_list(args.ids_file)
        ids_source = args.ids_file
    if not ids:
        print("error: no valid recording IDs in the input file", file=sys.stderr)
        return 2

    print("CoMind v1 dataset downloader")
    print(f"  source : {args.base_url}")
    print(f"  target : {target}")
    print(f"  ids    : {ids_source}")
    print(f"  records: {len(ids)}\n")

    manifests, failures = fetch_all_manifests(args.base_url, ids, args.timeout)

    # Fetch the meshes / annotations common assets up front, so they can be
    # offered as selectable entries in the parts list and -- when selected --
    # downloaded *before* any recordings.  --no-meshes / --no-annotations skip
    # them entirely (they are then neither offered nor downloaded).
    aux_mans = []
    if not args.no_meshes:
        m = fetch_aux_manifest(args.base_url, MESHES_SUBDIR, args.timeout)
        if m:
            aux_mans.append(m)
    if not args.no_annotations:
        a = fetch_aux_manifest(args.base_url, ANNOTATIONS_SUBDIR, args.timeout)
        if a:
            aux_mans.append(a)

    # Parts selection: the per-recording parts (vrs, mp4, ...) plus the meshes /
    # annotations common assets, each offered as its own selectable entry.
    rows = part_breakdown(manifests) + aux_rows(aux_mans)
    if args.list_parts:
        if not rows:
            print("\nNothing available to list parts for.", file=sys.stderr)
            return 1
        print_part_table(rows)
        return 0
    if not rows:
        print("\nNothing to download (no recordings matched and no "
              "meshes/annotations).", file=sys.stderr)
        return 1
    selected = resolve_part_selection(args, rows)
    if selected is None:
        return 2

    manifests, dropped = apply_part_selection(manifests, selected)
    aux_mans = [m for m in aux_mans if m["recording_id"] in selected]

    # Meshes / annotations are downloaded before any recordings.
    download_list = aux_mans + list(manifests)
    if not download_list:
        print("\nNo files match the selected parts; nothing to download.")
        return 0

    print(f"\nSelected parts: "
          f"{', '.join(k for k in (PART_KEYS + AUX_KEYS) if k in selected)}")
    if dropped:
        print(f"  ({dropped} recording(s) have none of the selected parts "
              f"-- skipped)")

    total_bytes = sum(m["total_size_bytes"] for m in download_list)
    aux_suffix = (" + " + " + ".join(m["recording_id"] for m in aux_mans)
                  if aux_mans else "")
    print(f"\nResolved {len(manifests)} recording(s){aux_suffix} "
          f"({len(failures)} unavailable).")
    print(f"Total dataset size to download: {human_bytes(total_bytes)}")
    for m in download_list:
        print(f"  - {m['recording_id']}: {len(m['files'])} files, "
              f"{human_bytes(m['total_size_bytes'])}")

    # How much is already on disk (resume-aware estimate).
    present = bytes_already_present(target, download_list)
    to_transfer = max(total_bytes - present, 0)
    if present:
        print(f"Already present locally: {human_bytes(present)} "
              f"(remaining to download ~ {human_bytes(to_transfer)})")

    # Free-space check (best effort, never requires sudo).
    free = free_space_bytes(target)
    if free is None:
        print("Free space: could not be determined for the target "
              "filesystem; proceeding without the space check.")
    else:
        margin = to_transfer + (1 << 30)  # +1 GiB headroom
        verdict = "OK" if free >= margin else "WARNING: may be insufficient"
        print(f"Free space at target: {human_bytes(free)} "
              f"(need ~ {human_bytes(to_transfer)}) -- {verdict}")

    if not args.yes:
        if not confirm("\nProceed with download? [y/N] "):
            print("Aborted.")
            return 0
    print()

    try:
        os.makedirs(target, exist_ok=True)
    except OSError as exc:
        print(f"error: cannot create target directory {target}: {exc}",
              file=sys.stderr)
        return 2

    # Write the built-in train/test split lists alongside the download.
    try:
        written = write_split_files(target)
        rels = ", ".join(os.path.relpath(p, target) for p in written)
        print(f"Wrote split lists: {rels}")
    except OSError as exc:
        print(f"  ! could not write split files: {exc}", file=sys.stderr)

    display = Display(total_bytes, to_transfer, enabled=not args.no_progress)
    display.rec_count = len(download_list)
    display.rec_idx = 0

    n_ok = n_failed = n_verify_failed = 0
    job_start = time.time()
    for man in download_list:
        rid = man["recording_id"]
        ok = download_recording(man, target, display, args)
        if not ok:
            n_failed += 1
            continue

        if args.skip_verification:
            print(f"  {rid}: downloaded ({human_bytes(man['total_size_bytes'])}), "
                  f"verification skipped.")
            n_ok += 1
            continue

        print(f"  {rid}: downloaded; verifying {len(man['files'])} files ...")
        fails = verify_recording(os.path.join(target, man["dest_subpath"]), man,
                                 not args.no_progress)
        if fails:
            n_verify_failed += 1
            print(f"  {rid}: VERIFICATION FAILED for {len(fails)} file(s):")
            for relpath, why in fails[:20]:
                print(f"      - {relpath}: {why}")
            if len(fails) > 20:
                print(f"      ... and {len(fails) - 20} more")
        else:
            n_ok += 1
            print(f"  {rid}: OK -- {len(man['files'])} files verified.")

    print(f"\nFinished in {fmt_duration(time.time() - job_start)}.")
    print(f"  downloaded ok           : {n_ok}")
    if n_failed:
        print(f"  download failures       : {n_failed}")
    if n_verify_failed:
        print(f"  verification failures   : {n_verify_failed}")
    if failures:
        print(f"  manifests unavailable   : {len(failures)}")
        for rid, why in failures:
            print(f"      - {rid}: {why}")
    return 0 if (n_failed == 0 and n_verify_failed == 0) else 1


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                     argparse.RawDescriptionHelpFormatter):
    """Show argument defaults *and* keep the epilog's hand-formatted layout."""


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="comind_download.py",
        description=(
            "Download CoMind v1 recordings (ETH Zurich, https://comind.ethz.ch).\n"
            "\n"
            "Recordings go to <target>/recordings/<RECORDING_ID>/; the meshes\n"
            "and annotations common assets go to <target>/meshes/ and\n"
            "<target>/annotations/.  Each download is driven by the server's\n"
            "healthcheck.json manifest and verified (re-hashed) afterwards.\n"
            "Interrupted downloads resume on re-run; complete files are skipped."
        ),
        epilog=(
            "which recordings:\n"
            "  (no IDS)        download the built-in default list (test, then train)\n"
            "  train | test    download only that built-in split\n"
            "  FILE            download the UUIDs listed in FILE (one per line)\n"
            "\n"
            "parts (recording parts + meshes/annotations common assets):\n"
            "  keys: " + ", ".join(PART_KEYS + AUX_KEYS) + "\n"
            "  --parts vrs,mp4              only those parts (no meshes/annotations)\n"
            "  --parts all,-gopro          everything except GoPro\n"
            "  --parts meshes,annotations  only the common assets\n"
            "  (interactive prompt by default; --list-parts to just list them)\n"
            "\n"
            "examples:\n"
            "  python3 comind_download.py                      # default list, ask parts\n"
            "  python3 comind_download.py train --yes          # train split, all parts\n"
            "  python3 comind_download.py test --parts vrs,mp4\n"
            "  python3 comind_download.py ids.txt --target /data/comind/v1\n"
            "  python3 comind_download.py --list-parts\n"
            "  python3 comind_download.py --no-meshes --no-annotations\n"
        ),
        formatter_class=_HelpFormatter,
    )
    p.add_argument("ids_file", nargs="?", metavar="IDS",
                   help="text file with one recording ID (UUID) per line, or "
                        "the literal 'train' / 'test' to download only that "
                        "built-in split; if omitted, the built-in default list "
                        "(train + test) is used")
    p.add_argument("--target", default=DEFAULT_TARGET,
                   help="directory to download into (recordings under "
                        "recordings/<RECORDING_ID>/, meshes under meshes/)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help="base URL serving <RECORDING_ID>/<files>")
    p.add_argument("--parts", default=None, metavar="LIST",
                   help="comma-separated parts to download, e.g. 'vrs,mp4' or "
                        "'all,-gopro'. Keys: " + ", ".join(PART_KEYS + AUX_KEYS) +
                        " (meshes/annotations are the common assets). Default: ask "
                        "interactively, or all when non-interactive")
    p.add_argument("--list-parts", action="store_true",
                   help="list the parts available for the requested recordings "
                        "and exit")
    p.add_argument("--no-meshes", action="store_true",
                   help="do not download the meshes common asset "
                        "(<base-url>/meshes/ into <target>/meshes/)")
    p.add_argument("--no-annotations", action="store_true",
                   help="do not download the annotations common asset "
                        "(<base-url>/annotations/ into <target>/annotations/)")
    p.add_argument("--skip-verification", action="store_true",
                   help="do not re-hash files after each recording downloads")
    p.add_argument("-y", "--yes", action="store_true",
                   help="do not ask for confirmation before downloading")
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                   help="network retries per file before giving up")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                   help="per-read network timeout in seconds")
    p.add_argument("--no-progress", action="store_true",
                   help="disable the live progress display")
    return p.parse_args(argv)


def read_version_tag(max_lines=40):
    """Return the 'Version: <tag>' value from this script's own header, if any.

    Reads the first *max_lines* lines of the script file (so a version tag in
    the module docstring is picked up) and returns the tag from the first line
    matching ``Version: <tag>``.  Returns None if no such line is found or the
    file cannot be read.
    """
    try:
        with open(os.path.abspath(__file__), "r", encoding="utf-8") as fh:
            for _ in range(max_lines):
                line = fh.readline()
                if not line:
                    break
                m = _VERSION_RE.match(line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return None


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    version = read_version_tag()
    if version:
        print(f"comind_download.py  version {version}")
    if (args.ids_file and args.ids_file.lower() not in ("train", "test")
            and not os.path.isfile(args.ids_file)):
        print(f"error: ids file not found: {args.ids_file}", file=sys.stderr)
        return 2
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\ninterrupted by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

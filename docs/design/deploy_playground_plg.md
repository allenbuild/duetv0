# Deploying the playground at plg.duetlabs.co

Status (2026-09-26): the static playground viewer is served at https://plg.duetlabs.co and https://duet-playground.vercel.app. The Vercel project is `duet-playground`, in a teammate's personal Vercel scope. TLS is issued, and video Range requests were confirmed on 2026-09-25. The live build is still the export made at `516ddf3`, before the review fixes: it serves full-length clips, has no robots.txt or noindex, and sends Vercel's default `access-control-allow-origin: *`. The next `scripts/deploy_playground.sh` replaces it with the export described below.

## A. Static site (live)

### What is published

Only episodes listed in `scripts/playground_publish.json` are published. Each entry must record `license`, `consent_basis`, `attribution` and `notes`; the page footer shows `attribution` and `license`, and the rest stays internal. An entry is published if its license is known, or if it has an explicit `approved` record (`{"by", "date": "YYYY-MM-DD", "note"}`), which prints a warning. An UNKNOWN license without approval is refused unless you pass `--allow-unknown-license`. Allowlist keys must be valid episode names. `--episodes a,b` must name a subset of the list.

| episode | license | consent basis | published because |
|---|---|---|---|
| `eidon_10004` | CC-BY-4.0 (`eidon-ai/tracker-pov`; the `tracker-pov-imu` license is not recorded) | UNKNOWN | known license (also owner-approved 2026-09-26) |
| `comind_43276420_clip` | UNKNOWN | UNKNOWN | owner-approved 2026-09-26 |

Both approvals read "kept live by owner decision; license/consent terms not yet recorded". The CoMind clip shows both participants' faces, a tattoo and a private home interior (review P0 #1). Getting written terms from CoMind is still open; the contact is in `docs/design/comind_sync_forensics.md`.

Per episode, the exporter publishes:

- **Videos.** Every usable (aligned) stream's video, trimmed to the analysed common window +-0.5 s, at most 640 px wide, H.264 at CRF 30. Clips carry no audio, subtitles, data tracks or metadata, are verified with ffprobe, and are named `streams/<stream>-<12-hex content key>.mp4`.
- **Trim.** The trim is an absolute coarse seek 5 s early followed by an exact output-side seek. `trim_start_s` is measured, not assumed: the export fails when the clip's first frame is more than half a frame off the first source frame after the trim point. This makes MPEG-TS, offset-MKV and offset-MP4 sources come out right. The encode version is part of the cache key, so every clip re-encodes on the first export with this code.
- **Data.** The overlay, body3d, world3d and IMU JSON, built by the server's own payload builders, from fresh stages only.
- **`episode.json`.** Stage states only (no details, logs or paths), each stream's `trim_start_s`, and the license and attribution. A stale stage is published as "stale", without its data, and noted as "not published (stale results; re-run the pipeline)". For comind_43276420_clip, world3d is stale. The QC report and notes are included, with absolute paths replaced by `<path>`.
- **UI files.** `index.html` and `app.js` are the server UI with its SERVER-ONLY blocks removed, and `index.html` loads `app.js?v=<content hash>`. The exporter checks that no server API reference and no upload form remain, and that three.js keeps its SRI hash.

At the site root it writes:

- `episodes.json`.
- `version.json`: `git_commit`, `git_dirty` (the playground sources differ from HEAD, untracked files included), `exported_at`, `episodes`, `exporter`. The page footer shows the commit, plus "-dirty" when the sources differ.
- `.duet-playground-export`: the exporter's marker file (see Deploy).
- `robots.txt` (`Disallow: /`).
- `vercel.json`, with these headers:

| path | headers |
|---|---|
| all | `Access-Control-Allow-Origin: https://plg.duetlabs.co` (`--site-origin` to change); CSP (`default-src 'self'`, scripts from self and cdnjs, `connect-src 'self'`, `form-action 'none'`, `frame-ancestors 'none'`); `X-Frame-Options: DENY`; `X-Content-Type-Options: nosniff`; `Referrer-Policy: no-referrer`; `X-Robots-Tag: noindex, nofollow, noarchive`; `Permissions-Policy` (camera, microphone, geolocation off) |
| `/`, `*.html`, `*.js`, `*.json`, `*.txt` | `Cache-Control: no-cache` |
| `*.mp4` | `Cache-Control: public, max-age=31536000, immutable` (file names carry the content key) |

The pinned origin is meant to replace Vercel's default `*`. Whether Vercel's own header gives way to it stays unconfirmed until the next deploy (see Verify).

### Deploy

```bash
scripts/deploy_playground.sh --adopt-old-export   # once, on the deploy machine (see step 2)
scripts/deploy_playground.sh [--allow-partial]    # PYTHON=/path/to/python overrides .venv/bin/python
```

1. **Link check.** The script aborts unless `data/playground/static_export/.vercel/project.json` links the existing `duet-playground` project. Without that link, `vercel deploy` would silently create a new project in whatever scope is logged in. The script also needs the `vercel` CLI, and the exporter needs the playground extra. On another machine, first run `cd data/playground/static_export && vercel link` (this needs access to the project's scope). The script passes its options through to the exporter.
2. **Export.** It runs `scripts/playground_export_static.py --out data/playground/static_export`.
   - The site is built in a fresh directory and swapped in only when every requested episode has exported, so removed episodes never linger; `.vercel/` is carried over.
   - The exporter replaces a non-empty `--out` only if it holds the marker file `.duet-playground-export`.
   - The existing `static_export` on the deploy machine was written by the previous exporter, which wrote no marker. So the first deploy needs `--adopt-old-export`, which replaces an old site only if it holds nothing but the old site layout (and `.vercel/`).
   - Deletes are confined to the export's own folders; a symlink pointing elsewhere is refused.
   - Clips are cached in `static_export.cache/`, keyed by source path, size, mtime and encode settings; cache entries the new site does not use are deleted.
3. The exporter's exit codes are 0 (site updated), 3 (an episode failed; the site is unchanged, and each failure is printed with its reason), 1 (configuration error) and 2 (usage error). On any non-zero exit, the script prints "nothing deployed" and exits 1 without calling Vercel.
4. Finally, the script runs `vercel deploy --prod --yes` from the export folder.

Missing sources stop a default deploy. If an allowlisted episode's source video is missing, preflight fails with `stream <s>: source video ... missing (dangling symlink -> ...)`; the exporter exits 3 and the live site stays as it is. This happens on any machine other than the one that created the committed episodes: their stream links are absolute paths on that machine, and the Eidon source video is not in the repo at all (the CoMind clips are, under `data/playground/clips/`). `--allow-partial` deploys the episodes that did export, and the failed ones disappear from the live site, because the new site replaces the old one as a whole. If no episode exports, nothing is deployed.

### Verify after a deploy

```bash
curl -sI https://plg.duetlabs.co/ | grep -iE 'access-control|content-security|x-robots|cache-control'
curl -s https://plg.duetlabs.co/version.json
curl -sI -H "Range: bytes=0-99" https://plg.duetlabs.co/<episode>/streams/<clip>.mp4 | head -3    # expect 206 + content-range
```

The first command should show `access-control-allow-origin: https://plg.duetlabs.co`; a `*` means Vercel's default still wins. Keep the export to a few hundred MB, and move larger episode sets to S3/CloudFront with the same folder layout.

### Domain (done 2026-09-25)

`duetlabs.co` belongs to a different Vercel account, so the subdomain had to prove control of the apex with a TXT record. It was verified through the API (`POST /v9/projects/{id}/domains/plg.duetlabs.co/verify`), because CLI 48 has no `domains verify`.

| type | host | value |
|---|---|---|
| TXT | `_vercel` | `vc-domain-verify=plg.duetlabs.co,5605f4c7c1b28ff43412` |
| CNAME | `plg` | `cname.vercel-dns.com` |

The CNAME was saved as `chame.vercel-dns.com`, a typo that resolves only through Vercel's wildcard DNS; correct it at the domain's DNS host. Deploys depend on one personal Vercel scope. Moving the project into the Vercel team that owns `duetlabs.co` would remove that dependency, since any team member could then link and deploy, and only the CNAME would be needed.

## B. Full pipeline on a GPU host (not deployed)

FastAPI plus GPU perception cannot run on Vercel. Run the server on a GPU box instead (an AWS g5/g6 instance, or the NVIDIA desktop used for ZED capture), behind a TLS reverse proxy such as Caddy, and point `plg-run.duetlabs.co` at it:

```bash
sudo apt install ffmpeg libgl1        # libgl1: the GUI OpenCV flavours need it; never pip-uninstall one cv2 flavour
pip install -e '.[playground]' && python scripts/playground.py fetch-models   # needs CLIP once, for the vocabulary bake
export PLAYGROUND_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export PLAYGROUND_EPISODES=/data/playground/episodes
python scripts/playground.py serve --host 127.0.0.1 --port 8765              # the proxy forwards https -> 127.0.0.1:8765
```

- **Set `PLAYGROUND_TOKEN`.** Basic auth at the proxy is not a substitute. Without a token, the server answers only loopback requests and refuses proxied ones, since those carry X-Forwarded-For. With a token, access is default-deny on the route path: everything except the UI shell and `/api/session` needs the token or the session cookie that users get by signing in once in the UI. Every failed credential counts toward a per-IP limit (10 per minute, then 429). The CSRF checks (`docs/design/egoexo_playground.md`, Server and viewer) refuse the cross-site requests that a browser holding basic-auth credentials would otherwise send. Surrounding whitespace in the token (e.g. from a .env file) is stripped.
- **Bind 127.0.0.1 and let the proxy terminate TLS.** Do not bind 0.0.0.0 without a token; `serve` refuses to. With a token it will start on 0.0.0.0, but the token and video then travel in clear text unless TLS fronts that port. The UI uses relative URLs and access checks use the route path, so the app also works under a path prefix (root_path).
- **Uploads and jobs.** `/api/upload` creates the episode and queues the pipeline, and the episode appears in the viewer when done. Jobs run one at a time in subprocesses (`PLAYGROUND_MAX_JOBS`). The UI's Cancel button stops a job: SIGTERM first, then SIGKILL after `PLAYGROUND_KILL_GRACE_S` (default 10 s). A failed or interrupted job shows as a banner. Uploads are capped by `PLAYGROUND_MAX_UPLOAD_MB` (default 8192 MB per request) and refused when less than `PLAYGROUND_MIN_FREE_MB` (default 2048 MB) would remain free.
- **Device.** `--device auto` picks cuda:0 (fp16) on an NVIDIA host. Throughput there has not been measured.
- **Publishing.** Uploaded episodes reach the static site only if they are added to the allowlist.

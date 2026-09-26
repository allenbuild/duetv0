# Human QA review (v0)

`src/duet/playground/api_review.py` (auto-mounted by `server.py`), page `static/review.html` + `review.js`. Built 2026-09-26.

## Purpose

Spot-check the perception stages (body2d skeletons, hand landmarks, object boxes) on a deterministic random sample
of frames per stream and accept/reject the autolabel proposals, producing per-item-type and per-stream accuracy
numbers that can be quoted for an episode.

## API

* `GET /api/episode/{name}/review/sample?frac=0.02&seed=0`
  For every stream, `round(frac * n_frames)` (min 1) frames drawn without replacement with
  `numpy.random.default_rng([seed, stream_index])`, sorted, so the same query always returns the same items. Each
  frame item: `{"id": "<stream>:<frame_idx>", "kind": "frame", "stream", "role", "person", "frame_idx", "image":
  "/episodes/<ep>/derived/frames/<stream>/<file>.jpg", "img_w", "img_h", "has": {body2d, hands, objects},
  "body2d": [[17 x [x, y, conf]] per tracked person], "hands": [{"side": "left|right", "landmarks": [21 x [x, y]]}],
  "objects": [{"name", "box": [x0, y0, x1, y1], "conf"}]}` — NaN slots are dropped, remaining NaN -> null.
  `has` says whether that stream has the derived file at all (exo streams have no hands/objects), so the page can
  disable those verdicts. If `derived/autolabel/proposals.json` exists (a list of
  `{"event","t0","t1","confidence","giver","receiver","status"}`) each proposal is appended as
  `{"id": "proposal:<i>", "kind": "proposal", "stream": "proposals", "index": i, ...proposal}`. The response also
  carries `verdicts` (latest verdict per id and item) so the page resumes where the reviewer stopped.
* `POST /api/episode/{name}/review` with `{"id", "stream", "frame_idx", "item": "body2d|hands|objects|proposal",
  "verdict": "ok|wrong|unsure", "note"}` appends one JSON line (+ `ts`) to `derived/review/verdicts.jsonl`.
  The log is append-only; readers take the last line per (id, item), so re-voting is idempotent. A proposal
  verdict also rewrites `status` in `proposals.json` (`ok -> accepted`, `wrong -> rejected`, `unsure -> unsure`,
  plus `review_note`), written atomically via a temp file because another stage may be producing that file.
* `GET /api/episode/{name}/review/summary` -> `{"total", "by_item": {type: {n, ok, wrong, unsure, accuracy}},
  "by_stream": {...}, "by_stream_item": {stream: {type: ...}}, "proposals": {total, accepted, rejected, pending}}`,
  accuracy = ok / (ok + wrong), null when nothing has been judged.

## Page

`review.html?ep=<name>` (frac / seed inputs, "Load sample"): the sampled frame on a canvas with the COCO-17
skeleton (purple), 21-point hands (green left / red right) and object boxes with name and confidence (blue), drawn
with the same code as the viewer. Three verdict rows with keyboard shortcuts: body `1/2/3`, hands `q/w/e`,
objects `a/s/d` (ok / wrong / unsure); `space` = next unfinished frame, `backspace`/arrows move; an optional note
is attached to the next verdict. When all applicable item types of a frame are judged it advances automatically.
Progress bar = frames fully judged / frames sampled. The proposals panel lists each proposal with accept /
reject / unsure and its status; the summary table refreshes after every verdict.

## Verification (2026-09-26)

* `tests/test_playground_review.py`: sample on the real `comind_43276420_clip` (4 streams x 9 frames at 2 %,
  deterministic across calls, differs by seed, shapes and JSON-cleanliness, exo streams flagged without
  hands/objects); verdicts + latest-wins + proposal status + summary on a symlinked copy of the episode
  (frames/body2d/hands/objects linked, private `derived/review` and `derived/autolabel`) so test verdicts never
  touch the real review log; input validation (422 / 404).
* Headless: served with uvicorn on port 8797, curled `review.html`, `review.js`, both GET endpoints and a frame
  JPEG (all 200), then rendered both pages in the local browser pane: overlays drawn, buttons and progress
  visible, no console errors. No verdicts were posted against the real episode.

## Limits / next

* Sampling is per stream and uniform; a stratified sample (by QC score, by motion) would spend reviewer time
  better.
* Verdicts are per frame per type, not per detection; a "which person/hand is wrong" click-to-flag is the obvious
  extension and the JSONL format has room for it (`note` today).
* No reviewer identity; add a `reviewer` field when more than one person reviews.

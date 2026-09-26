# annotate stage: dense VLM annotation (v0)

`src/duet/playground/annotate.py`, API `api_annotate.py`, page `static/annotate.html` + `annotate.js`. Built 2026-09-26.

## What it does

Every `interval_s` (default 2.0 s) of the common timeline the stage builds ONE contact sheet: all streams at that
instant tiled in a grid at most 1280 px wide (2 columns up to 4 streams, 3 above), each tile labelled
`<stream> [ego|exo] person: <name>`, plus a `t = .. s (frame k)` footer. The sheet goes to
`derived/annotate/keyframes/<idx>.jpg` (idx = keyframe ordinal, zero-padded) and a vision-language model is asked
for strict JSON:

```json
{"persons": [{"label": "leader", "action": "stirring bowl", "holding": ["whisk"], "attending_to": "partner|<object>|elsewhere"}],
 "coordination": "none|request|handover|joint_work|waiting",
 "handover": {"occurring": false, "giver": null, "receiver": null, "object": null},
 "summary": "one sentence"}
```

The reply is parsed robustly (`extract_json`: fences, prose around the object, braces inside strings) and coerced
to the schema (`normalize`) so every record has the same keys. Per keyframe the record is cached as
`keyframes/<idx>.json` and skipped on re-runs; all records are merged into `derived/annotate/annotations.json`
(a list of `{"idx", "frame", "t", "image", ...schema..., "_meta": {backend, model, elapsed_s, tokens...}}`, `t` in
reference seconds = `common_start_s + frame / proc_fps`). `derived/annotate/log.json` records backend, model,
count, new/cached/errors, elapsed, per-keyframe seconds and total input/output tokens.

Cache rules: a cached record is reused unless it was produced by the `dry` backend and a real backend is now
available, or it carries an `error` (failed calls are retried, never cached).

## Backends

| backend | how | needs |
|---|---|---|
| `anthropic` | Anthropic SDK 1.8, one `messages.create` per keyframe, image as base64 JPEG + prompt, `max_tokens=1024`; default model `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| `claude_cli` | `claude -p "<prompt naming the absolute sheet path>" --output-format json --allowedTools Read [--model m]`; the `result` field is parsed, `usage`/`total_cost_usd`/`duration_api_ms` logged. `CLAUDECODE` is stripped from the child env so a nested session is not refused | a logged-in `claude` CLI |
| `dry` | no model call: schema with nulls and a `note` | nothing |

Selection: `DUET_ANNOTATE_BACKEND` env, else `rig.json` `{"annotate": {"backend": ..}}`, else an episode note
`annotate: backend=.. model=.. interval_s=.. max_keyframes=..`; when none is given, `anthropic` if the key is set,
else `claude_cli` if `claude -p "reply with exactly: ok" --output-format json` answers without `is_error`, else `dry`.
Other options: `DUET_ANNOTATE_MODEL`, `DUET_ANNOTATE_INTERVAL_S`, `DUET_ANNOTATE_MAX_KEYFRAMES` (stage default:
unlimited). `run_annotate(ep, call=fn)` takes any `(image, prompt, model) -> (text, meta)` function (tests).

CLI: `PYTHONPATH=src .venv/bin/python -m duet.playground.annotate <episode> [--backend ..] [--model ..] [--interval 2] [--max-keyframes N] [--force]`
(`--force` deletes the cached keyframe JSON first, not the sheets).

## API and page

`GET /api/episode/{name}/annotations` -> `{available, status, log, streams, items: [record + image_url]}`.
`annotate.html?ep=<name>`: timeline list (thumbnail, t, summary, per-person action/holding, coordination and
handover badges); click or j/k / arrows selects; the right panel shows the sheet at full width, the person table,
handover line and the raw JSON.

## Status on this machine (2026-09-26)

* `ANTHROPIC_API_KEY` is not set; no `ant` profile. `claude` CLI 2.1.220 is installed but `claude -p` returns
  `is_error: true, "Failed to authenticate: OAuth session expired and could not be refreshed"` (with a clean env:
  `"Not logged in · Please run /login"`), so auto-selection lands on `dry`.
* Ran on `comind_43276420_clip` with `dry`: 24 keyframes (466 frames / 20), sheets 1280x1000, 0.22 s total. No
  model-produced annotation exists yet; run `claude /login` or export a key and re-run the stage (dry placeholders
  are redone automatically).
* Tests: `tests/test_playground_annotate.py` (schedule, sheet, JSON extraction, mocked backend incl. cache /
  error retry / dry upgrade, options, backend selection, CLI envelope parsing with a fake subprocess, SDK request
  shape with a fake client, API).

## Limits / next

* One model call per keyframe, sequential. Anthropic path could be batched (Message Batches) for long episodes.
* Prompt labels people by tile label; the model is not given the previous keyframe, so actions are per-instant.
  Adding the previous summary as context is a one-line prompt change.
* Sheets downscale ego frames only to the tile width (640 px); fine for kitchen-scale objects, small items
  (spoon vs whisk) may be missed at 2 s spacing.

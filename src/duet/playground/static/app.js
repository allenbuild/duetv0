/* Duet EgoExo Playground: synced multi-view playback with overlays, 3D panel, quality table.

   Server mode talks to the FastAPI app (src/duet/playground/server.py). Static mode (the exported site) is switched on
   by <script id="playground-static" type="application/json"> in index.html and reads the pre-exported JSON next to
   the page. The server-only blocks (between the begin/end marker comments) are removed by scripts/playground_export_static.py.
   Text that comes from episodes/servers is only ever inserted with textContent / setAttribute, never innerHTML.

   Time: t = seconds since common_start_s on the reference clock. A stream's own time (C1) is
   t_stream = (t_ref - offset_s) / (1 + drift_ppm * 1e-6); its <video> time is t_stream - media start (server: the
   original file, whose first video frame is at video_start_s; static: a clip trimmed to start at trim_start_s).
   Processed frame k sits at t = k / proc_fps. */
"use strict";
const $ = (s) => document.querySelector(s);
const COCO = [[5,7],[7,9],[6,8],[8,10],[5,6],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,5],[0,6]];
const HAND = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
// MediaPipe pose (33), upper body only for the monocular 3D panel: shoulders 11/12, elbows 13/14, wrists 15/16, hips 23/24,
// nose 0. Legs (knees 25/26, ankles 27/28, feet 29-32) are left out: the monocular estimates are too noisy to be useful.
const MP33_UPPER = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[0,11],[0,12]];
const C = {left:"#57E39B", right:"#FF6B6B", partner:"#FFB36B", body:"#B58CFF", obj:"#3EA7FF", held:"#FFD447", accent:"#3FA58C", mute:"#8A9591"};
const STATES = new Set(["done", "skipped", "failed", "running", "interrupted", "stale", "queued"]);
const WIN = 1200, WHOLE_MAX = 3000;  // frames per lazily fetched window; episodes up to WHOLE_MAX frames load in one request
const STATIC = readStaticConfig();

let ep = null, views = {}, series = {}, playing = false, t = 0, master = null, raf = 0, three = null, gen = 0;
let pollTimer = 0, pollDelay = 2000, onUnauthorized = null, threeError = "";
const reported = new Set();

function readStaticConfig() {
  const node = document.getElementById("playground-static");
  if (node) { try { return JSON.parse(node.textContent); } catch (e) { console.error("bad static config", e); } }
  return window.PLAYGROUND_STATIC || null;
}

/* ---------------- URLs ---------------- */
const enc = encodeURIComponent;
let URLS = {  // static site layout, relative to the page (whole-episode files, no windows)
  windowed: false,
  episodes: () => "episodes.json",
  episode: (n) => `${enc(n)}/episode.json`,
  overlay: (n, s) => `${enc(n)}/overlay_${enc(s)}.json`,
  body3d: (n) => `${enc(n)}/body3d.json`,
  world3d: (n) => `${enc(n)}/world3d.json`,
  imu: (n, s) => `${enc(n)}/imu_${enc(s)}.json`,
  video: (n, s) => `${enc(n)}/${String(s.path).split("/").map(enc).join("/")}`,
};
/* SERVER-ONLY BEGIN */
const win = (w) => (w ? `?start=${w.start}&end=${w.end}` : "");
if (!STATIC) URLS = {  // relative to the page, so the UI also works behind a proxy root path (e.g. /pg/)
  windowed: true,
  episodes: () => "api/episodes",
  episode: (n) => `api/episode/${enc(n)}`,
  overlay: (n, s, w) => `api/episode/${enc(n)}/overlay/${enc(s)}${win(w)}`,
  body3d: (n, w) => `api/episode/${enc(n)}/body3d${win(w)}`,
  world3d: (n, w) => `api/episode/${enc(n)}/world3d${win(w)}`,
  imu: (n, s, w) => `api/episode/${enc(n)}/imu_arm/${enc(s)}${win(w)}`,
  video: (n, s) => `api/episode/${enc(n)}/video/${enc(s.name)}`,
  run: (n, force) => `api/episode/${enc(n)}/run?force=${force ? "true" : "false"}`,
  cancel: (n) => `api/episode/${enc(n)}/cancel`,
  upload: () => "api/upload",
  session: () => "api/session",
};
/* SERVER-ONLY END */

/* ---------------- helpers ---------------- */
class ApiError extends Error { constructor(status, message) { super(message); this.status = status; } }

async function api(url, opts = {}) {
  const o = { credentials: "same-origin", ...opts, headers: { ...(opts.headers || {}) } };
  if (o.method && o.method !== "GET") o.headers["X-Playground-Request"] = "1";  // CSRF header the server requires
  let r;
  try { r = await fetch(url, o); } catch (e) { throw new ApiError(0, e.name === "AbortError" ? "aborted" : `network error (${e.message})`); }
  const text = await r.text(); let body = null;
  try { body = text ? JSON.parse(text) : null; } catch (e) { body = undefined; }
  if (!r.ok) {
    if (r.status === 401 && onUnauthorized) onUnauthorized();
    const d = body && body.detail;
    const msg = typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => (x && x.msg) || JSON.stringify(x)).join("; ") : `HTTP ${r.status}${text ? ": " + text.slice(0, 120) : ""}`;
    throw new ApiError(r.status, msg);
  }
  if (body === undefined) throw new ApiError(r.status, "the server sent something that is not JSON");
  return body;
}

function el(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) { if (k === "class") e.className = v; else if (k === "text") e.textContent = v; else e.setAttribute(k, v); }
  e.append(...kids);  // strings become text nodes
  return e;
}
const num = (x) => typeof x === "number" && isFinite(x);
const fmt = (x, d) => (num(x) ? x.toFixed(d) : "–");
function say(msg, bad = false) { const m = $("#msg"); if (!m) return; m.textContent = msg || ""; m.className = bad ? "mono bad" : "mono"; }
const dur = () => (ep ? Math.max(0, (ep.common_end_s || 0) - (ep.common_start_s || 0)) : 0);
const fpsOk = () => !!ep && num(ep.proc_fps) && ep.proc_fps > 0;
const nFrames = () => (fpsOk() ? (num(ep.n_frames) ? ep.n_frames : Math.floor(dur() * ep.proc_fps + 1e-9)) : 0);
function frameIndex() { return fpsOk() && nFrames() > 0 ? Math.max(0, Math.min(nFrames() - 1, Math.round(t * ep.proc_fps))) : -1; }  // -1: nothing to draw

/* stream time (C1) and <video> time */
const streamTime = (s, tRef) => (tRef - (s.offset_s || 0)) / (1 + (s.drift_ppm || 0) * 1e-6);
const mediaStart = (s) => (num(s.trim_start_s) ? s.trim_start_s : -(s.video_start_s || 0));
const mediaTime = (s, tRef) => streamTime(s, tRef) - mediaStart(s);
const refTime = (s, ct) => (1 + (s.drift_ppm || 0) * 1e-6) * (ct + mediaStart(s)) + (s.offset_s || 0);

/* ---------------- lazily fetched per-frame series (windowed in server mode for long episodes) ---------------- */
function makeSeries(url) { return { my: gen, url, whole: !URLS.windowed || nFrames() <= WHOLE_MAX, wins: new Map(), inflight: new Map(), failed: new Map(), off: false, timer: 0 }; }
function seriesAt(sr, k) {
  if (!sr || sr.off || k < 0) return null;
  const w = sr.whole ? 0 : Math.floor(k / WIN), d = sr.wins.get(w);
  if (!d) { wantWin(sr, w); return null; }
  sr.wins.delete(w); sr.wins.set(w, d);  // LRU touch
  if (!sr.whole && k - w * WIN > WIN / 2) wantWin(sr, w + 1);  // prefetch the next window
  return { d, i: k - (d.start || 0) };
}
function seriesHas(sr) { return !!sr && !sr.off && sr.wins.size > 0; }
function wantWin(sr, w) {  // debounced: while the scrubber moves only the window it settles on is fetched
  if (sr.whole || !playing && !sr.wins.size) { fetchWin(sr, w); return; }  // first window / whole file / playback: at once
  clearTimeout(sr.timer); sr.timer = setTimeout(() => fetchWin(sr, w), playing ? 0 : 150);
}
function fetchWin(sr, w) {
  if (sr.off || sr.my !== gen || sr.wins.has(w) || sr.inflight.has(w) || (!sr.whole && w * WIN >= nFrames())) return;
  const failedAt = sr.failed.get(w); if (failedAt && performance.now() - failedAt < 10000) return;
  for (const [v, ctrl] of sr.inflight) if (Math.abs(v - w) > 1) { ctrl.abort(); sr.inflight.delete(v); }  // superseded by a seek
  const ctrl = new AbortController(); sr.inflight.set(w, ctrl);
  api(sr.url(sr.whole ? null : { start: w * WIN, end: (w + 1) * WIN }), { signal: ctrl.signal }).then((d) => {
    if (sr.my !== gen) return;  // a different episode was opened meanwhile
    if (!d || d.available === false) { sr.off = true; } else { sr.wins.set(w, d); while (sr.wins.size > 4) sr.wins.delete(sr.wins.keys().next().value); }
    setupThree(); draw();  // never throw (display errors are reported, not treated as load failures)
  }).catch((e) => {
    if (sr.my !== gen || ctrl.signal.aborted) return;
    if (e.status === 404) sr.off = true; else { sr.failed.set(w, performance.now()); say(`could not load data: ${e.message}`, true); }
  }).finally(() => { if (sr.inflight.get(w) === ctrl) sr.inflight.delete(w); });
}

/* ---------------- episodes ---------------- */
async function loadEpisodes(select) {
  let eps;
  try { eps = await api(URLS.episodes()); } catch (e) { say(`could not load the episode list: ${e.message}`, true); return; }
  const sel = $("#episode"); sel.replaceChildren();
  for (const e of eps) {
    const o = el("option", { value: e.name, text: e.error ? `${e.name} (${e.error})` : `${e.name}  (${e.streams.length} streams, ${fmt(e.duration_s, 0)} s)` });
    if (e.error) o.disabled = true;
    sel.append(o);
  }
  const ok = eps.filter((e) => !e.error), want = ok.some((e) => e.name === select) ? select : ok.length ? ok[0].name : null;
  if (want) { sel.value = want; await loadEpisode(want); } else { clearEpisode(); say(eps.length ? "no readable episodes" : "no episodes yet", eps.length > 0); }
}

function clearEpisode() {
  stop(); clearTimeout(pollTimer);
  for (const v of Object.values(views)) { v.video.removeAttribute("src"); v.video.load(); }  // stop downloads
  $("#egos").replaceChildren(); $("#exos").replaceChildren(); views = {}; master = null; series = {};
}

let loadReq = 0;
async function loadEpisode(name) {
  const req = ++loadReq; stop();
  let d;
  try { d = await api(URLS.episode(name)); }
  catch (e) {  // keep showing (and polling) the current episode; the selector goes back to it
    if (req !== loadReq) return;
    say(e.message.startsWith(`episode ${name}`) ? e.message : `episode ${name}: ${e.message}`, true);
    if (ep) { $("#episode").value = ep.name; if (ep.running) schedulePoll(pollDelay); }
    return;
  }
  if (req !== loadReq) return;
  ++gen; clearEpisode(); ep = d; ep.name = name; say(ep.problem ? `episode ${name}: ${ep.problem}` : "", !!ep.problem);
  buildViews(); renderStages(); renderNotes(); renderQC(); renderAttribution();
  $("#scrub").max = String(Math.max(1, Math.round(dur() * 100)));
  for (const s of ep.streams) series[`ov:${s.name}`] = makeSeries((w) => URLS.overlay(name, s.name, w));
  series.body3d = makeSeries((w) => URLS.body3d(name, w));
  series.world3d = makeSeries((w) => URLS.world3d(name, w));
  for (const s of ep.streams) if (s.imu) series[`imu:${s.name}`] = makeSeries((w) => URLS.imu(name, s.name, w));
  setupThree(); seek(0);
  if (ep.running) { pollDelay = 2000; schedulePoll(pollDelay); }
}

function buildViews() {
  for (const s of ep.streams) {
    const video = el("video", { playsinline: "", preload: "auto" }); video.muted = true; video.src = URLS.video(ep.name, s);
    const canvas = el("canvas"), status = el("span", { class: "warn" });
    const label = ` · ${s.role}${s.person ? " · " + s.person : ""} · offset ${num(s.offset_s) && s.offset_s >= 0 ? "+" : ""}${fmt(s.offset_s, 2)} s`;
    const tag = el("div", { class: "tag" }, el("b", { text: s.name }), label, s.usable === false ? el("span", { class: "warn", text: " · UNALIGNED (not synced)" }) : "", status);
    const view = views[s.name] = { s, video, canvas, failed: false };
    video.addEventListener("error", () => {  // the playback clock must not follow a video that cannot play
      status.textContent = " · video failed to load"; view.failed = true;
      if (master === view) master = Object.values(views).find((x) => !x.failed) || view;
    });
    video.addEventListener("loadedmetadata", draw);
    (s.role === "ego" ? $("#egos") : $("#exos")).append(el("div", { class: "view" }, video, canvas, tag));
    if (!master || s.name === ep.reference) master = view;
  }
}

function renderStages() {
  const div = $("#stages"); div.replaceChildren();
  for (const st of ep.stages || []) {
    const info = (ep.status || {})[st] || {}, why = (ep.stale || {})[st];  // "done" but out of date: not served
    const state = why && info.state === "done" ? "stale" : String(info.state || "pending");
    const chip = el("div", { class: `stage ${STATES.has(state) ? state : ""}` }, el("span", { class: "dot" }), el("span", { class: "mono", text: st }), el("span", { class: "state", text: state }));
    chip.title = why ? `out of date (${why}); re-run the pipeline` : typeof info.detail === "string" ? info.detail : "";  // plain text tooltip
    div.append(chip);
  }
  const job = ep.job, failed = !ep.running && job && (job.state === "failed" || job.state === "interrupted");
  $("#runstate").textContent = ep.running ? "queued / running…" : failed ? `last job ${job.state}: ${String(job.detail || "").slice(0, 160)}` : "";
  $("#runstate").title = failed && job.detail ? String(job.detail) : "";
  $("#runstate").className = failed ? "mono bad" : "mono mute";
  const cancel = $("#cancelRun"); if (cancel) cancel.hidden = !ep.running;  // server UI only
}
function renderNotes() {
  const lines = (ep.notes || []).map(String);
  lines.push(`reference: ${ep.reference} · window ${fmt(ep.common_start_s, 2)}–${fmt(ep.common_end_s, 2)} s · ${ep.proc_fps} fps processing`);
  $("#notes").textContent = lines.join("\n");
}
function renderQC() {
  const tb = $("#qc tbody"); tb.replaceChildren();
  const qc = ep.qc;
  if (!qc || typeof qc !== "object") { tb.append(el("tr", {}, el("td", { class: "mute", text: ep.qc_error || (STATIC ? "no quality report" : "run the pipeline") }))); return; }
  const trow = (k, text, why = "", cls = "num") => { const tr = el("tr", {}, el("td", { text: k }), el("td", { class: cls, text })); tr.title = why; tb.append(tr); };
  const pct = (x) => (x * 100).toFixed(0) + "%", f2 = (x) => x.toFixed(2), VERDICT = { pass: "num", flag: "num warn", reject: "num bad" };
  const verdict = (k, v, why) => { if (v) trow(k, String(v), why, VERDICT[v] || "num"); };
  for (const [name, q] of Object.entries(qc.streams || {})) {
    if (!q || typeof q !== "object") continue;
    const miss = q.missing || {}, row = (k, key, f, always = true) => { const v = q[key];  // null + a reason = not measurable, shown as such
      if (num(v)) trow(`${name} ${k}`, f(v)); else if (always || key in miss) trow(`${name} ${k}`, "–", miss[key] ? String(miss[key]) : ""); };
    verdict(`${name} verdict`, q.verdict, Object.entries(q.flags || {}).filter(([, f]) => f !== "pass").map(([m, f]) => `${m}: ${f}`).join("; "));
    row("good frames", "good_frame_percent", pct); row("hands present", "hand_presence_ratio", pct, false); row("person present", "person_presence_ratio", pct, false);
    row("stability", "stability_score", f2); row("lighting", "lighting_score", f2); row("sync confidence", "alignment_confidence", (x) => "×" + x.toFixed(1), false);
  }
  const e = qc.episode || {};
  verdict("episode verdict", e.verdict, (e.reasons || []).map(String).join("; "));
  if (e.verdict && e.verdict !== "pass") for (const r of e.reasons || []) tb.append(el("tr", {}, el("td", { colspan: "2", class: "mute", text: String(r) })));
  if ("both_visible_ratio" in e) trow("both people visible (exo)", num(e.both_visible_ratio) ? pct(e.both_visible_ratio) : "–");
  if ("hands_all_egos_ratio" in e) trow("hands in every ego view", num(e.hands_all_egos_ratio) ? pct(e.hands_all_egos_ratio) : "–");
}
function renderAttribution() {
  const f = $("#attribution"); if (!f) return;
  const parts = [];
  if (ep && (ep.attribution || ep.license)) parts.push(`Data (${ep.name}): ${[ep.attribution, ep.license].filter(Boolean).join(" · ")}`);
  if (STATIC && STATIC.notice) parts.push(String(STATIC.notice));
  if (STATIC && STATIC.version) parts.push(`build ${STATIC.version}`);
  f.textContent = parts.join("  |  "); f.hidden = parts.length === 0;
}

/* ---------------- playback ---------------- */
function seek(tt) {
  if (!ep) return;
  t = Math.max(0, Math.min(tt, dur()));
  for (const v of Object.values(views)) { const want = Math.max(0, mediaTime(v.s, ep.common_start_s + t)); if (Math.abs(v.video.currentTime - want) > 0.08) v.video.currentTime = want; v.video.playbackRate = 1; }
  updateLabel(); draw();
}
function updateLabel() { $("#tlabel").textContent = `${t.toFixed(2)} / ${dur().toFixed(2)} s`; $("#scrub").value = String(Math.round(t * 100)); }
function play() { if (!ep || !master) return; playing = true; $("#play").textContent = "Pause"; for (const v of Object.values(views)) { v.video.playbackRate = 1; v.video.play().catch(() => {}); } loop(); }
function stop() { playing = false; $("#play").textContent = "Play"; for (const v of Object.values(views)) { v.video.pause(); v.video.playbackRate = 1; } cancelAnimationFrame(raf); }
function syncView(v) {
  // small drift: nudge playbackRate (no stutter on long-GOP video); only a large error hard-seeks
  if (v.video.seeking || v.video.readyState < 2) return;
  const want = mediaTime(v.s, ep.common_start_s + t), d = v.video.currentTime - want;
  if (Math.abs(d) > 0.5) { v.video.currentTime = Math.max(0, want); v.video.playbackRate = 1; }
  else if (Math.abs(d) > 0.02) v.video.playbackRate = Math.min(1.1, Math.max(0.9, 1 - 0.8 * d));
  else if (v.video.playbackRate !== 1) v.video.playbackRate = 1;
}
function loop() {
  if (!playing) return;
  t = Math.max(0, refTime(master.s, master.video.currentTime) - ep.common_start_s);
  for (const v of Object.values(views)) if (v !== master) syncView(v);
  if (t >= dur()) stop();
  updateLabel(); draw(); raf = requestAnimationFrame(loop);
}

/* ---------------- overlays ---------------- */
function report(e) { const m = `display error: ${(e && e.message) || e}`; if (!reported.has(m)) { reported.add(m); console.error(e); say(m, true); } }
function draw() {
  if (!ep) return;
  const show = $("#showOverlays").checked, k = frameIndex(), dpr = window.devicePixelRatio || 1;
  for (const v of Object.values(views)) { try { drawView(v, show, k, dpr); } catch (e) { report(e); } }
  try { drawThree(); } catch (e) { report(e); }
}
function drawView(v, show, k, dpr) {
  const cv = v.canvas, W = cv.clientWidth, H = cv.clientHeight, bw = Math.round(W * dpr), bh = Math.round(H * dpr);
  if (cv.width !== bw || cv.height !== bh) { cv.width = bw; cv.height = bh; }
  const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  const got = show ? seriesAt(series[`ov:${v.s.name}`], k) : null; if (!got) return;
  const o = got.d, i = got.i; if (!o.img_w || !o.img_h) return;
  // letterboxed <video> (object-fit: contain): processed-frame px -> canvas css px
  const vw = v.video.videoWidth || o.img_w, vh = v.video.videoHeight || o.img_h, scale = Math.min(W / vw, H / vh);
  const ox = (W - vw * scale) / 2, oy = (H - vh * scale) / 2, sx = vw / o.img_w * scale, sy = vh / o.img_h * scale;
  const P = (x, y) => [ox + x * sx, oy + y * sy];
  ctx.lineWidth = 1.5; ctx.font = "10px 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  const hands = o.hands2d && o.hands2d[i], boxes = o.objects && o.objects[i];
  if (boxes) { const held = heldBoxes(boxes, hands), names = o.object_vocab && o.object_idx && o.object_idx[i];
    boxes.forEach((b, j) => { if (!b || b[4] == null) return; const [x0, y0] = P(b[0], b[1]), [x1, y1] = P(b[2], b[3]), h = held.has(j);
      ctx.strokeStyle = ctx.fillStyle = h ? C.held : C.obj; ctx.lineWidth = h ? 2 : 1; ctx.globalAlpha = h ? 1 : 0.6; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      const idx = names ? names[j] : -1; if (idx >= 0) ctx.fillText(String(o.object_vocab[idx]), x0 + 2, y0 - 2); ctx.globalAlpha = 1; }); }
  if (o.body2d && o.body2d[i]) for (const person of o.body2d[i]) { if (!person || person[0][0] == null) continue; ctx.strokeStyle = ctx.fillStyle = C.body; ctx.lineWidth = 1.5;
    for (const [a, b] of COCO) { if (person[a][2] > 0.45 && person[b][2] > 0.45) { const [x0, y0] = P(person[a][0], person[a][1]), [x1, y1] = P(person[b][0], person[b][1]); ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); } }
    for (const kp of person) if (kp[2] > 0.45) { const [x, y] = P(kp[0], kp[1]); ctx.beginPath(); ctx.arc(x, y, 2, 0, 7); ctx.fill(); } }
  drawHands(ctx, P, hands, (side) => (side === 0 ? C.left : C.right), false);
  drawHands(ctx, P, o.partner_hands2d && o.partner_hands2d[i], () => C.partner, true);  // hands v2: other people's hands in this view
}
function drawHands(ctx, P, hands, color, dashed) {
  if (!hands) return;
  ctx.setLineDash(dashed ? [4, 3] : []);
  hands.forEach((hand, side) => { if (!hand || hand[0][0] == null) return; ctx.strokeStyle = ctx.fillStyle = color(side); ctx.lineWidth = 1.6;
    for (const [a, b] of HAND) { const [x0, y0] = P(hand[a][0], hand[a][1]), [x1, y1] = P(hand[b][0], hand[b][1]); ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); }
    for (const lm of hand) { const [x, y] = P(lm[0], lm[1]); ctx.beginPath(); ctx.arc(x, y, 2, 0, 7); ctx.fill(); } });
  ctx.setLineDash([]);
}
function heldBoxes(boxes, hands) {
  const held = new Set(); if (!hands) return held;
  boxes.forEach((b, j) => { if (!b || b[4] == null) return; for (const hand of hands) { if (!hand || hand[0][0] == null) continue; let inside = 0;
    for (const lm of hand) if (lm[0] >= b[0] && lm[0] <= b[2] && lm[1] >= b[1] && lm[1] <= b[3]) inside++; if (inside >= 8) held.add(j); } });
  return held;
}

/* ---------------- 3D panel (three.js; objects are pooled and reused, never re-allocated per frame) ---------------- */
function setupThree() {
  if (!ep) return;
  if (!three && !threeError) {
    try {
      if (typeof THREE === "undefined") throw new Error("three.js did not load");
      const renderer = new THREE.WebGLRenderer({ canvas: $("#three"), antialias: true }), scene = new THREE.Scene(); scene.background = new THREE.Color(0x0b0f0e);
      const camera = new THREE.PerspectiveCamera(40, 4 / 3, 0.05, 20); camera.position.set(1.6, 1.6, 2.4); camera.lookAt(0, 0.9, 0);
      scene.add(new THREE.GridHelper(3, 12, 0x26302d, 0x1a2220)); scene.add(new THREE.AxesHelper(0.3));
      three = { renderer, scene, camera, lines: {}, meshes: {}, geoms: {}, mats: {} };
    } catch (e) { threeError = (e && e.message) || String(e); console.warn("3D panel disabled:", e); }  // e.g. no WebGL: the rest still works
  }
  const hasWorld = seriesHas(series.world3d), has3d = !hasWorld && seriesHas(series.body3d), imu = Object.keys(series).some((k) => k.startsWith("imu:") && seriesHas(series[k]));
  const w = hasWorld ? [...series.world3d.wins.values()][0] : null, b = has3d ? [...series.body3d.wins.values()][0] : null;
  $("#threeTitle").textContent = hasWorld ? "3D · world frame (board): triangulated bodies, heads, objects" : has3d ? `3D body · monocular from ${b.stream}` : imu ? "3D arms · IMU harness (Eidon 7-slot)" : "3D";
  $("#threeNote").textContent = threeError ? `3D unavailable: ${threeError}`
    : hasWorld ? `metres in the board frame. heads: ${Object.entries(w.headpose_backends || {}).map(([k, v]) => k + "=" + v).join(", ") || "none"}`
    : has3d ? "MediaPipe world landmarks: metres, hip-centred, single camera, camera-aligned (you look from the camera side). Not triangulated."
    : imu ? "Arm chain from sensor quaternions, chest-relative (ported from Eidon Sim)." : "run body3d / world3d or add IMU";
  const sync = Object.entries(series).filter(([key, sr]) => key.startsWith("imu:") && seriesHas(sr)).map(([key, sr]) => [key.slice(4), [...sr.wins.values()][0].imu_sync]).filter(([, v]) => v);
  if (sync.length && !threeError) $("#threeNote").textContent += ` IMU clock: ${sync.map(([n, v]) => `${n} ${v}${v === "unsynced" || v === "drift_suspected" || v === "legacy_unverified" ? " (NOT verified against the video)" : ""}`).join(", ")}.`;
  drawThree();
}
function lineSet(key, color) {
  let L = three.lines[key];
  if (!L) { L = { color, n: 0, cap: 0, buf: null, obj: new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color })) }; L.obj.frustumCulled = false; three.scene.add(L.obj); three.lines[key] = L; grow(L, 512); }
  else if (L.color !== color) { L.obj.material.color.set(color); L.color = color; }
  return L;
}
function grow(L, cap) {
  const buf = new Float32Array(cap * 3); if (L.buf) buf.set(L.buf.subarray(0, L.n * 3));
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(buf, 3).setUsage(THREE.DynamicDrawUsage));
  L.obj.geometry.dispose(); L.obj.geometry = g; L.buf = buf; L.cap = cap;
}
function seg(L, a, b) { if (!finite3(a) || !finite3(b)) return; if (L.n + 2 > L.cap) grow(L, L.cap * 2); L.buf.set(a, L.n * 3); L.buf.set(b, L.n * 3 + 3); L.n += 2; }
function segLines(key, color, pairs, pts) { const L = lineSet(key, color); for (const [a, b] of pairs) seg(L, pts[a], pts[b]); }
function mesh(kind, color, p) {  // pooled spheres / boxes / cones sharing one geometry per kind and one material per colour
  if (!p) return;
  const pool = three.meshes[kind] || (three.meshes[kind] = { used: 0, list: [] });
  if (!three.geoms[kind]) three.geoms[kind] = kind === "sphere" ? new THREE.SphereGeometry(0.06, 12, 12) : kind === "box" ? new THREE.BoxGeometry(0.05, 0.05, 0.05) : new THREE.ConeGeometry(0.04, 0.08, 8);
  const mat = three.mats[color] || (three.mats[color] = new THREE.MeshBasicMaterial({ color }));
  let m = pool.list[pool.used];
  if (!m) { m = new THREE.Mesh(three.geoms[kind], mat); pool.list.push(m); three.scene.add(m); }
  m.material = mat; m.position.set(p[0], p[1], p[2]); m.visible = true; pool.used++;
}
const finite3 = (p) => (p && p[0] != null && p[1] != null && p[2] != null ? p : null);
function nearestSample(ts, tt) {  // index of the sample closest to tt (ts sorted), -1 if empty
  if (!ts || !ts.length) return -1;
  let lo = 0, hi = ts.length - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (ts[mid] < tt) lo = mid + 1; else hi = mid; }
  return lo > 0 && Math.abs(ts[lo - 1] - tt) < Math.abs(ts[lo] - tt) ? lo - 1 : lo;
}
function drawThree() {
  if (!three || !ep) return;
  for (const L of Object.values(three.lines)) L.n = 0;
  for (const pool of Object.values(three.meshes)) pool.used = 0;
  const k = frameIndex(), noWorld = !series.world3d || series.world3d.off;
  const W3 = noWorld ? null : seriesAt(series.world3d, k), B3 = noWorld ? seriesAt(series.body3d, k) : null;
  if (W3) {
    // board frame: X right, Y down the board, Z into the board -> scene: x = X, up = -Z, z = Y (a proper rotation, det +1)
    const w = W3.d, i = W3.i, toScene = (p) => (finite3(p) ? [p[0], -p[2], p[1]] : null), egos = ep.streams.filter((s) => s.role === "ego").map((s) => s.name);
    const headColor = (name) => (egos.indexOf(name) === 0 ? 0x8fa3cb : 0xe39468);
    ((w.bodies && w.bodies[i]) || []).forEach((person, p) => { const who = w.body_person && w.body_person[p];  // slot linked to an ego wearer
      segLines(`body${p}`, who ? headColor(who) : 0xb58cff, COCO, person.map(toScene)); });
    for (const [name, a] of Object.entries(w.heads || {})) mesh("sphere", headColor(name), toScene(a[i]));
    for (const [name, a] of Object.entries(w.head_poses || {})) { const T = a[i]; if (!T || T[0][3] == null) continue;  // T_world_cam: camera z (column 2) looks forward
      const o = [T[0][3], T[1][3], T[2][3]], f = [o[0] + 0.2 * T[0][2], o[1] + 0.2 * T[1][2], o[2] + 0.2 * T[2][2]]; seg(lineSet(`gaze:${name}`, headColor(name)), toScene(o), toScene(f)); }
    for (const a of Object.values(w.objects || {})) mesh("box", 0xffd447, toScene(a[i]));
    for (const a of Object.values(w.hands3d || {})) for (const hand of a[i] || []) segLines("hands3d", 0x57e39b, HAND, hand.map(toScene));
    for (const T of Object.values(w.cameras || {})) mesh("cone", 0x8a9591, toScene([T[0][3], T[1][3], T[2][3]]));
  } else if (B3) {
    // MediaPipe world (camera-aligned): x right, y down, z AWAY from the camera -> scene (x, -y, -z): a rotation (det +1), not
    // a mirror; +0.9 m lifts the hip-centred skeleton above the grid
    for (const person of B3.d.world[B3.i] || []) { if (!person || person[0][0] == null) continue; segLines("body3d", 0xb58cff, MP33_UPPER, person.map((p) => (finite3(p) ? [p[0], -p[1] + 0.9, -p[2]] : null))); }
  }
  for (const [key, sr] of Object.entries(series)) {
    if (!key.startsWith("imu:")) continue;
    const got = seriesAt(sr, k); if (!got) continue;
    const a = got.d, j = nearestSample(a.t_s, ep.common_start_s + t);  // IMU samples are irregular and have gaps
    if (j < 0 || !(Math.abs(a.t_s[j] - (ep.common_start_s + t)) <= 0.75 * (a.period_s || 0.05))) continue;
    (a.points[j] || []).forEach((side, si) => { if (side[0][0] == null) return; segLines(`imu${si}`, si === 0 ? 0x57e39b : 0xff6b6b, [[0, 1], [1, 2], [2, 3]], side); });
  }
  for (const L of Object.values(three.lines)) { L.obj.geometry.getAttribute("position").needsUpdate = true; L.obj.geometry.setDrawRange(0, L.n); L.obj.visible = L.n > 0; }
  for (const pool of Object.values(three.meshes)) for (let j = pool.used; j < pool.list.length; j++) pool.list[j].visible = false;
  const cv = $("#three"), dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = Math.round(W * 0.75);
  if (W && (three.w !== W || three.dpr !== dpr)) { three.renderer.setPixelRatio(dpr); three.renderer.setSize(W, H, false); three.camera.aspect = W / H; three.camera.updateProjectionMatrix(); three.w = W; three.dpr = dpr; }
  three.renderer.render(three.scene, three.camera);
}

/* ---------------- controls ---------------- */
$("#episode").addEventListener("change", (e) => loadEpisode(e.target.value));
$("#play").addEventListener("click", () => (playing ? stop() : play()));
$("#scrub").addEventListener("input", (e) => { stop(); seek(Number(e.target.value) / 100); });
$("#showOverlays").addEventListener("change", draw);
window.addEventListener("resize", draw);
function schedulePoll(ms) { clearTimeout(pollTimer); pollTimer = setTimeout(pollStatus, ms); }  // one timer, never parallel chains
async function pollStatus() {
  if (STATIC || !ep) return;
  const name = ep.name, my = gen;
  let d;
  try { d = await api(URLS.episode(name)); }
  catch (e) { if (my !== gen) return; pollDelay = Math.min(pollDelay * 2, 30000); say(`status update failed (${e.message}); retrying in ${Math.round(pollDelay / 1000)} s`, true); schedulePoll(pollDelay); return; }
  if (my !== gen) return;
  const was = ep.running; pollDelay = 2000; say("");
  Object.assign(ep, { status: d.status, running: d.running, notes: d.notes, qc: d.qc, qc_error: d.qc_error, stages: d.stages, job: d.job, problem: d.problem, stale: d.stale });
  renderStages(); renderNotes(); renderQC();
  if (d.running) schedulePoll(pollDelay); else if (was) loadEpisode(name);  // finished: reload offsets, overlays, 3D
}
if (STATIC) document.querySelectorAll("[data-server-only]").forEach((n) => n.remove());

/* SERVER-ONLY BEGIN */
onUnauthorized = () => { const f = $("#login"); if (f && f.hidden) { f.hidden = false; say("this playground needs an access token", true); $("#token").focus(); } };
$("#login").addEventListener("submit", async (e) => {
  e.preventDefault(); const token = $("#token").value.trim(); if (!token) return;
  try { await api(URLS.session(), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ token }) }); }
  catch (err) { say(`sign-in failed: ${err.message}`, true); return; }
  $("#token").value = ""; $("#login").hidden = true; say(""); loadEpisodes(ep && ep.name);
});
async function runPipeline(force) {
  if (!ep) return;
  const name = ep.name, my = gen;
  try { await api(URLS.run(name, force), { method: "POST" }); }
  catch (e) { if (my === gen && e.status !== 409) say(`could not start the pipeline: ${e.message}`, true); if (e.status !== 409) return; }  // 409: already queued/running
  if (my !== gen) return;  // another episode was opened meanwhile
  ep.running = true; renderStages(); pollDelay = 2000; schedulePoll(500);
}
$("#runAll").addEventListener("click", () => runPipeline(false));
$("#rerun").addEventListener("click", () => runPipeline(true));
$("#cancelRun").addEventListener("click", async () => {
  if (!ep) return; const btn = $("#cancelRun"), name = ep.name, my = gen; btn.disabled = true;
  let msg = "cancel requested; running stages end as interrupted", bad = false;
  try { await api(URLS.cancel(name), { method: "POST" }); }
  catch (e) { msg = e.status === 409 ? "nothing to cancel (no job of this server is queued or running)" : `could not cancel: ${e.message}`; bad = e.status !== 409; }
  finally { btn.disabled = false; }
  if (my !== gen) return;  // another episode was opened meanwhile
  say(`${name}: ${msg}`, bad); pollDelay = 2000; schedulePoll(300);  // the single poll timer picks up the new state (and hides the button)
});
$("#upFiles").addEventListener("change", (e) => {
  const rows = $("#upRows"); rows.replaceChildren();
  $("#upRef").replaceChildren(el("option", { value: "", text: "time reference: first ego stream (default)" }),
    ...Array.from(e.target.files, (f) => el("option", { value: f.name, text: `time reference: ${f.name}` })));
  for (const f of e.target.files) {
    const role = el("select", { class: "upRole" }, el("option", { value: "ego", text: "ego (head camera)" }), el("option", { value: "exo", text: "exo (fixed camera)" }));
    const person = el("input", { class: "upPerson", placeholder: "who wears it (ego)", maxlength: "32" });
    rows.append(el("div", { class: "mono uprow" }, el("span", { class: "upname", text: f.name }), role, person));
  }
});
$("#uploadForm").addEventListener("submit", async (e) => {
  e.preventDefault(); const files = $("#upFiles").files; if (!files.length) return;
  const fd = new FormData(); fd.append("name", $("#upName").value);
  fd.append("roles", Array.from(document.querySelectorAll(".upRole")).map((s) => s.value).join(","));
  fd.append("persons", Array.from(document.querySelectorAll(".upPerson")).map((s) => s.value.replace(/,/g, " ").trim()).join(","));
  if ($("#upRef").value) fd.append("reference", $("#upRef").value);  // original filename; the server maps it to the stream
  for (const f of files) fd.append("files", f);
  const btn = $("#uploadForm button[type=submit]"); btn.disabled = true; $("#upStatus").textContent = `uploading ${files.length} files…`;
  let j;
  try { j = await api(URLS.upload(), { method: "POST", body: fd }); }
  catch (err) { $("#upStatus").textContent = `upload failed: ${err.message}`; return; }
  finally { btn.disabled = false; }
  $("#upStatus").textContent = `created ${j.name}; ${j.queued ? "pipeline queued" : "press Run pipeline to process it"}`;
  await loadEpisodes(j.name);
});
/* SERVER-ONLY END */

loadEpisodes();

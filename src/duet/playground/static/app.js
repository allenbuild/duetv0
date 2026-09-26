/* Duet EgoExo Playground: synced multi-view playback with overlays, 3D panel, quality table. */
const $ = (s) => document.querySelector(s);
const COCO = [[5,7],[7,9],[6,8],[8,10],[5,6],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,5],[0,6]];
const HAND = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
// MediaPipe pose (33): shoulders 11/12, elbows 13/14, wrists 15/16, hips 23/24, knees 25/26, ankles 27/28, nose 0
const MP33 = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[0,11],[0,12]];
const C = {left:"#57E39B", right:"#FF6B6B", body:"#B58CFF", obj:"#3EA7FF", held:"#FFD447", accent:"#3FA58C", mute:"#8A9591"};

let ep = null, overlays = {}, body3d = null, world3d = null, imuArm = {}, views = {}, playing = false, t = 0, master = null, raf = 0, three = null;
const COCO17 = [[5,7],[7,9],[6,8],[8,10],[5,6],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,5],[0,6]];

async function api(path, opts) { const r = await fetch(path, opts); return r.json(); }

async function loadEpisodes() {
  const eps = await api("/api/episodes"); const sel = $("#episode"); sel.innerHTML = "";
  for (const e of eps) { const o = document.createElement("option"); o.value = e.name; o.textContent = `${e.name}  (${e.streams.length} streams, ${e.duration_s.toFixed(0)} s)`; sel.appendChild(o); }
  if (eps.length) loadEpisode(sel.value);
}

async function loadEpisode(name) {
  stop(); ep = await api(`/api/episode/${name}`); overlays = {}; body3d = null; imuArm = {}; views = {};
  renderStages(); renderNotes(); renderQC();
  $("#egos").innerHTML = ""; $("#exos").innerHTML = "";
  for (const s of ep.streams) {
    const div = document.createElement("div"); div.className = "view"; div.innerHTML = `<video muted playsinline preload="auto" src="/episodes/${name}/${s.path}"></video><canvas></canvas><div class="tag"><b>${s.name}</b> · ${s.role}${s.person ? " · " + s.person : ""} · offset ${s.offset_s >= 0 ? "+" : ""}${s.offset_s.toFixed(2)} s</div>`;
    (s.role === "ego" ? $("#egos") : $("#exos")).appendChild(div);
    views[s.name] = { s, video: div.querySelector("video"), canvas: div.querySelector("canvas") };
    if (!master || s.name === ep.reference) master = views[s.name];
  }
  const dur = ep.common_end_s - ep.common_start_s; $("#scrub").max = Math.max(1, Math.round(dur * 100));
  seek(0);
  // overlays, 3D, IMU load lazily in the background
  for (const s of ep.streams) api(`/api/episode/${name}/overlay/${s.name}`).then((o) => { overlays[s.name] = o; draw(); });
  api(`/api/episode/${name}/body3d`).then((b) => { body3d = b.available ? b : null; setupThree(); });
  api(`/api/episode/${name}/world3d`).then((w) => { world3d = w.available ? w : null; setupThree(); });
  for (const s of ep.streams) if (s.imu) api(`/api/episode/${name}/imu_arm/${s.name}`).then((a) => { if (a.available) { imuArm[s.name] = a; setupThree(); } });
}

function renderStages() {
  const div = $("#stages"); div.innerHTML = "";
  for (const st of ep.stages) { const info = ep.status[st] || {}; const el = document.createElement("div"); el.className = `stage ${info.state || ""}`;
    el.innerHTML = `<span class="dot"></span><span class="mono">${st}</span><span style="color:var(--mute)">${info.state || "pending"}</span>`; el.title = info.detail || ""; div.appendChild(el); }
  $("#runstate").textContent = ep.running ? "running…" : "";
}
function renderNotes() { $("#notes").textContent = (ep.notes || []).join("\n") + (ep.notes && ep.notes.length ? "\n" : "") + `reference: ${ep.reference} · window ${ep.common_start_s.toFixed(2)}–${ep.common_end_s.toFixed(2)} s · ${ep.proc_fps} fps processing`; }
function renderQC() {
  const tb = $("#qc tbody"); tb.innerHTML = "";
  if (!ep.qc) { tb.innerHTML = `<tr><td style="color:var(--mute)">run the pipeline</td></tr>`; return; }
  const row = (k, v, fmt = (x) => x) => { const tr = document.createElement("tr"); tr.innerHTML = `<td>${k}</td><td class="num">${v == null ? "–" : fmt(v)}</td>`; tb.appendChild(tr); };
  const pct = (x) => (x * 100).toFixed(0) + "%";
  for (const [name, q] of Object.entries(ep.qc.streams)) {
    row(`${name} good frames`, q.good_frame_percent, pct);
    if (q.hand_presence_ratio != null) row(`${name} hands present`, q.hand_presence_ratio, pct);
    if (q.person_presence_ratio != null) row(`${name} person present`, q.person_presence_ratio, pct);
    row(`${name} stability`, q.stability_score, (x) => x.toFixed(2)); row(`${name} lighting`, q.lighting_score, (x) => x.toFixed(2));
    if (q.alignment_confidence != null) row(`${name} sync confidence`, q.alignment_confidence, (x) => "×" + x.toFixed(1));
  }
  const e = ep.qc.episode; row("both people visible (exo)", e.both_visible_ratio, pct); row("hands in every ego view", e.hands_all_egos_ratio, pct);
}

/* ---------------- playback ---------------- */
function seek(tt) { t = Math.max(0, Math.min(tt, ep.common_end_s - ep.common_start_s)); for (const v of Object.values(views)) { const st = ep.common_start_s + t - v.s.offset_s; if (Math.abs(v.video.currentTime - st) > 0.08) v.video.currentTime = st; } updateLabel(); draw(); }
function updateLabel() { $("#tlabel").textContent = `${t.toFixed(2)} / ${(ep.common_end_s - ep.common_start_s).toFixed(2)} s`; $("#scrub").value = Math.round(t * 100); }
function play() { playing = true; $("#play").textContent = "Pause"; for (const v of Object.values(views)) v.video.play().catch(() => {}); loop(); }
function stop() { playing = false; $("#play").textContent = "Play"; for (const v of Object.values(views)) v.video.pause(); cancelAnimationFrame(raf); }
function loop() { if (!playing) return; t = master.video.currentTime - (ep.common_start_s - master.s.offset_s);
  for (const v of Object.values(views)) { if (v === master) continue; const want = ep.common_start_s + t - v.s.offset_s; if (Math.abs(v.video.currentTime - want) > 0.12) v.video.currentTime = want; }
  if (t >= ep.common_end_s - ep.common_start_s) { stop(); } updateLabel(); draw(); raf = requestAnimationFrame(loop); }

/* ---------------- overlays ---------------- */
function frameIndex() { return Math.round(t * ep.proc_fps); }
function draw() {
  const show = $("#showOverlays").checked; const k = frameIndex();
  for (const v of Object.values(views)) {
    const cv = v.canvas, ctx = cv.getContext("2d"); const W = cv.clientWidth, H = cv.clientHeight; if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    ctx.clearRect(0, 0, W, H); const o = overlays[v.s.name]; if (!show || !o || !o.img_w) continue;
    // letterboxed video: map processed-frame px -> canvas px
    const vw = v.video.videoWidth || o.img_w, vh = v.video.videoHeight || o.img_h; const scale = Math.min(W / vw, H / vh); const ox = (W - vw * scale) / 2, oy = (H - vh * scale) / 2; const sx = vw / o.img_w * scale, sy = vh / o.img_h * scale;
    const P = (x, y) => [ox + x * sx, oy + y * sy];
    ctx.lineWidth = 1.5; ctx.font = "10px IBM Plex Mono, monospace";
    if (o.objects && o.objects[k]) { const held = heldBoxes(o, k); o.objects[k].forEach((b, j) => { if (b[4] == null) return; const [x0, y0] = P(b[0], b[1]), [x1, y1] = P(b[2], b[3]); ctx.strokeStyle = held.has(j) ? C.held : C.obj; ctx.lineWidth = held.has(j) ? 2 : 1; ctx.globalAlpha = held.has(j) ? 1 : 0.6; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0); ctx.fillStyle = ctx.strokeStyle; ctx.fillText(o.object_names[k][j], x0 + 2, y0 - 2); ctx.globalAlpha = 1; }); }
    if (o.body2d && o.body2d[k]) for (const person of o.body2d[k]) { if (person[0][0] == null) continue; ctx.strokeStyle = C.body; ctx.fillStyle = C.body; ctx.lineWidth = 1.5;
      for (const [a, b] of COCO) { if (person[a][2] > 0.45 && person[b][2] > 0.45) { const [x0, y0] = P(person[a][0], person[a][1]), [x1, y1] = P(person[b][0], person[b][1]); ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); } }
      for (const kp of person) if (kp[2] > 0.45) { const [x, y] = P(kp[0], kp[1]); ctx.beginPath(); ctx.arc(x, y, 2, 0, 7); ctx.fill(); } }
    if (o.hands2d && o.hands2d[k]) o.hands2d[k].forEach((hand, side) => { if (hand[0][0] == null) return; ctx.strokeStyle = ctx.fillStyle = side === 0 ? C.left : C.right; ctx.lineWidth = 1.6;
      for (const [a, b] of HAND) { const [x0, y0] = P(hand[a][0], hand[a][1]), [x1, y1] = P(hand[b][0], hand[b][1]); ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); }
      for (const lm of hand) { const [x, y] = P(lm[0], lm[1]); ctx.beginPath(); ctx.arc(x, y, 2, 0, 7); ctx.fill(); } });
  }
  drawThree();
}
function heldBoxes(o, k) { const held = new Set(); if (!o.hands2d || !o.hands2d[k]) return held;
  o.objects[k].forEach((b, j) => { if (b[4] == null) return; for (const hand of o.hands2d[k]) { if (hand[0][0] == null) continue; let inside = 0; for (const lm of hand) if (lm[0] >= b[0] && lm[0] <= b[2] && lm[1] >= b[1] && lm[1] <= b[3]) inside++; if (inside >= 8) held.add(j); } }); return held; }

/* ---------------- 3D panel (three.js) ---------------- */
function setupThree() {
  const cv = $("#three"); if (!three) {
    const renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true }); const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0b0f0e);
    const camera = new THREE.PerspectiveCamera(40, 4 / 3, 0.05, 20); camera.position.set(1.6, 1.6, 2.4); camera.lookAt(0, 0.9, 0);
    const grid = new THREE.GridHelper(3, 12, 0x26302d, 0x1a2220); scene.add(grid); scene.add(new THREE.AxesHelper(0.3));
    three = { renderer, scene, camera, lines: [] };
  }
  const has3d = !!body3d, hasImu = Object.keys(imuArm).length > 0, hasWorld = !!world3d;
  $("#threeTitle").textContent = hasWorld ? "3D · world frame (board): triangulated bodies, heads, objects" : has3d ? `3D body · monocular from ${body3d.stream}` : hasImu ? "3D arms · IMU harness (Eidon 7-slot)" : "3D";
  $("#threeNote").textContent = hasWorld ? `metres in the board frame. heads: ${Object.entries(world3d.headpose_backends || {}).map(([k, v]) => k + "=" + v).join(", ") || "none"}` : has3d ? "MediaPipe world landmarks: metres, hip-centred, single camera. Not triangulated." : hasImu ? "Arm chain from sensor quaternions, chest-relative (ported from Eidon Sim)." : "run body3d or add IMU";
  drawThree();
}
function segLines(pairs, pts, color) { const g = new THREE.BufferGeometry(); const arr = []; for (const [a, b] of pairs) { if (!pts[a] || !pts[b] || pts[a][0] == null || pts[b][0] == null) continue; arr.push(...pts[a], ...pts[b]); }
  g.setAttribute("position", new THREE.Float32BufferAttribute(arr, 3)); return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color })); }
function drawThree() {
  if (!three) return; for (const l of three.lines) three.scene.remove(l); three.lines = [];
  const k = frameIndex();
  if (world3d) {
    // board frame: X right, Y down the board, Z into the board. Scene: x = X, up = -Z, z = Y
    const toScene = (p) => (p == null || p[0] == null ? null : [p[0], -p[2], p[1]]);
    if (world3d.bodies[k]) for (const person of world3d.bodies[k]) { const pts = person.map(toScene); if (pts.every((p) => p == null)) continue; const l = segLines(COCO17, pts, 0xb58cff); three.scene.add(l); three.lines.push(l); }
    for (const key of Object.keys(world3d)) {
      if (key.startsWith("head_") && world3d[key][k] && world3d[key][k][0] != null) { const g = new THREE.SphereGeometry(0.06, 12, 12); const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: key.includes("leader") ? 0x8fa3cb : 0xe39468 })); const p = toScene(world3d[key][k]); m.position.set(...p); three.scene.add(m); three.lines.push(m); }
      if (key.startsWith("object_") && world3d[key][k] && world3d[key][k][0] != null) { const g = new THREE.BoxGeometry(0.05, 0.05, 0.05); const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0xffd447 })); const p = toScene(world3d[key][k]); m.position.set(...p); three.scene.add(m); three.lines.push(m); }
      if (key.startsWith("hands3d_") && world3d[key][k]) for (const hand of world3d[key][k]) { const pts = hand.map(toScene); if (pts.every((p) => p == null)) continue; const l = segLines(HAND, pts, 0x57e39b); three.scene.add(l); three.lines.push(l); }
    }
    for (const [name, T] of Object.entries(world3d.cameras || {})) { const g = new THREE.ConeGeometry(0.04, 0.08, 8); const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0x8a9591 })); const p = toScene([T[0][3], T[1][3], T[2][3]]); m.position.set(...p); three.scene.add(m); three.lines.push(m); }
  } else if (body3d && body3d.world[k]) for (const person of body3d.world[k]) { if (person[0][0] == null) continue;
    // MediaPipe world: x right, y down, z toward camera (metres) -> scene: x, up = -y, z
    const pts = person.map((p) => (p[0] == null ? null : [p[0], -p[1] + 0.9, p[2]])); const l = segLines(MP33, pts, 0xb58cff); three.scene.add(l); three.lines.push(l); }
  for (const a of Object.values(imuArm)) { const tt = ep.common_start_s + t; let i = 0, lo = 0, hi = a.t_s.length - 1; while (lo < hi) { i = (lo + hi) >> 1; if (a.t_s[i] < tt) lo = i + 1; else hi = i; } const pts = a.points[lo]; if (!pts) continue;
    pts.forEach((side, si) => { if (side[0][0] == null) return; const l = segLines([[0, 1], [1, 2], [2, 3]], side, si === 0 ? 0x57e39b : 0xff6b6b); three.scene.add(l); three.lines.push(l); }); }
  const cv = $("#three"); const W = cv.clientWidth, H = Math.round(W * 0.75); if (cv.width !== W || cv.height !== H) { three.renderer.setSize(W, H, false); three.camera.aspect = W / H; three.camera.updateProjectionMatrix(); }
  three.renderer.render(three.scene, three.camera);
}

/* ---------------- controls ---------------- */
$("#episode").addEventListener("change", (e) => loadEpisode(e.target.value));
$("#play").addEventListener("click", () => (playing ? stop() : play()));
$("#scrub").addEventListener("input", (e) => { stop(); seek(Number(e.target.value) / 100); });
$("#showOverlays").addEventListener("change", draw);
async function runPipeline(force) { await api(`/api/episode/${ep.name}/run?force=${force}`, { method: "POST" }); pollStatus(); }
$("#runAll").addEventListener("click", () => runPipeline(false)); $("#rerun").addEventListener("click", () => runPipeline(true));
async function pollStatus() { const name = ep.name; const d = await api(`/api/episode/${name}`); if (ep.name !== name) return; const wasRunning = ep.running; ep.status = d.status; ep.running = d.running; ep.notes = d.notes; ep.qc = d.qc; renderStages(); renderNotes(); renderQC();
  if (d.running) setTimeout(pollStatus, 2000); else if (wasRunning) loadEpisode(name); }
window.addEventListener("resize", draw);
loadEpisodes();

/* ---------------- upload ---------------- */
$("#upFiles").addEventListener("change", (e) => {
  const rows = $("#upRows"); rows.innerHTML = "";
  for (const f of e.target.files) { const r = document.createElement("div"); r.className = "mono"; r.style.cssText = "display:flex;gap:8px;align-items:center;font-size:12px";
    r.innerHTML = `<span style="min-width:220px;overflow:hidden;text-overflow:ellipsis">${f.name}</span><select class="upRole"><option value="ego">ego (head camera)</option><option value="exo">exo (fixed camera)</option></select><input class="upPerson" placeholder="who wears it (ego)" style="background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:4px;padding:3px 6px;font:inherit;width:160px">`; rows.appendChild(r); }
});
$("#uploadForm").addEventListener("submit", async (e) => {
  e.preventDefault(); const files = $("#upFiles").files; if (!files.length) return;
  const fd = new FormData(); fd.append("name", $("#upName").value);
  fd.append("roles", Array.from(document.querySelectorAll(".upRole")).map((s) => s.value).join(","));
  fd.append("persons", Array.from(document.querySelectorAll(".upPerson")).map((s) => s.value).join(","));
  for (const f of files) fd.append("files", f);
  $("#upStatus").textContent = `uploading ${files.length} files…`;
  const r = await fetch("/api/upload", { method: "POST", body: fd }); const j = await r.json();
  if (!r.ok) { $("#upStatus").textContent = "error: " + (j.detail || r.status); return; }
  $("#upStatus").textContent = `created ${j.name}; pipeline running`; await loadEpisodes(); $("#episode").value = j.name; await loadEpisode(j.name); pollStatus();
});

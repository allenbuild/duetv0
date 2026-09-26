/* Duet EgoExo Playground: human QA review of body2d / hands / objects on sampled frames plus autolabel proposals.
   Keys: 1/2/3 body ok/wrong/unsure, q/w/e hands, a/s/d objects, space = next, backspace = prev. */
const $ = (s) => document.querySelector(s);
const COCO = [[5,7],[7,9],[6,8],[8,10],[5,6],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,5],[0,6]];
const HAND = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const C = { left: "#57E39B", right: "#FF6B6B", body: "#B58CFF", obj: "#3EA7FF" };
const KEYS = { "1": ["body2d", "ok"], "2": ["body2d", "wrong"], "3": ["body2d", "unsure"], q: ["hands", "ok"], w: ["hands", "wrong"], e: ["hands", "unsure"], a: ["objects", "ok"], s: ["objects", "wrong"], d: ["objects", "unsure"] };
let ep = null, sample = null, frames = [], proposals = [], verdicts = {}, cur = 0, img = null;

async function api(path, opts) { const r = await fetch(path, opts); const j = await r.json().catch(() => ({})); if (!r.ok) throw new Error(`${path}: ${r.status} ${j.detail ? JSON.stringify(j.detail) : ""}`); return j; }
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function loadEpisodes() {
  const eps = await api("/api/episodes"); const s = $("#episode"); s.innerHTML = "";
  for (const e of eps) { const o = document.createElement("option"); o.value = e.name; o.textContent = e.name; s.appendChild(o); }
  const want = new URLSearchParams(location.search).get("ep");
  if (want && eps.some((e) => e.name === want)) s.value = want;
  if (eps.length) load();
}

async function load() {
  ep = $("#episode").value; history.replaceState(null, "", `?ep=${encodeURIComponent(ep)}`);
  $("#ptext").textContent = "loading sample…";
  sample = await api(`/api/episode/${encodeURIComponent(ep)}/review/sample?frac=${Number($("#frac").value) || 0.02}&seed=${Number($("#seed").value) || 0}`);
  frames = sample.items.filter((it) => it.kind === "frame"); proposals = sample.items.filter((it) => it.kind === "proposal"); verdicts = sample.verdicts || {};
  cur = Math.max(0, frames.findIndex((it) => !done(it))); if (cur < 0) cur = 0;
  renderProposals(); show(cur); refreshSummary();
}

/* which item types apply to a frame: a stream without that derived file is not reviewable for it */
const types = (it) => ["body2d", "hands", "objects"].filter((t) => it.has && it.has[t]);
const done = (it) => types(it).every((t) => verdicts[it.id] && verdicts[it.id][t]);
function progress() {
  const n = frames.length, d = frames.filter(done).length; const nv = frames.reduce((a, it) => a + types(it).filter((t) => verdicts[it.id] && verdicts[it.id][t]).length, 0), tv = frames.reduce((a, it) => a + types(it).length, 0);
  $("#progress i").style.width = (n ? (100 * d / n) : 0) + "%";
  $("#ptext").textContent = `${d} / ${n} frames complete · ${nv} / ${tv} verdicts · ${proposals.length} proposals · frames per stream: ${Object.entries(sample.n_frames || {}).map(([k, v]) => k + " " + v).join(", ")}`;
}

function show(i) {
  if (!frames.length) { $("#ftag").textContent = "no frames sampled"; $("#pos").textContent = ""; progress(); return; }
  cur = Math.max(0, Math.min(i, frames.length - 1)); const it = frames[cur];
  $("#ftag").innerHTML = `<b>${esc(it.stream)}</b> · ${esc(it.role)}${it.person ? " · " + esc(it.person) : ""} · frame ${it.frame_idx} · ${it.img_w}×${it.img_h}`;
  $("#pos").textContent = `${cur + 1} / ${frames.length}`;
  for (const t of ["body2d", "hands", "objects"]) {
    const row = document.querySelector(`.vrow[data-item="${t}"]`); const ok = it.has && it.has[t]; const v = verdicts[it.id] && verdicts[it.id][t];
    row.querySelectorAll("button").forEach((b) => { b.disabled = !ok; b.classList.toggle("on", !!v && v.verdict === b.dataset.v); });
    const n = t === "body2d" ? it.body2d.length : t === "hands" ? it.hands.length : it.objects.length;
    $(`#n-${t}`).textContent = ok ? `${n} detected${v ? " · " + v.verdict : ""}` : "not computed for this stream";
  }
  img = new Image(); img.onload = draw; img.src = it.image; progress();
}

function draw() {
  const it = frames[cur]; const cv = $("#canvas"), ctx = cv.getContext("2d"); if (!img) return;
  const W = Math.max(320, Math.round(cv.clientWidth || 640)), H = Math.round(W * img.naturalHeight / img.naturalWidth);
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  ctx.clearRect(0, 0, W, H); ctx.drawImage(img, 0, 0, W, H);
  if (!$("#showOverlays").checked) return;
  const sx = W / (it.img_w || img.naturalWidth), sy = H / (it.img_h || img.naturalHeight); const P = (x, y) => [x * sx, y * sy];
  ctx.lineWidth = 1.5; ctx.font = "11px IBM Plex Mono, monospace";
  for (const o of it.objects || []) { if (o.box[0] == null) continue; const [x0, y0] = P(o.box[0], o.box[1]), [x1, y1] = P(o.box[2], o.box[3]); ctx.strokeStyle = ctx.fillStyle = C.obj; ctx.globalAlpha = .8; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0); ctx.fillText(`${o.name} ${o.conf.toFixed(2)}`, x0 + 2, y0 - 2); ctx.globalAlpha = 1; }
  for (const person of it.body2d || []) { ctx.strokeStyle = ctx.fillStyle = C.body; ctx.lineWidth = 1.5;
    for (const [a, b] of COCO) { if (person[a][2] > 0.45 && person[b][2] > 0.45) { const [x0, y0] = P(person[a][0], person[a][1]), [x1, y1] = P(person[b][0], person[b][1]); ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); } }
    for (const kp of person) if (kp[2] > 0.45) { const [x, y] = P(kp[0], kp[1]); ctx.beginPath(); ctx.arc(x, y, 2.5, 0, 7); ctx.fill(); } }
  for (const h of it.hands || []) { const lm = h.landmarks; if (!lm || lm[0][0] == null) continue; ctx.strokeStyle = ctx.fillStyle = h.side === "left" ? C.left : C.right; ctx.lineWidth = 1.6;
    for (const [a, b] of HAND) { const [x0, y0] = P(lm[a][0], lm[a][1]), [x1, y1] = P(lm[b][0], lm[b][1]); ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); }
    for (const p of lm) { const [x, y] = P(p[0], p[1]); ctx.beginPath(); ctx.arc(x, y, 2, 0, 7); ctx.fill(); } }
}

async function vote(item, verdict) {
  const it = frames[cur]; if (!it || !(it.has && it.has[item])) return;
  const body = { id: it.id, stream: it.stream, frame_idx: it.frame_idx, item, verdict, note: $("#note").value };
  $("#status").textContent = "saving…";
  try { await api(`/api/episode/${encodeURIComponent(ep)}/review`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); }
  catch (e) { $("#status").textContent = "error: " + e.message; return; }
  (verdicts[it.id] = verdicts[it.id] || {})[item] = body; $("#note").value = ""; $("#status").textContent = `saved ${item}: ${verdict}`;
  show(cur); refreshSummary();
  if (done(it)) setTimeout(() => { if (frames[cur] === it) next(); }, 250);
}
function next() { const i = frames.findIndex((it, j) => j > cur && !done(it)); show(i >= 0 ? i : Math.min(cur + 1, frames.length - 1)); }

async function voteProposal(p, verdict) {
  $("#status").textContent = "saving…";
  try { await api(`/api/episode/${encodeURIComponent(ep)}/review`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: p.id, stream: "proposals", frame_idx: null, item: "proposal", verdict, note: $("#note").value }) }); }
  catch (e) { $("#status").textContent = "error: " + e.message; return; }
  p.status = verdict === "ok" ? "accepted" : verdict === "wrong" ? "rejected" : "unsure"; (verdicts[p.id] = verdicts[p.id] || {}).proposal = { verdict }; $("#note").value = ""; $("#status").textContent = `proposal ${p.index}: ${p.status}`;
  renderProposals(); refreshSummary();
}

function renderProposals() {
  const div = $("#proposals"); div.innerHTML = "";
  if (!proposals.length) { div.innerHTML = `<div style="color:var(--mute)">no proposals (derived/autolabel/proposals.json missing or empty)</div>`; return; }
  for (const p of proposals) {
    const row = document.createElement("div"); row.className = "prop"; const st = p.status || "pending";
    row.innerHTML = `<div><b>${esc(p.event)}</b> <span class="mono" style="color:var(--mute)">${Number(p.t0).toFixed(1)}–${Number(p.t1).toFixed(1)} s</span>${p.giver || p.receiver ? ` · ${esc(p.giver ?? "?")} → ${esc(p.receiver ?? "?")}` : ""}${p.confidence != null ? ` · conf ${Number(p.confidence).toFixed(2)}` : ""}<div class="st ${esc(st)}">${esc(st)}</div></div>
      <div style="display:flex;gap:6px"><button class="v-ok ${st === "accepted" ? "on" : ""}">accept</button><button class="v-wrong ${st === "rejected" ? "on" : ""}">reject</button><button class="v-unsure ${st === "unsure" ? "on" : ""}">unsure</button></div>`;
    const [b1, b2, b3] = row.querySelectorAll("button"); b1.onclick = () => voteProposal(p, "ok"); b2.onclick = () => voteProposal(p, "wrong"); b3.onclick = () => voteProposal(p, "unsure");
    div.appendChild(row);
  }
}

async function refreshSummary() {
  const s = await api(`/api/episode/${encodeURIComponent(ep)}/review/summary`); const tb = $("#summary tbody"); tb.innerHTML = "";
  const row = (name, b, dim) => { const tr = document.createElement("tr"); tr.innerHTML = `<td${dim ? ' style="color:var(--mute);padding-left:16px"' : ""}>${esc(name)}</td><td class="num">${b.n}</td><td class="num">${b.ok}</td><td class="num">${b.wrong}</td><td class="num">${b.unsure}</td><td class="num">${b.accuracy == null ? "–" : (100 * b.accuracy).toFixed(0) + "%"}</td>`; tb.appendChild(tr); };
  for (const [k, b] of Object.entries(s.by_item)) row(k, b);
  for (const [st, b] of Object.entries(s.by_stream)) { row(st, b, true); }
  const tr = document.createElement("tr"); tr.innerHTML = `<td colspan="6" style="color:var(--mute)">proposals: ${s.proposals.accepted} accepted · ${s.proposals.rejected} rejected · ${s.proposals.pending} pending of ${s.proposals.total} · ${s.total} verdicts total</td>`; tb.appendChild(tr);
}

$("#episode").addEventListener("change", load); $("#reload").addEventListener("click", load); $("#showOverlays").addEventListener("change", draw);
$("#prev").addEventListener("click", () => show(cur - 1)); $("#next").addEventListener("click", next);
document.querySelectorAll(".vrow button").forEach((b) => b.addEventListener("click", () => vote(b.closest(".vrow").dataset.item, b.dataset.v)));
document.addEventListener("keydown", (e) => {
  if (["TEXTAREA", "INPUT", "SELECT"].includes(e.target.tagName)) { if (e.key === "Escape") e.target.blur(); return; }
  if (e.key === " ") { e.preventDefault(); next(); return; }
  if (e.key === "Backspace" || e.key === "ArrowLeft") { e.preventDefault(); show(cur - 1); return; }
  if (e.key === "ArrowRight") { e.preventDefault(); show(cur + 1); return; }
  const k = KEYS[e.key.toLowerCase()]; if (k) { e.preventDefault(); vote(k[0], k[1]); }
});
window.addEventListener("resize", draw);
loadEpisodes();

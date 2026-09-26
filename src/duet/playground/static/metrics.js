/* Duet EgoExo Playground: operator metrics page. Reads /api/episode/<name>/metrics (written by the metrics stage). */
const $ = (s) => document.querySelector(s);
const C = { accent: "#3FA58C", warn: "#D9B64A", bad: "#E06060", obj: "#3EA7FF", body: "#B58CFF", mute: "#8A9591", line: "#26302D", ink: "#E6ECE9", dim: "#2A3431" };
let M = null, epName = null, extra = null;

async function api(path, opts) { const r = await fetch(path, opts); return r.json(); }
const fmt = (v, d = 2, unit = "") => (v === null || v === undefined || Number.isNaN(v)) ? "–" : `${Number(v).toFixed(d)}${unit}`;
const pct = (v) => (v === null || v === undefined) ? "–" : `${Math.round(v * 100)}%`;

async function loadEpisodes() {
  const eps = await api("/api/episodes"); const sel = $("#episode"); sel.innerHTML = "";
  for (const e of eps) { const o = document.createElement("option"); o.value = e.name; o.textContent = `${e.name}  (${e.duration_s.toFixed(0)} s)`; sel.appendChild(o); }
  const want = new URLSearchParams(location.search).get("ep");
  if (want && eps.some((e) => e.name === want)) sel.value = want;
  if (eps.length) load(sel.value);
}

async function load(name) {
  epName = name; history.replaceState(null, "", `?ep=${encodeURIComponent(name)}`);
  M = await api(`/api/episode/${name}/metrics`);
  if (!M.available) { $("#cards").innerHTML = `<div class="empty">No metrics for ${name} yet (stage: ${M.status ? M.status.state + " · " + (M.status.detail || "") : "not run"}). Press "Recompute metrics".</div>`;
    $("#score").textContent = ""; $("#components").innerHTML = ""; $("#formula").textContent = ""; $("#handovers thead").innerHTML = ""; $("#handovers tbody").innerHTML = ""; $("#notes").textContent = "";
    drawPerMinute([]); drawTimeline(); return; }
  extra = M.frames || null;
  renderCards(); renderScore(); drawPerMinute(M.per_minute); drawTimeline(); renderTable(); renderNotes();
}

function renderCards() {
  const h = M.handovers, cards = [];
  cards.push(["session", `${fmt(M.duration_s, 1)}<small>s</small>`, `${M.n_frames} frames @ ${M.fps} fps`]);
  cards.push(["handovers", `${h.count}<small>${fmt(h.rate_per_min, 1)}/min</small>`, `source: ${h.source}`]);
  cards.push(["receiver latency", `${fmt(h.median_receiver_response_s, 2)}<small>s</small>`, h.anticipation_fraction === null ? "no measurable onsets" : `${pct(h.anticipation_fraction)} receiver moved first`]);
  cards.push(["giver hold", `${fmt(h.median_giver_hold_s, 2)}<small>s</small>`, "median, before transfer"]);
  cards.push(["synchrony", `${fmt(M.synchrony.session, 2)}`, `session xcorr (lag ${fmt(M.synchrony.session_lag_s, 1)} s) · per-handover median ${fmt(M.synchrony.median_handover, 2)}`]);
  for (const [p, v] of Object.entries(M.hand_speed || {})) {
    cards.push([`${p} idle`, pct(v.idle_fraction), `hands visible ${pct(v.hands_visible_fraction)} · mean speed ${fmt(v.mean_speed, 2)} ${v.unit}`]);
  }
  if (M.speaking_fraction) for (const [p, v] of Object.entries(M.speaking_fraction)) cards.push([`${p} speaking`, pct(v), "from speech stage"]);
  $("#cards").innerHTML = cards.map(([t, v, s]) => `<div class="card"><h3>${t}</h3><div class="v">${v}</div><div class="sub">${s}</div></div>`).join("");
}

function renderScore() {
  const cs = M.coordination_score; $("#score").textContent = cs.score === null ? "–" : `${cs.score.toFixed(0)}${cs.partial ? " (partial)" : ""}`;
  $("#score").style.color = cs.score === null ? C.mute : cs.score >= 70 ? C.accent : cs.score >= 40 ? C.warn : C.bad;
  $("#components").innerHTML = `<table>${Object.entries(cs.components).map(([k, v]) => `<tr><td>${k}</td><td class="num">${fmt(v, 3)}</td></tr>`).join("")}</table>`;
  $("#formula").textContent = cs.formula;
}

function setupCanvas(cv) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight; cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h); g.font = "11px IBM Plex Mono, monospace"; return [g, w, h];
}

function drawPerMinute(rows) {
  const [g, w, h] = setupCanvas($("#perMinute")); if (!rows || !rows.length) { g.fillStyle = C.mute; g.fillText("no data", 10, 20); return; }
  const L = 36, R = 36, T = 12, B = 26, pw = w - L - R, ph = h - T - B, n = rows.length, gw = pw / n, bw = Math.min(22, gw / 5);
  const maxH = Math.max(1, ...rows.map((r) => r.handovers));
  g.strokeStyle = C.line; g.beginPath(); g.moveTo(L, T + ph); g.lineTo(L + pw, T + ph); g.stroke();
  // left axis: handovers (count); right axis: fractions 0..1
  g.fillStyle = C.mute; g.textAlign = "right"; g.fillText(`${maxH}`, L - 4, T + 8); g.fillText("0", L - 4, T + ph); g.textAlign = "left"; g.fillText("100%", L + pw + 4, T + 8); g.fillText("0%", L + pw + 4, T + ph);
  rows.forEach((r, i) => {
    const x0 = L + i * gw + gw / 2 - 2 * bw;
    const series = [[r.handovers / maxH, C.accent], [r.idle_fraction, C.warn], [r.speech_fraction, C.obj], [r.hands_visible_fraction, C.body]];
    series.forEach(([v, col], j) => { if (v === null || v === undefined) return; const bh = Math.max(1, v * ph); g.fillStyle = col; g.fillRect(x0 + j * bw, T + ph - bh, bw - 2, bh); });
    g.fillStyle = C.mute; g.textAlign = "center"; g.fillText(`min ${r.minute}`, L + i * gw + gw / 2, h - 8); g.textAlign = "left";
  });
}

function drawTimeline() {
  const [g, w, h] = setupCanvas($("#timeline")); if (!M || !M.available) return;
  const persons = M.persons || [], L = 100, R = 10, T = 8, pw = w - L - R, dur = M.duration_s, t0 = M.per_minute.length ? M.per_minute[0].t0_s : 0;
  const x = (t) => L + ((t - t0) / dur) * pw, rowH = 22, rows = 1 + persons.length * (M.speaking_fraction ? 2 : 1);
  // handover row
  g.fillStyle = C.mute; g.fillText("handovers", 4, T + 15);
  for (const e of M.handovers.events) { g.fillStyle = C.accent; g.fillRect(x(e.t0), T + 4, Math.max(2, x(e.t1) - x(e.t0)), rowH - 8); }
  // per-person idle / visibility / speech strips come per frame from the API ("frames"); the per-minute bands are the fallback
  let y = T + rowH;
  persons.forEach((p) => {
    g.fillStyle = C.mute; g.fillText(`${p} idle`, 4, y + 15);
    if (extra && extra[`idle_${p}`]) {
      const idle = extra[`idle_${p}`], vis = extra[`hand_visible_${p}`], n = idle.length;
      for (let k = 0; k < n; k++) { const xa = L + (k / n) * pw, xb = L + ((k + 1) / n) * pw; if (vis && !vis[k]) { g.fillStyle = C.dim; g.fillRect(xa, y + 4, xb - xa + 0.5, rowH - 8); } if (idle[k]) { g.fillStyle = C.warn; g.fillRect(xa, y + 4, xb - xa + 0.5, rowH - 8); } }
    } else {
      M.per_minute.forEach((r) => { if (r.idle_fraction === null) return; g.fillStyle = C.warn; g.globalAlpha = 0.2 + 0.8 * r.idle_fraction; g.fillRect(x(r.t0_s), y + 4, x(r.t1_s) - x(r.t0_s), rowH - 8); g.globalAlpha = 1; });
    }
    y += rowH;
    if (M.speaking_fraction && p in M.speaking_fraction) {
      g.fillStyle = C.mute; g.fillText(`${p} speech`, 4, y + 15);
      if (extra && extra[`speaking_${p}`]) { const s = extra[`speaking_${p}`], n = s.length; for (let k = 0; k < n; k++) if (s[k]) { g.fillStyle = C.obj; g.fillRect(L + (k / n) * pw, y + 4, pw / n + 0.5, rowH - 8); } }
      else { M.per_minute.forEach((r) => { if (r.speech_fraction === null) return; g.fillStyle = C.obj; g.globalAlpha = 0.2 + 0.8 * r.speech_fraction; g.fillRect(x(r.t0_s), y + 4, x(r.t1_s) - x(r.t0_s), rowH - 8); g.globalAlpha = 1; }); }
      y += rowH;
    }
  });
  // time axis
  g.strokeStyle = C.line; g.beginPath(); g.moveTo(L, y + 4); g.lineTo(L + pw, y + 4); g.stroke(); g.fillStyle = C.mute;
  const step = dur > 120 ? 30 : dur > 40 ? 10 : 5;
  for (let t = 0; t <= dur; t += step) { g.fillText(`${(t0 + t).toFixed(0)}s`, x(t0 + t) - 8, y + 16); }
}

function renderTable() {
  const cols = [["id", "#"], ["t0", "t0 s"], ["t1", "t1 s"], ["giver", "giver"], ["receiver", "receiver"], ["object", "object"], ["giver_reach_onset_s", "giver onset s"], ["receiver_response_s", "receiver latency s"],
                ["transfer_s", "transfer s"], ["giver_hold_s", "giver hold s"], ["synchrony", "synchrony"], ["synchrony_lag_s", "lag s"], ["min_wrist_dist_m", "min wrist dist m"], ["giver_basis", "giver from"], ["source", "source"]];
  $("#handovers thead").innerHTML = `<tr>${cols.map(([k, t]) => `<th class="${typeof M.handovers.events[0]?.[k] === "number" ? "num" : ""}">${t}</th>`).join("")}</tr>`;
  $("#handovers tbody").innerHTML = M.handovers.events.length ? M.handovers.events.map((e) => `<tr>${cols.map(([k]) => { const v = e[k]; const num = typeof v === "number"; return `<td class="${num ? "num" : ""}">${num ? fmt(v, k === "id" ? 0 : 2) : (v ?? "–")}</td>`; }).join("")}</tr>`).join("")
    : `<tr><td colspan="${cols.length}" class="empty">no handover events (source: ${M.handovers.source})</td></tr>`;
}

function renderNotes() {
  const d = M.definitions || {}; const hs = Object.entries(M.hand_speed || {}).map(([p, v]) => `${p}: ${v.source} (unit ${v.unit}, still < ${v.still_threshold})`).join("\n");
  $("#notes").textContent = `hand speed\n${hs}\n\nthresholds ${JSON.stringify(d)}\n\n${(M.notes || []).join("\n")}\ncomputed ${M.computed_utc}`;
}

$("#episode").addEventListener("change", (e) => load(e.target.value));
$("#run").addEventListener("click", async () => {
  $("#runstate").textContent = "running…"; await api(`/api/episode/${epName}/run?stages=metrics&force=true`, { method: "POST" });
  const poll = setInterval(async () => { const e = await api(`/api/episode/${epName}`); const st = e.status.metrics || {}; $("#runstate").textContent = `${st.state || ""} ${st.detail || ""}`;
    if (!e.running) { clearInterval(poll); load(epName); } }, 1500);
});
window.addEventListener("resize", () => { if (M && M.available) { drawPerMinute(M.per_minute); drawTimeline(); } });
loadEpisodes();

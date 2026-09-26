/* Duet EgoExo Playground: dense VLM annotation timeline. Episode from ?ep=<name>, default = first episode. */
const $ = (s) => document.querySelector(s);
let data = null, sel = -1;

async function api(path) { const r = await fetch(path); if (!r.ok) throw new Error(`${path}: ${r.status}`); return r.json(); }
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const dash = (v) => (v == null || v === "" ? "–" : esc(v));

async function loadEpisodes() {
  const eps = await api("/api/episodes"); const s = $("#episode"); s.innerHTML = "";
  for (const e of eps) { const o = document.createElement("option"); o.value = e.name; o.textContent = e.name; s.appendChild(o); }
  const want = new URLSearchParams(location.search).get("ep");
  if (want && eps.some((e) => e.name === want)) s.value = want;
  if (eps.length) load(s.value);
}

async function load(name) {
  history.replaceState(null, "", `?ep=${encodeURIComponent(name)}`);
  data = await api(`/api/episode/${encodeURIComponent(name)}/annotations`); sel = -1;
  const st = data.status || {}; const lg = data.log || {};
  const bits = [`stage: ${st.state || "not run"}${st.detail ? " · " + st.detail : ""}`];
  if (data.available) bits.push(`backend ${lg.backend || "?"}${lg.model ? " / " + lg.model : ""}`, `${data.items.length} keyframes every ${lg.interval_s ?? "?"} s`,
    `elapsed ${lg.elapsed_s ?? "?"} s`, `tokens in/out ${lg.input_tokens ?? 0}/${lg.output_tokens ?? 0}`, `errors ${lg.errors ?? 0}`);
  $("#meta").innerHTML = bits.map(esc).join(" &nbsp;·&nbsp; ");
  renderList();
  if (data.items.length) show(0);
  else { $("#big").removeAttribute("src"); $("#dTitle").textContent = "no annotations"; $("#dSummary").textContent = "run the annotate stage"; $("#dPersons tbody").innerHTML = ""; $("#dHandover").textContent = ""; $("#dRaw").textContent = ""; }
}

function coordBadge(it) {
  if (it.error) return `<span class="badge error">error</span>`;
  const c = it.coordination || "none"; const h = it.handover && it.handover.occurring ? `<span class="badge handover">handover: ${esc(it.handover.giver)} → ${esc(it.handover.receiver)} (${esc(it.handover.object)})</span>` : "";
  return `<span class="badge ${esc(c)}">${esc(c)}</span>${h}`;
}

function renderList() {
  const list = $("#list"); list.innerHTML = "";
  if (!data.available || !data.items.length) { list.innerHTML = `<div class="empty">No annotations yet for this episode. Run the <span class="mono">annotate</span> stage from the viewer (Run pipeline), or <span class="mono">python -m duet.playground.annotate ${esc(data.episode)}</span>.</div>`; return; }
  data.items.forEach((it, i) => {
    const div = document.createElement("div"); div.className = "item"; div.dataset.i = i;
    const persons = (it.persons || []).map((p) => `<b>${esc(p.label)}</b> ${dash(p.action)}${p.holding && p.holding.length ? " · holding " + esc(p.holding.join(", ")) : ""}`).join("<br>");
    div.innerHTML = `<img loading="lazy" src="${esc(it.image_url)}" alt=""><div><div class="mono"><span class="t">t = ${Number(it.t).toFixed(2)} s</span> <span style="color:var(--mute)">· #${it.idx} · frame ${it.frame}</span></div>
      <div class="sum">${it.summary ? esc(it.summary) : `<span style="color:var(--mute)">${esc(it.note || it.error || "no summary")}</span>`}</div><div class="row">${persons}</div><div style="margin-top:4px">${coordBadge(it)}</div></div>`;
    div.addEventListener("click", () => show(i)); list.appendChild(div);
  });
}

function show(i) {
  const it = data.items[i]; if (!it) return; sel = i;
  document.querySelectorAll(".item").forEach((el) => el.classList.toggle("active", Number(el.dataset.i) === i));
  const active = document.querySelector(".item.active"); if (active) active.scrollIntoView({ block: "nearest" });
  $("#big").src = it.image_url;
  $("#dTitle").textContent = `keyframe #${it.idx} · t = ${Number(it.t).toFixed(2)} s · frame ${it.frame}${it._meta ? " · " + it._meta.backend + (it._meta.model ? "/" + it._meta.model : "") + (it._meta.elapsed_s != null ? " · " + it._meta.elapsed_s + " s" : "") : ""}`;
  $("#dSummary").innerHTML = `${coordBadge(it)}<div style="margin-top:6px">${it.summary ? esc(it.summary) : `<span style="color:var(--mute)">${esc(it.note || it.error || "no summary")}</span>`}</div>`;
  const tb = $("#dPersons tbody"); tb.innerHTML = "";
  for (const p of it.persons || []) { const tr = document.createElement("tr"); tr.innerHTML = `<td><b>${dash(p.label)}</b></td><td>${dash(p.action)}</td><td>${p.holding && p.holding.length ? esc(p.holding.join(", ")) : "<span style='color:var(--mute)'>free</span>"}</td><td>${dash(p.attending_to)}</td>`; tb.appendChild(tr); }
  if (!(it.persons || []).length) tb.innerHTML = `<tr><td colspan="4" style="color:var(--mute)">no persons reported</td></tr>`;
  const h = it.handover || {};
  $("#dHandover").textContent = h.occurring == null ? "handover: unknown" : h.occurring ? `handover: ${h.giver ?? "?"} gives ${h.object ?? "?"} to ${h.receiver ?? "?"}` : "handover: none";
  const raw = Object.fromEntries(Object.entries(it).filter(([k]) => k !== "image_url")); $("#dRaw").textContent = JSON.stringify(raw, null, 1);
}

$("#episode").addEventListener("change", (e) => load(e.target.value));
$("#reload").addEventListener("click", () => load($("#episode").value));
document.addEventListener("keydown", (e) => { if (!data || !data.items.length || e.target.tagName === "SELECT") return;
  if (e.key === "ArrowDown" || e.key === "j") { e.preventDefault(); show(Math.min(sel + 1, data.items.length - 1)); }
  if (e.key === "ArrowUp" || e.key === "k") { e.preventDefault(); show(Math.max(sel - 1, 0)); } });
loadEpisodes();

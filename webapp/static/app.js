"use strict";
const $ = (s) => document.querySelector(s);
const COL = { Post: "var(--green)", "Do not post": "var(--red)", Unsure: "var(--amber)" };
let mode = "file";

// segmented toggle: unpublished (file) <-> published (url)
$("#seg").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-mode]");
  if (!b) return;
  mode = b.dataset.mode;
  $("#seg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
  $("#panel-file").classList.toggle("hidden", mode !== "file");
  $("#panel-url").classList.toggle("hidden", mode !== "url");
});
$("#cost").addEventListener("input", (e) => ($("#costv").textContent = e.target.value));

// ---- default demo samples: one click loads a bundled video + its caption ----
fetch("/static/samples/manifest.json")
  .then((r) => (r.ok ? r.json() : []))
  .then((list) => {
    const box = $("#samples");
    if (!list.length) { box.textContent = "(no bundled examples)"; return; }
    box.innerHTML = "";
    list.forEach((m) => {
      const b = document.createElement("button");
      b.type = "button";
      b.style.cssText = "border:1px solid var(--line);background:#fff;border-radius:8px;padding:6px 12px;margin:0 8px 4px 0;font-size:13px;cursor:pointer;";
      b.textContent = `${m.name}  (${m.label})`;
      b.onclick = () => loadSample(m);
      box.appendChild(b);
    });
  })
  .catch(() => ($("#samples").textContent = "(examples unavailable)"));

async function loadSample(m) {
  const loaded = $("#loaded");
  loaded.style.display = "block";
  loaded.textContent = "loading example…";
  try {
    const [vidRes, txtRes] = await Promise.all([
      fetch(`/static/samples/${m.name}.mp4`),
      fetch(`/static/samples/${m.name}.txt`),
    ]);
    const blob = await vidRes.blob();
    const f = new File([blob], `${m.name}.mp4`, { type: "video/mp4" });
    const dt = new DataTransfer();
    dt.items.add(f);
    $("#file").files = dt.files;                          // set the file input programmatically
    // caption from the feature-description text (line "caption: ...")
    const txt = await txtRes.text();
    const cap = (txt.match(/^caption:\s*(.+)$/m) || [])[1];
    if (cap) $("#cap").value = cap.trim();
    if (m.post_time) $("#when").value = m.post_time;         // true post time (decoded from the video id)
    loaded.innerHTML = `loaded <b>${m.name}.mp4</b> + caption from <code>${m.name}.txt</code>` +
      (m.post_time ? ` + post time <code>${m.post_time}</code>` : "") +
      ` · topic hint: ${m.topic}. Press <b>Evaluate</b>.`;
    // make sure we are on the file (unpublished) tab
    if (mode !== "file") $("#seg").querySelector('button[data-mode="file"]').click();
  } catch (e) {
    loaded.textContent = "could not load the example: " + (e.message || e);
  }
}

$("#go").addEventListener("click", async () => {
  const out = $("#out");
  out.classList.remove("hidden");
  const cost = parseFloat($("#cost").value);
  const lang = $("#lang").value;
  const plays = parseFloat($("#plays").value), likes = parseFloat($("#likes").value);
  const day1 = Number.isFinite(plays) && plays > 0
    ? { plays: plays, likes: Number.isFinite(likes) ? likes : 0 } : null;

  out.innerHTML = `<div><span class="spinner"></span>Extracting features and scoring…
      <span class="hint" style="display:inline">(Whisper can take 10–30 s on the first video)</span></div>`;
  $("#go").disabled = true;
  try {
    let res;
    if (mode === "url") {
      const url = $("#url").value.trim();
      if (!url) throw new Error("Paste a video URL first.");
      res = await postJSON("/api/score/url", { url, cost_ratio: cost, day1, language: lang });
    } else {
      const f = $("#file").files[0];
      if (!f) throw new Error("Choose a video file first.");
      const fd = new FormData();
      fd.append("file", f);
      fd.append("caption", $("#cap").value || "");
      fd.append("post_time", $("#when").value || "");
      fd.append("cost_ratio", cost);
      fd.append("language", lang);
      if (day1) { fd.append("plays", day1.plays); fd.append("likes", day1.likes); }
      res = await postForm("/api/score/file", fd);
    }
    render(res);
  } catch (e) {
    out.innerHTML = `<div class="err">${escapeHtml(e.message || String(e))}</div>`;
  } finally {
    $("#go").disabled = false;
  }
});

// ---- system readiness dot (models pre-warmed at server start) ----
(async function warmDot() {
  const el = $("#warmDot");
  if (!el) return;
  try {
    const h = await fetch("/api/health").then((r) => r.json());
    if (h.warmup === "warm") {
      el.innerHTML = `<span style="color:var(--green)">●</span> ready · LLM: ${h.llm_backend}`;
      return;
    }
    el.innerHTML = `<span style="color:var(--amber)">●</span> warming models…`;
  } catch { el.innerHTML = `<span style="color:var(--red)">●</span> backend offline`; }
  setTimeout(warmDot, 4000);
})();

// ---- self-learning flywheel: track a posted video + show pool status ----
const fwOut = () => $("#fwOut");
$("#trackBtn")?.addEventListener("click", async () => {
  const url = $("#url").value.trim();
  if (!url) { fwOut().textContent = "Paste the video URL above first."; return; }
  fwOut().innerHTML = `<span class="spinner"></span>tracking (metadata + transcript + creator baseline)…`;
  try {
    const r = await postJSON("/api/track", { url, language: $("#lang").value });
    fwOut().innerHTML =
      `tracked <b>${escapeHtml(r.video_id || "")}</b> by @${escapeHtml(r.creator || "?")} · ` +
      `views now ${r.views_now ?? "?"} · creator median ${r.creator_median_views ?? "?"} ` +
      `(${r.baseline_n} recent videos) · transcript: ${r.transcript_extracted ? "yes" : "no"}` +
      (r.note ? ` · ${escapeHtml(r.note)}` : "") +
      `<br>Label matures in 14 days; poll runs via scripts/flywheel_tick.py or the status button.`;
  } catch (e) { fwOut().innerHTML = `<span style="color:var(--red)">${escapeHtml(e.message || e)}</span>`; }
});
$("#fwStatusBtn")?.addEventListener("click", async () => {
  fwOut().innerHTML = `<span class="spinner"></span>polling tracked videos…`;
  try {
    const r = await fetch("/api/flywheel/tick", { method: "POST" }).then((x) => x.json());
    const s = r.status || {};
    fwOut().innerHTML =
      `poll: ${JSON.stringify(r.poll)} · pool: ${JSON.stringify(s.counts || {})} · ` +
      `labels: hit ${s.labels?.hit ?? 0} / flop ${s.labels?.flop ?? 0} · ` +
      `retrain ready: ${s.ready_to_retrain ? "YES" : "no (need " + s.min_new + ")"}` +
      (s.bias_warning ? `<br><span style="color:var(--amber)">${escapeHtml(s.bias_warning)}</span>` : "");
  } catch (e) { fwOut().innerHTML = `<span style="color:var(--red)">${escapeHtml(e.message || e)}</span>`; }
});

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
  return r.json();
}
async function postForm(url, fd) {
  const r = await fetch(url, { method: "POST", body: fd });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
  return r.json();
}

function render(r) {
  const ex = r.extracted || {}, inp = r.inputs || {};
  const pct = Math.round((r.probability || 0) * 100);
  const kv = (k, v) => `<div class="k">${k}</div><div>${v}</div>`;
  let html =
    `<div class="rec" style="color:${COL[r.recommendation] || ""}">${r.recommendation}
       <span style="font-weight:400;font-size:16px;color:var(--mut)">P(success) = ${pct}% · ${escapeHtml(r.model || "")}</span></div>
     <div class="bar"><div style="width:${pct}%"></div></div>
     <p style="font-size:14px">${escapeHtml(r.rationale || "")}</p>`;

  if (r.video_boost) {
    const b = r.video_boost, bp = Math.round((b.probability || 0) * 100);
    html +=
      `<div style="border-top:1px solid var(--line);margin-top:14px;padding-top:12px">
         <p style="margin:0 0 2px;font-weight:600">Video boost — multimodal model (frames + audio)</p>
         <div class="rec" style="font-size:20px;color:${COL[b.recommendation] || ""}">${b.recommendation}
           <span style="font-weight:400;font-size:14px;color:var(--mut)">P(success) = ${bp}%</span></div>
         <div class="bar"><div style="width:${bp}%;background:var(--green)"></div></div>
         <p class="hint">${escapeHtml(b.auc_note || "")}</p>
       </div>`;
  }
  html += `<p style="margin:14px 0 4px;font-weight:600">Extracted features`;
  if (ex.backend) {
    html += ` <span class="hint" style="display:inline">(LLM: ${ex.llm_used ? ex.backend : "off — transcript only"})</span>`;
  }
  html += `</p><div class="kv">`;
  html += kv("Caption", escapeHtml(inp.caption || "—"));
  html += kv("Duration", inp.duration_s != null ? `${Math.round(inp.duration_s)} s` : "—");
  if (ex.language) html += kv("Language", ex.language === "translate" ? "translated to English" : escapeHtml(ex.language));
  if (ex.transcript !== undefined) {
    const t = ex.transcript || "";
    html += kv("Transcript", t ? escapeHtml(t.slice(0, 240)) + (t.length > 240 ? "…" : "") : "(none — no speech / Whisper unavailable)");
  }
  if (ex.summary) html += kv("Summary", escapeHtml(ex.summary));
  if (ex.topic) html += kv("Topic", `<span class="chip">${escapeHtml(ex.topic)}</span> hook ${(ex.hook ?? 0).toFixed(2)} · cta ${(ex.cta ?? 0).toFixed(2)}`);
  if (ex.emotions) {
    const emo = Object.entries(ex.emotions).sort((a, b) => b[1] - a[1]).slice(0, 3)
      .map(([k, v]) => `${k} ${Number(v).toFixed(2)}`).join(", ");
    html += kv("Emotions", emo);
  }
  html += `</div>`;

  if (r.factors && r.factors.length) {
    html += `<p style="margin:12px 0 2px;font-weight:600">Main factors</p>`;
    html += r.factors.map((f) => `<div class="factor">${f.direction || ""} ${escapeHtml(f.note || f.factor || "")}</div>`).join("");
  }
  if (r.examples && r.examples.length) {
    html += `<p style="margin:12px 0 2px;font-weight:600">Similar past videos</p>`;
    html += r.examples.map((e) => `<div class="factor">${e.outcome === "hit" ? "hit" : "flop"} (similarity ${e.similarity}): ${escapeHtml(e.caption || "")}</div>`).join("");
  }
  if (r.warnings && r.warnings.length) html += `<div class="warn">${r.warnings.map(escapeHtml).join("<br>")}</div>`;
  if (r.what_model_cannot_know) html += `<div class="note">${escapeHtml(r.what_model_cannot_know)}</div>`;
  $("#out").innerHTML = html;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

"use strict";

window.QuickRDPage = (function () {
  const $ = (id) => document.getElementById(id);
  const S = {
    built: false, poll: null, rows: [], fsKind: null, mpMin: null, mpMax: null,
    logx: true, logy: false, psnr: false,
    show: { oldPresets: true, liveOldPresets: true, newPresets: true, jpeg: true, jp2: true, webp: true, jxl: true, avif: true, downsample: false, png: true },
  };
  const COLORS = { old: "#ff9b54", newer: "#ff3b41", jpeg: "#7d8cff", jp2: "#4fa3ff", webp: "#54c0ff", jxl: "#c084fc", avif: "#34d39a", downsample: "#e6e6e6", png: "#ef19ef" };
  const TOGGLES = [["oldPresets", "DB Old PBC Presets"], ["liveOldPresets", "Live Old PBC Presets"], ["newPresets", "Live New PBC Presets"], ["jpeg", "JPEG"], ["jp2", "JPEG2000"], ["webp", "WebP"], ["jxl", "JXL"], ["avif", "AVIF"], ["downsample", "Downsample"], ["png", "PNG"]];
  const P2D = {
    qc: { x: "bpp", y: "mse", xt: "bpp", yt: "MSE", title: "MSE vs Compression", note: "↙ lower-left is best" },
    qs: { x: "speed", y: "mse", xt: "seconds", yt: "MSE", title: "MSE vs Speed", note: "↙ lower-left is best" },
    cs: { x: "speed", y: "bpp", xt: "seconds", yt: "bpp", title: "Compression vs Speed", note: "↙ lower-left is best" },
    qcs: { x: "speed", y: "eff", xt: "seconds", yt: "efficiency 1/(bpp·MSE)", title: "Efficiency vs Speed", note: "↖ upper-left is best" },
  };
  const FS_STYLE = `
    .sw-plot-box{position:relative;}.sw-fs-btn{position:absolute;top:10px;right:10px;z-index:5;background:rgba(18,18,20,.72);border:1px solid var(--border2);color:var(--text);border-radius:7px;width:28px;height:28px;cursor:pointer;font-size:15px;display:flex;align-items:center;justify-content:center}.sw-fs-btn:hover{border-color:var(--red-dim);color:#fff;background:rgba(18,18,20,.9)}
    #qrd-fs{position:fixed;inset:0;z-index:1000;background:rgba(0,0,0,.82);backdrop-filter:blur(6px);display:flex;padding:24px}#qrd-fs[hidden]{display:none}.sw-fs-inner{margin:auto;width:96vw;height:92vh;background:var(--surface);border:1px solid var(--border);border-radius:16px;display:flex;flex-direction:column;padding:14px 16px;box-shadow:0 24px 70px rgba(0,0,0,.55)}.sw-fs-bar{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}.sw-fs-bar span{font-size:13px;font-weight:600;color:var(--text)}#qrd-fs-plot{flex:1;min-height:0}`;

  const fmt = (v, d = 4) => (v == null || isNaN(v)) ? "—" : (+v).toFixed(d);
  const psnr = (mse) => mse <= 0 ? 99 : 10 * Math.log10(255 * 255 / mse);
  const eff = (r) => (r.bpp > 0 && r.mse > 0) ? 1 / (r.bpp * r.mse) : 0;
  const yMetric = (r) => S.psnr ? psnr(r.mse) : r.mse;
  const codecKey = (c) => c === "JPEG2000" ? "jp2" : c.toLowerCase();
  const name = (r) => r.kind === "preset" ? `${r.source === "new" ? "New" : "Old"} ${r.preset}` : `${r.codec}${r.q != null ? " q" + r.q : ""}`;

  function onShow() { build(); refreshResults(); refreshStatus(); resizeSoon(); }
  function onHide() { stopPolling(); closeFs(); }

  function resizeSoon() {
    for (const delay of [0, 40, 160, 400]) {
      setTimeout(() => {
        for (const id of ["qrd-p-qc", "qrd-p-qs", "qrd-p-cs", "qrd-p-qcs", "qrd-fs-plot"]) {
          const el = $(id);
          if (el && el.data) Plotly.Plots.resize(el);
        }
      }, delay);
    }
  }

  function build() {
    if (S.built) return;
    const st = document.createElement("style"); st.textContent = FS_STYLE; document.head.appendChild(st);
    $("view-quick-rd").innerHTML = `
    <div class="sw-wrap">
      <div class="sw-card">
        <h3>Tuning database<span class="sub">loads old presets/codecs from pbc3_sweep.db</span></h3>
        <div class="sw-row" style="gap:16px">
          <label class="sw-btn">Upload database<input id="qrd-upload" type="file" accept=".db" hidden></label>
          <button id="qrd-download" class="sw-btn">Download tuning database</button>
        </div>
        <div id="qrd-dbmsg" class="sw-muted" style="margin-top:8px"></div>
      </div>
      <div class="sw-card">
        <h3>Quick RD benchmark<span class="sub">runs current 4 PBC3 presets over hpt_data/</span></h3>
        <div class="sw-row" style="gap:16px;align-items:end">
          <div class="sw-f" style="min-width:220px"><label>n trials: <span id="qrd-n-val">1</span></label><input id="qrd-n" type="range" min="1" max="10" step="1" value="1"></div>
          <label class="sw-muted" style="text-transform:none;letter-spacing:0"><input type="checkbox" id="qrd-run-old" checked> compare live old_pbc</label>
          <button id="qrd-run" class="sw-btn primary">Run quick RD benchmark</button><button id="qrd-stop" class="sw-btn" disabled>Stop</button>
        </div>
        <div id="qrd-status" class="sw-muted" style="margin-top:10px"></div><pre id="qrd-log" class="run-log"></pre>
      </div>
      <div class="sw-card">
        <h3>Chart data</h3>
        <div class="sw-row" style="gap:16px">
          <button id="qrd-refresh" class="sw-btn primary">Refresh</button>
          <div class="sw-f"><label>MP ≥</label><input id="qrd-mpmin" class="sw-input" type="number" step="0.1" placeholder="any"></div>
          <div class="sw-f"><label>MP ≤</label><input id="qrd-mpmax" class="sw-input" type="number" step="0.1" placeholder="any"></div>
          <button id="qrd-mpreset" class="sw-btn sm">Reset MP</button>
        </div>
        <div id="qrd-msg" class="sw-muted" style="margin-top:8px"></div>
      </div>
      <div class="sw-card">
        <h3>Tradeoffs<span class="sub">old DB vs current code</span></h3>
        <div class="sw-row" id="qrd-toggles" style="margin-bottom:10px"></div>
        <div class="sw-row" style="margin-bottom:10px"><span class="sw-muted">Scale:</span>
          <label class="sw-muted" style="text-transform:none;letter-spacing:0"><input type="checkbox" id="qrd-logx" checked> log x</label>
          <label class="sw-muted" style="text-transform:none;letter-spacing:0"><input type="checkbox" id="qrd-logy"> log y</label>
          <label class="sw-muted" style="text-transform:none;letter-spacing:0"><input type="checkbox" id="qrd-psnr"> PSNR</label>
        </div>
        <div class="sw-plots">
          <div class="sw-plot-box"><button class="sw-fs-btn" data-fs="qc">⤢</button><div id="qrd-p-qc" class="sw-plot"></div></div>
          <div class="sw-plot-box"><button class="sw-fs-btn" data-fs="qs">⤢</button><div id="qrd-p-qs" class="sw-plot"></div></div>
          <div class="sw-plot-box"><button class="sw-fs-btn" data-fs="cs">⤢</button><div id="qrd-p-cs" class="sw-plot"></div></div>
          <div class="sw-plot-box"><button class="sw-fs-btn" data-fs="qcs">⤢</button><div id="qrd-p-qcs" class="sw-plot"></div></div>
        </div>
      </div>
      <div id="qrd-fs" hidden><div class="sw-fs-inner"><div class="sw-fs-bar"><span id="qrd-fs-title"></span><button id="qrd-fs-close" class="sw-btn sm">Close ✕</button></div><div id="qrd-fs-plot"></div></div></div>
    </div>`;
    buildToggles();
    $("qrd-upload").onchange = e => e.target.files[0] && uploadDb(e.target.files[0]);
    $("qrd-download").onclick = () => window.open("/api/sweeps/download_db", "_blank");
    $("qrd-n").oninput = e => $("qrd-n-val").textContent = e.target.value;
    $("qrd-run").onclick = run;
    $("qrd-stop").onclick = stop;
    $("qrd-refresh").onclick = refreshResults;
    ["qrd-mpmin", "qrd-mpmax"].forEach(id => $(id).addEventListener("change", refreshResults));
    $("qrd-mpreset").onclick = () => { $("qrd-mpmin").value = ""; $("qrd-mpmax").value = ""; refreshResults(); };
    $("qrd-logx").onchange = e => { S.logx = e.target.checked; render(); };
    $("qrd-logy").onchange = e => { S.logy = e.target.checked; render(); };
    $("qrd-psnr").onchange = e => { S.psnr = e.target.checked; render(); };
    $("view-quick-rd").querySelectorAll(".sw-fs-btn").forEach(b => b.onclick = () => openFs(b.dataset.fs));
    $("qrd-fs-close").onclick = closeFs;
    document.addEventListener("keydown", e => { if (e.key === "Escape" && !$('qrd-fs').hidden) closeFs(); });
    S.built = true;
  }

  function buildToggles() {
    $("qrd-toggles").innerHTML = `<span class="sw-muted">Show:</span>` + TOGGLES.map(([k, l]) => `<label class="sw-muted" style="text-transform:none;letter-spacing:0"><input type="checkbox" data-show="${k}" ${S.show[k] ? "checked" : ""}> ${l}</label>`).join("");
    $("qrd-toggles").querySelectorAll("input").forEach(el => el.onchange = () => { S.show[el.dataset.show] = el.checked; render(); });
  }

  async function uploadDb(file) { const fd = new FormData(); fd.append("file", file); const d = await (await fetch("/api/sweeps/run/upload_db", { method: "POST", body: fd })).json(); $("qrd-dbmsg").textContent = d.error || "Database loaded."; refreshResults(); }
  async function run() {
    const body = { n_trials: +$("qrd-n").value, run_old: $("qrd-run-old").checked };
    const d = await (await fetch("/api/quick_rd/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
    if (d.error) return alert(d.error);
    startPolling(); refreshStatus();
  }
  async function stop() { await fetch("/api/quick_rd/stop", { method: "POST" }); refreshStatus(); }
  function startPolling() { if (!S.poll) S.poll = setInterval(refreshStatus, 2000); }
  function stopPolling() { if (S.poll) { clearInterval(S.poll); S.poll = null; } }

  async function refreshStatus() {
    if (!S.built) return;
    const s = await (await fetch("/api/quick_rd/status")).json();
    $("qrd-run").disabled = s.running; $("qrd-stop").disabled = !s.running;
    $("qrd-status").innerHTML = s.running ? `Running · ${s.done}/${s.total || 4} · ${s.current || ""}` : (s.error ? `Error: ${s.error}` : "Idle.");
    $("qrd-log").textContent = (s.log || []).slice(-30).join("\n");
    if (s.running) { startPolling(); refreshResults(); } else { stopPolling(); refreshResults(); }
  }

  async function refreshResults() {
    const qs = new URLSearchParams();
    if ($("qrd-mpmin").value !== "") qs.set("mp_min", $("qrd-mpmin").value);
    if ($("qrd-mpmax").value !== "") qs.set("mp_max", $("qrd-mpmax").value);
    const d = await (await fetch("/api/quick_rd/results?" + qs.toString())).json();
    S.rows = (d.rows || []).map(r => ({ ...r, eff: eff(r) })).filter(r => r.speed != null && r.bpp != null && r.mse != null);
    $("qrd-msg").textContent = `${S.rows.length} points · old DB ${d.old_db_exists ? "loaded" : "missing"} · new results ${d.new_exists ? "loaded" : "missing"}`;
    render();
  }

  function visibleRows() {
    return S.rows.filter(r => {
      if (r.kind === "preset") {
        if (r.source === "new") return S.show.newPresets;
        if (r.source === "old_live") return S.show.liveOldPresets;
        return S.show.oldPresets;
      }
      return S.show[codecKey(r.codec)];
    });
  }
  function traceFor(rows, label, color, symbol) {
    return { x: rows.map(r => r._x), y: rows.map(r => r._y), mode: "markers+lines", type: "scatter", name: label, marker: { color, size: 9, symbol }, line: { color, width: 1.5 }, text: rows.map(r => `${name(r)}<br>MSE ${fmt(r.mse, 3)} · PSNR ${fmt(psnr(r.mse), 2)}<br>bpp ${fmt(r.bpp, 5)} · ${fmt(r.speed, 3)}s`), hovertemplate: "%{text}<extra></extra>" };
  }
  function renderOne(kind, target) {
    if (!window.Plotly) return;
    const def = P2D[kind];
    const rows = visibleRows().map(r => ({ ...r, _x: r[def.x], _y: def.y === "mse" ? yMetric(r) : r[def.y] }));
    const traces = [];
    const old = rows.filter(r => r.kind === "preset" && r.source === "old"); if (old.length) traces.push(traceFor(old, "DB Old PBC Presets", COLORS.old, "diamond"));
    const liveOld = rows.filter(r => r.kind === "preset" && r.source === "old_live"); if (liveOld.length) traces.push(traceFor(liveOld, "Live Old PBC Presets", "#ffb86b", "square"));
    const newer = rows.filter(r => r.kind === "preset" && r.source === "new"); if (newer.length) traces.push(traceFor(newer, "Live New PBC Presets", COLORS.newer, "circle"));
    for (const [key, label, color] of [["JPEG", "JPEG", COLORS.jpeg], ["JPEG2000", "JPEG2000", COLORS.jp2], ["WEBP", "WebP", COLORS.webp], ["JXL", "JXL", COLORS.jxl], ["AVIF", "AVIF", COLORS.avif], ["DOWNSAMPLE", "Downsample", COLORS.downsample], ["PNG", "PNG", COLORS.png]]) {
      const pts = rows.filter(r => r.codec === key).sort((a, b) => (a.q || 0) - (b.q || 0));
      if (pts.length) traces.push(traceFor(pts, label, color, "diamond"));
    }
    const yt = def.y === "mse" && S.psnr ? "PSNR" : def.yt;
    const title = def.y === "mse" && S.psnr ? def.title.replace("MSE", "PSNR") : def.title;
    const el = $(target);
    if (!el || el.offsetWidth < 50) return;
    Plotly.react(el, traces, { autosize: true, title: { text: `${title} <span style="font-size:10px;color:#888">${def.note}</span>`, font: { size: 13, color: "#ddd" } }, paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { color: "#cfcfcf", size: 11 }, margin: { l: 56, r: 16, t: 42, b: 44 }, xaxis: { title: def.xt, type: S.logx ? "log" : "linear", tickformat: S.logx ? "~g" : undefined, gridcolor: "rgba(255,255,255,.08)", zeroline: false, automargin: true }, yaxis: { title: yt, type: S.logy ? "log" : "linear", tickformat: S.logy ? "~g" : undefined, autorange: def.y === "mse" && S.psnr ? true : undefined, gridcolor: "rgba(255,255,255,.08)", zeroline: false, automargin: true }, legend: { orientation: "h" } }, { responsive: true, displaylogo: false, displayModeBar: false });
    requestAnimationFrame(() => Plotly.Plots.resize(el));
  }
  function render() { if (!S.built || $("view-quick-rd").hidden) return; for (const k of Object.keys(P2D)) renderOne(k, `qrd-p-${k}`); if (S.fsKind) renderFs(); resizeSoon(); }
  function openFs(k) { S.fsKind = k; $("qrd-fs").hidden = false; renderFs(); }
  function closeFs() { if (!S.built || $("qrd-fs").hidden) return; S.fsKind = null; try { Plotly.purge("qrd-fs-plot"); } catch {} $("qrd-fs").hidden = true; }
  function renderFs() { if (!S.fsKind) return; $("qrd-fs-title").textContent = P2D[S.fsKind].title; renderOne(S.fsKind, "qrd-fs-plot"); setTimeout(() => Plotly.Plots.resize($("qrd-fs-plot")), 30); }

  return { onShow, onHide, refreshResults };
})();

"use strict";

window.BenchmarkPage = (function () {
  const $ = (id) => document.getElementById(id);
  const CODECS = ["PBC3", "JPEG", "JPEG2000", "WEBP", "JXL", "AVIF", "PNG"];
  const COLORS = { PBC3: "#ff4d3d", JPEG: "#7d8cff", JPEG2000: "#4fa3ff", WEBP: "#54c0ff", JXL: "#c084fc", AVIF: "#34d39a", PNG: "#ef19ef" };
  const S = { built: false, poll: null, points: [], filtered: [], show: Object.fromEntries(CODECS.map(c => [c, true])), logx: false, logy: false, fsKind: null, resultsRequest: 0 };

  const FS_STYLE = `
    .sw-plot-box{position:relative;}
    .sw-fs-btn{position:absolute;top:10px;right:10px;z-index:5;background:rgba(18,18,20,.72);
      border:1px solid var(--border2);color:var(--text);border-radius:7px;width:28px;height:28px;
      cursor:pointer;font-size:15px;line-height:1;display:flex;align-items:center;justify-content:center;transition:.15s;}
    .sw-fs-btn:hover{border-color:var(--red-dim);color:#fff;background:rgba(18,18,20,.9);}
    #bm-fs{position:fixed;inset:0;z-index:1000;background:rgba(0,0,0,.82);backdrop-filter:blur(6px);
      display:flex;padding:24px;animation:swfs .14s ease;}
    #bm-fs[hidden]{display:none;}
    @keyframes swfs{from{opacity:0}to{opacity:1}}
    .sw-fs-inner{margin:auto;width:96vw;height:92vh;background:var(--surface);border:1px solid var(--border);
      border-radius:16px;display:flex;flex-direction:column;padding:14px 16px;box-shadow:0 24px 70px rgba(0,0,0,.55);}
    .sw-fs-bar{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}
    .sw-fs-bar span{font-size:13px;font-weight:600;color:var(--text);}
    #bm-fs-plot{flex:1;min-height:0;}`;

  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const psnr = (mse) => mse <= 0 ? 99 : 10 * Math.log10(255 * 255 / mse);
  const f3 = (v) => Number.isFinite(+v) ? (+v).toFixed(3) : "—";
  const f5 = (v) => Number.isFinite(+v) ? (+v).toFixed(5) : "—";
  const codecLabel = (p) => p.codec === "PBC3" ? `PBC3 ${p.variant}` : (p.codec === "PNG" ? "PNG" : `${p.codec} ${p.variant}`);

  function onShow() { build(); refreshResults(); refreshStatus(); }
  function onHide() { stopPolling(); closeFs(); }

  function build() {
    if (S.built) return;
    const st = document.createElement("style");
    st.textContent = FS_STYLE;
    document.head.appendChild(st);

    $("view-benchmark").innerHTML = `
    <div class="sw-wrap">
      <div class="sw-card">
        <h3>Benchmark Database<span class="sub">pbc3_benchmark.db on the Space</span></h3>
        <div class="sw-row">
          <label class="sw-btn">Upload benchmark data<input id="bm-upload" type="file" accept=".db" hidden></label>
          <button id="bm-download" class="sw-btn">Download benchmark data</button>
          <button id="bm-reset-pbc" class="sw-btn danger">Reset PBC3</button>
        </div>
        <div id="bm-dbmsg" class="sw-muted" style="margin-top:8px"></div>
      </div>

      <div class="sw-card">
        <h3>Run benchmark<span class="sub">fixed dataset from hpt_data/</span></h3>
        <div id="bm-dataset" class="sw-muted" style="margin-bottom:12px">loading dataset…</div>
        <div class="sw-f"><label>Codecs</label><div id="bm-codecs" class="sw-row"></div></div>
        <div class="sw-row" style="margin-top:14px;gap:16px">
          <div class="sw-f" style="min-width:220px"><label>n trials: <span id="bm-n-val">1</span></label><input id="bm-n" type="range" min="1" max="10" step="1" value="1"></div>
          <button id="bm-run" class="sw-btn primary">Run benchmark</button>
          <button id="bm-stop" class="sw-btn" disabled>Stop benchmark</button>
        </div>
      </div>

      <div class="sw-card">
        <h3>Status<span id="bm-badge" class="sub">idle</span></h3>
        <div id="bm-status" class="sw-muted"></div>
        <pre id="bm-log" class="run-log"></pre>
      </div>

      <div style="height:12px"></div>
      <h3 class="bm-section-title">Benchmark Database Results</h3>

      <div class="sw-card">
        <h3>Run and dataset range</h3>
        <div class="sw-f"><label>Run</label><select id="bm-run-select" class="sw-input"><option value="latest">Latest complete run (coverage inferred)</option><option value="all">All historical runs — composite</option></select></div>
        <div id="bm-run-info" class="sw-muted" style="margin:8px 0"></div>
        <div class="sw-row" style="gap:16px">
          <div class="sw-f"><label>MP ≥</label><input id="bm-mpmin" class="sw-input" type="number" step="0.1" placeholder="any"></div>
          <div class="sw-f"><label>MP ≤</label><input id="bm-mpmax" class="sw-input" type="number" step="0.1" placeholder="any"></div>
          <button id="bm-mpreset" class="sw-btn sm">Reset MP</button>
          <button id="bm-refresh" class="sw-btn sm">Refresh</button>
        </div>
      </div>

      <div class="sw-card">
        <h3>Filters</h3>
        <div class="sw-filters">
          <div class="sw-f"><label>PSNR ≥</label><input id="bm-psnrmin" class="sw-input" type="number" step="0.1" placeholder="any"></div>
          <div class="sw-f"><label>MSE ≤</label><input id="bm-msemax" class="sw-input" type="number" step="1" placeholder="any"></div>
          <div class="sw-f"><label>BPP ≤</label><input id="bm-bppmax" class="sw-input" type="number" step="0.01" placeholder="any"></div>
          <div class="sw-f"><label>encode seconds ≤</label><input id="bm-encmax" class="sw-input" type="number" step="0.1" placeholder="any"></div>
          <div class="sw-f"><label>decode seconds ≤</label><input id="bm-decmax" class="sw-input" type="number" step="0.1" placeholder="any"></div>
        </div>
        <div style="height:12px"></div>
        <div class="sw-f"><label>Show codecs</label><div id="bm-show-codecs" class="sw-row"></div></div>
        <div class="sw-row" style="margin-top:12px">
          <label class="sw-muted" style="text-transform:none;letter-spacing:0"><input type="checkbox" id="bm-logx"> log x</label>
          <label class="sw-muted" style="text-transform:none;letter-spacing:0"><input type="checkbox" id="bm-logy"> log y</label>
        </div>
      </div>

      <div class="sw-card">
        <h3>Efficiency<span class="sub">PSNR / bpp vs seconds</span></h3>
        <div class="sw-plots">
          <div class="sw-plot-box"><button class="sw-fs-btn" data-bm-fs="eff-enc" title="Fullscreen">⤢</button><div id="bm-p-eff" class="sw-plot"></div></div>
          <div class="sw-plot-box"><button class="sw-fs-btn" data-bm-fs="eff-dec" title="Fullscreen">⤢</button><div id="bm-p-eff-dec" class="sw-plot"></div></div>
        </div>
      </div>

      <div class="sw-card">
        <h3>Speed averages<span class="sub">averaged over visible filtered points</span></h3>
        <div class="sw-plots">
          <div class="sw-plot-box"><button class="sw-fs-btn" data-bm-fs="bar-enc" title="Fullscreen">⤢</button><div id="bm-p-enc" class="sw-plot"></div></div>
          <div class="sw-plot-box"><button class="sw-fs-btn" data-bm-fs="bar-dec" title="Fullscreen">⤢</button><div id="bm-p-dec" class="sw-plot"></div></div>
        </div>
      </div>

      <div id="bm-fs" hidden>
        <div class="sw-fs-inner">
          <div class="sw-fs-bar"><span id="bm-fs-title"></span><button id="bm-fs-close" class="sw-btn sm">Close ✕</button></div>
          <div id="bm-fs-plot"></div>
        </div>
      </div>
    </div>`;

    renderCodecChecks("bm-codecs", Object.fromEntries(CODECS.map(c => [c, true])));
    renderCodecChecks("bm-show-codecs", S.show, () => { applyFilters(); renderPlots(); });

    $("bm-upload").onchange = (e) => e.target.files[0] && uploadDb(e.target.files[0]);
    $("bm-download").onclick = () => window.open("/api/benchmark/download_db", "_blank");
    $("bm-reset-pbc").onclick = resetPbcRows;
    $("bm-n").oninput = (e) => $("bm-n-val").textContent = e.target.value;
    $("bm-run").onclick = start;
    $("bm-stop").onclick = stop;
    ["bm-mpmin", "bm-mpmax"].forEach(id => $(id).addEventListener("change", refreshResults));
    $("bm-mpreset").onclick = () => { $("bm-mpmin").value = ""; $("bm-mpmax").value = ""; refreshResults(); };
    $("bm-refresh").onclick = refreshResults;
    ["bm-psnrmin", "bm-msemax", "bm-bppmax", "bm-encmax", "bm-decmax"].forEach(id => $(id).addEventListener("input", () => { applyFilters(); renderPlots(); }));
    $("bm-logx").onchange = (e) => { S.logx = e.target.checked; renderPlots(); };
    $("bm-logy").onchange = (e) => { S.logy = e.target.checked; renderPlots(); };
    $("view-benchmark").querySelectorAll(".sw-fs-btn").forEach(b => b.onclick = () => openFs(b.dataset.bmFs));
    $("bm-fs-close").onclick = closeFs;
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$('bm-fs').hidden) closeFs(); });

    fetch("/api/benchmark/dataset").then(r => r.json()).then(ds => {
      $("bm-dataset").innerHTML = ds.count
        ? `<b>${ds.count}</b> images · ` + ds.images.map(i => `${esc(i.name)} (${i.mp} MP)`).join(" · ")
        : "No images found in hpt_data/.";
    });
    S.built = true;
    $("bm-run-select").onchange = refreshResults;
  }

  function renderCodecChecks(id, state, onchange) {
    $(id).innerHTML = CODECS.map(c => `<label class="sw-muted" style="text-transform:none;letter-spacing:0;color:${COLORS[c]}"><input type="checkbox" data-codec="${c}" ${state[c] ? "checked" : ""}> ${c}</label>`).join("");
    $(id).querySelectorAll("input").forEach(el => el.onchange = () => {
      state[el.dataset.codec] = el.checked;
      if (onchange) onchange();
    });
  }

  function selectedCodecs() {
    return [...$("bm-codecs").querySelectorAll("input:checked")].map(x => x.dataset.codec);
  }

  async function start() {
    const body = { codecs: selectedCodecs(), n_trials: +$("bm-n").value };
    const d = await (await fetch("/api/benchmark/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
    if (d.error) return window.toast ? window.toast(d.error) : alert(d.error);
    startPolling();
    refreshStatus();
  }

  async function stop() { await fetch("/api/benchmark/stop", { method: "POST" }); refreshStatus(); }

  async function uploadDb(file) {
    const fd = new FormData(); fd.append("file", file);
    const d = await (await fetch("/api/benchmark/upload_db", { method: "POST", body: fd })).json();
    $("bm-dbmsg").textContent = d.error || "Benchmark database loaded.";
    refreshResults();
  }

  async function resetPbcRows() {
    if (!confirm("Reset all PBC3 rows from the benchmark database? Other codec rows will be kept.")) return;
    const r = await fetch("/api/benchmark/reset_pbc", { method: "POST" });
    const d = await r.json();
    $("bm-dbmsg").textContent = d.error || `Deleted ${d.deleted || 0} PBC3 benchmark row(s).`;
    refreshResults();
  }

  function startPolling() { if (!S.poll) S.poll = setInterval(refreshStatus, 2000); }
  function stopPolling() { if (S.poll) { clearInterval(S.poll); S.poll = null; } }

  async function refreshStatus() {
    if (!S.built) return;
    let s; try { s = await (await fetch("/api/benchmark/status")).json(); } catch { return; }
    $("bm-badge").textContent = s.running ? "running" : (s.error ? "error" : "idle");
    $("bm-badge").style.color = s.running ? "var(--red)" : "";
    $("bm-run").disabled = s.running;
    $("bm-stop").disabled = !s.running;
    const prog = s.total ? ` · ${s.done}/${s.total}` : "";
    $("bm-status").innerHTML = (s.running ? `Benchmarking${prog}` : (s.error ? `Error: ${esc(s.error)}` : "Idle.")) +
      (s.current ? `<br><span class="sw-muted">current: ${esc(JSON.stringify(s.current))}</span>` : "");
    $("bm-log").textContent = (s.log || []).slice(-40).join("\n");
    if (s.running) { startPolling(); refreshResults(); }
    else { stopPolling(); refreshResults(); }
  }

  async function refreshResults() {
    if (!S.built) return;
    const request = ++S.resultsRequest;
    const qs = new URLSearchParams();
    const selection = $("bm-run-select").value;
    qs.set("run_id", selection);
    if ($("bm-mpmin").value !== "") qs.set("mp_min", $("bm-mpmin").value);
    if ($("bm-mpmax").value !== "") qs.set("mp_max", $("bm-mpmax").value);
    const d = await (await fetch("/api/benchmark/results?" + qs.toString())).json();
    if (request !== S.resultsRequest) return;
    const selector = $("bm-run-select");
    const options = [["latest", "Latest complete run (coverage inferred)"], ["all", "All historical runs — composite"], ...(d.runs || []).map(r => [r.run_id, `${r.run_id} · ${r.status} · ${r.codecs.join(", ")}`])];
    if (!options.some(([value]) => value === selection)) options.push([selection, `${selection} · missing run`]);
    if (JSON.stringify([...selector.options].map(o => [o.value, o.text])) !== JSON.stringify(options)) {
      selector.replaceChildren();
      for (const [value, label] of options) selector.add(new Option(label, value));
    }
    selector.value = selection;
    const run = (d.runs || []).find(r => r.run_id === d.selected_run);
    $("bm-run-info").textContent = (run
      ? `${run.run_id} · ${new Date(run.first_ts * 1000).toISOString()} · ${run.status} · ${run.trials} trial(s), ${run.images} images, ${run.count} total samples · ${d.count} selected samples. `
      : `${d.selection_status || "No runs"} · ${d.count || 0} selected samples. `) + (d.coverage_note || "") + (selection === "all" ? " Historical codecs may come from different runs and CPU conditions." : "");
    S.points = (d.points || []).map(p => ({ ...p, psnr: +p.psnr || psnr(+p.mse), eff: (+p.psnr || psnr(+p.mse)) / Math.max(+p.bpp, 1e-9) }));
    applyFilters();
    renderPlots();
  }

  function num(id) { const v = $(id).value; return v === "" ? NaN : +v; }
  function applyFilters() {
    const psnrMin = num("bm-psnrmin"), mseMax = num("bm-msemax"), bppMax = num("bm-bppmax"), encMax = num("bm-encmax"), decMax = num("bm-decmax");
    S.filtered = S.points.filter(p => S.show[p.codec])
      .filter(p => !Number.isFinite(psnrMin) || p.psnr >= psnrMin)
      .filter(p => !Number.isFinite(mseMax) || p.mse <= mseMax)
      .filter(p => !Number.isFinite(bppMax) || p.bpp <= bppMax)
      .filter(p => !Number.isFinite(encMax) || p.encode_seconds <= encMax)
      .filter(p => !Number.isFinite(decMax) || p.decode_seconds <= decMax);
  }

  function scatterTrace(codec, speedKey = "encode_seconds") {
    const pts = S.filtered.filter(p => p.codec === codec).sort((a, b) => a.bpp - b.bpp);
    return {
      x: pts.map(p => p[speedKey]), y: pts.map(p => p.eff), mode: "markers+lines", type: "scatter", name: codec,
      marker: { color: COLORS[codec], size: codec === "PBC3" ? 10 : 8 }, line: { color: COLORS[codec], width: 1.5 },
      text: pts.map(p => `${codecLabel(p)}<br>PSNR ${f3(p.psnr)} · bpp ${f5(p.bpp)}<br>encode ${f3(p.encode_seconds)}s · decode ${f3(p.decode_seconds)}s`),
      hovertemplate: "%{text}<extra></extra>",
    };
  }

  function avgByCodec(key) {
    return CODECS.map(codec => {
      const pts = S.filtered.filter(p => p.codec === codec);
      const n = pts.reduce((a, p) => a + (+p.count || 1), 0);
      const v = n ? pts.reduce((a, p) => a + (+p[key]) * (+p.count || 1), 0) / n : 0;
      return { codec, v, n };
    }).filter(x => x.n > 0);
  }

  function barTrace(rows, key, title) {
    const max = Math.max(...rows.map(r => r.v), 0);
    return [{
      x: rows.map(r => r.codec), y: rows.map(r => r.v), type: "bar",
      marker: { color: rows.map(r => COLORS[r.codec]) },
      text: rows.map(r => `${f3(r.v)}s`), textposition: "outside",
      textangle: 0, cliponaxis: false,
      hovertext: rows.map(r => `${r.codec}<br>${title}: ${f3(r.v)}s`), hovertemplate: "%{hovertext}<extra></extra>",
    }];
  }

  function layout(title, x, y) {
    return {
      title: { text: title, font: { size: 13, color: "#ddd" } }, paper_bgcolor: "transparent", plot_bgcolor: "transparent",
      font: { color: "#cfcfcf", size: 11 }, margin: { l: 56, r: 16, t: 42, b: 44 },
      xaxis: { title: x, type: S.logx ? "log" : "linear", tickformat: S.logx ? "~g" : undefined, gridcolor: "rgba(255,255,255,.08)", zeroline: false },
      yaxis: { title: y, type: S.logy ? "log" : "linear", tickformat: S.logy ? "~g" : undefined, gridcolor: "rgba(255,255,255,.08)", zeroline: false },
      legend: { orientation: "h" },
    };
  }

  function barLayout(title, y) {
    return {
      title: { text: title, font: { size: 13, color: "#ddd" } }, paper_bgcolor: "transparent", plot_bgcolor: "transparent",
      font: { color: "#cfcfcf", size: 11 }, margin: { l: 56, r: 16, t: 42, b: 52 },
      xaxis: { gridcolor: "rgba(255,255,255,.08)" },
      yaxis: { title: y, type: S.logy ? "log" : "linear", tickformat: S.logy ? "~g" : undefined, gridcolor: "rgba(255,255,255,.08)", zeroline: false },
      uniformtext: { mode: "show", minsize: 9 },
    };
  }

  function visibleCodecs() {
    return CODECS.filter(c => S.show[c] && S.filtered.some(p => p.codec === c));
  }

  function plotEfficiency(elId, speedKey, title, xt) {
    Plotly.react(elId, visibleCodecs().map(c => scatterTrace(c, speedKey)), layout(title, xt, "PSNR / bpp"), { responsive: true, displaylogo: false, displayModeBar: false });
  }

  function plotBar(elId, key, title) {
    Plotly.react(elId, barTrace(avgByCodec(key), key, title), barLayout(title + " Speed", "seconds"), { responsive: true, displaylogo: false, displayModeBar: false });
  }

  function renderPlots() {
    if (!window.Plotly || !S.built) return;
    plotEfficiency("bm-p-eff", "encode_seconds", "Efficiency vs Encode Speed", "encode seconds");
    plotEfficiency("bm-p-eff-dec", "decode_seconds", "Efficiency vs Decode Speed", "decode seconds");
    plotBar("bm-p-enc", "encode_seconds", "Encode");
    plotBar("bm-p-dec", "decode_seconds", "Decode");
    if (S.fsKind) renderFsPlot();
  }

  function openFs(kind) {
    S.fsKind = kind;
    $("bm-fs").hidden = false;
    renderFsPlot();
    setTimeout(resizeFs, 30);
  }

  function closeFs() {
    if (!S.built || !$('bm-fs') || $('bm-fs').hidden) return;
    S.fsKind = null;
    try { Plotly.purge("bm-fs-plot"); } catch (_) {}
    $("bm-fs").hidden = true;
  }

  function renderFsPlot() {
    if (!S.fsKind) return;
    const titles = { "eff-enc": "Efficiency vs Encode Speed", "eff-dec": "Efficiency vs Decode Speed", "bar-enc": "Encode Speed", "bar-dec": "Decode Speed" };
    $("bm-fs-title").textContent = titles[S.fsKind] || "Benchmark plot";
    if (S.fsKind === "eff-enc") plotEfficiency("bm-fs-plot", "encode_seconds", "Efficiency vs Encode Speed", "encode seconds");
    else if (S.fsKind === "eff-dec") plotEfficiency("bm-fs-plot", "decode_seconds", "Efficiency vs Decode Speed", "decode seconds");
    else if (S.fsKind === "bar-enc") plotBar("bm-fs-plot", "encode_seconds", "Encode");
    else if (S.fsKind === "bar-dec") plotBar("bm-fs-plot", "decode_seconds", "Decode");
    resizeFs();
  }

  function resizeFs() {
    const el = $("bm-fs-plot");
    if (el && el.data && !$('bm-fs').hidden) Plotly.Plots.resize(el);
  }

  return { onShow, onHide, refreshResults };
})();

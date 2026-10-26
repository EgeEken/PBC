(() => {
  const $ = (selector) => document.querySelector(selector);
  const runRoot = $("#view-train");
  const checkpointRoot = $("#view-train-checkpoints");
  if (!runRoot || !checkpointRoot) return;

  let poll = null;
  let checkpoints = [];
  let activeTab = "run";
  let latestStatus = {};
  let customOutputName = false;

  const presets = [
    ["compression", "Compression"],
    ["balanced", "Balanced"],
    ["quality", "Quality"],
    ["high_quality", "High quality"],
  ];

  const esc = (value) => String(value ?? "").replace(/[&<>\"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[c]));
  const api = async (path, options) => {
    const response = await fetch(path, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
  };
  const metricColor = (value, baseline, observed) => {
    if (value == null || baseline == null) return "";
    const values = observed.filter((item) => Number.isFinite(Number(item)));
    const best = Math.min(Number(baseline), ...(values.length ? values : [Number(baseline)]));
    const worst = Math.max(Number(baseline), ...(values.length ? values : [Number(baseline)]));
    const current = Number(value);
    if (current <= Number(baseline)) {
      const span = Number(baseline) - best;
      const amount = span > 0 ? Math.min(1, (Number(baseline) - current) / span) : 0;
      return `rgb(${Math.round(250 - 198 * amount)}, ${Math.round(204 + 31 * amount)}, ${Math.round(21 + 79 * amount)})`;
    }
    const span = worst - Number(baseline);
    const amount = span > 0 ? Math.min(1, (current - Number(baseline)) / span) : 0;
    return `rgb(${Math.round(250 - 11 * amount)}, ${Math.round(204 - 136 * amount)}, ${Math.round(21 + 47 * amount)})`;
  };
  const metricValue = (metric, value, baseline, observed, digits) => {
    const color = metricColor(value, baseline, observed);
    return color ? `<span style="color:${color}" title="Baseline ${fmt(baseline, digits)}">${fmt(value, digits)}</span>` : fmt(value, digits);
  };
  const download = (path) => window.open(`/api/train/download_checkpoint?path=${encodeURIComponent(path)}`, "_blank");
  const fmt = (value, digits = 4) => value == null ? "—" : Number(value).toFixed(digits);

  function showLab() {
    ["landing", "app", "sweep-lab"].forEach((id) => { const el = $(`#${id}`); if (el) el.hidden = true; });
    $("#train-lab").hidden = false;
    renderRun();
    loadCheckpoints();
    refresh();
  }

  function showLanding() {
    $("#train-lab").hidden = true;
    $("#landing").hidden = false;
    if (poll) clearInterval(poll);
    poll = null;
  }

  $("#enter-train")?.addEventListener("click", showLab);
  $("#train-back")?.addEventListener("click", showLanding);
  document.querySelectorAll("[data-tr-tab]").forEach((tab) => tab.addEventListener("click", () => {
    activeTab = tab.dataset.trTab;
    document.querySelectorAll("[data-tr-tab]").forEach((item) => item.classList.toggle("active", item.dataset.trTab === activeTab));
    runRoot.hidden = activeTab !== "run";
    checkpointRoot.hidden = activeTab !== "checkpoints";
    if (activeTab === "checkpoints") renderCheckpoints();
  }));

  function renderRun() {
    customOutputName = false;
    runRoot.innerHTML = `
      <div class="params">
        <div class="params-head"><span class="field-label">Long-horizon RL training</span><span class="card-date">hpt_data → hpt_data_val</span></div>
        <p class="card-date">Runs the current filler trainer in a resumable subprocess. Start the same output again to continue from its optimizer and RNG checkpoint.</p>
        <div class="tr-fields">
          <div class="tr-f full"><span>Preset policies</span><div class="tr-presets">${presets.map(([id, label], i) => `<label><input type="checkbox" data-tr-preset="${id}" ${i === 3 ? "checked" : ""}> ${label}</label>`).join("")}</div></div>
          <label class="tr-f"><span>Additional epochs</span><input id="tr-epochs" type="number" min="1" step="1" value="100"></label>
          <label class="tr-f"><span>Batch size</span><input id="tr-batch" type="number" min="1" step="1" value="4"></label>
          <label class="tr-f"><span>Initial model</span><select id="tr-init"></select><input id="tr-upload" type="file" accept=".npz"></label>
          <label class="tr-f"><span>Output name</span><input id="tr-output" value="rl_high_quality_top2.npz"></label>
          <label class="tr-f"><span>Rate weight</span><input id="tr-rate" type="number" step="0.05" value="1.5"></label>
          <label class="tr-f"><span>Speed weight</span><input id="tr-speed" type="number" step="0.05" value="0.15"></label>
          <label class="tr-f"><span>Temperature</span><input id="tr-temp" type="number" step="0.05" value="0.9"></label>
          <label class="tr-f"><span>Entropy weight</span><input id="tr-entropy" type="number" step="0.0005" value="0.003"></label>
          <label class="tr-f"><span>KL weight</span><input id="tr-kl" type="number" step="0.001" value="0.015"></label>
        </div>
        <div class="tr-actions"><button id="tr-start" class="primary-btn">Start / resume</button><button id="tr-stop" class="hold-btn">Stop</button><button id="tr-refresh" class="hold-btn">Refresh</button><button id="tr-log" class="hold-btn">Download log</button><button id="tr-copy-results" class="hold-btn">Copy results</button><button id="tr-download-results" class="hold-btn">Download results</button><label class="tr-check tr-live"><input id="tr-pause" type="checkbox"> Pause live table/log updates</label><label class="tr-check"><input id="tr-resume" type="checkbox" checked> Resume existing output state</label></div>
      </div>
      <div class="params"><div class="params-head"><span class="field-label">Run status</span></div><div id="tr-status" class="tr-ck">Loading…</div><pre id="tr-log-tail" class="tr-log-tail"></pre></div>
      <div class="params"><div class="params-head"><span class="field-label">Learning graphics</span></div><div class="tr-chart-grid"><div id="tr-chart-reward"></div><div id="tr-chart-metrics"></div><div id="tr-chart-work"></div><div id="tr-chart-train"></div><div id="tr-chart-speed"></div><div id="tr-chart-rd" class="tr-chart-wide"></div></div></div>
      <div class="params"><div class="params-head"><span class="field-label">Best checkpoints</span></div><div id="tr-best-checkpoints"></div></div>
      <div class="params"><div class="params-head"><span class="field-label">Validation history</span><label class="tr-sort">Sort by <select id="tr-sort"><option value="epoch">Epoch</option><option value="reward">Per-image reward</option><option value="aggregate_score">Aggregate score</option><option value="mse">Validation MSE</option><option value="bpp">Validation bpp</option><option value="work">Validation grid coefficients</option><option value="encode_seconds">Validation encode seconds</option><option value="train_reward">Training reward</option><option value="train_mse">Training MSE</option><option value="train_bpp">Training bpp</option><option value="train_work">Training grid coefficients</option><option value="train_encode_seconds">Training encode seconds</option></select><button id="tr-sort-dir" class="hold-btn sm">↓</button></label></div><div id="tr-history"></div></div>`;
    $("#tr-start").onclick = start;
    $("#tr-stop").onclick = async () => { await api("/api/train/stop", { method: "POST" }); refresh(); };
    $("#tr-refresh").onclick = refresh;
    $("#tr-log").onclick = () => window.open("/api/train/log", "_blank");
    $("#tr-copy-results").onclick = copyResults;
    $("#tr-download-results").onclick = downloadResults;
    $("#tr-upload").onchange = uploadCheckpoint;
    $("#tr-sort").onchange = () => renderHistory(latestStatus.history || []);
    $("#tr-sort-dir").onclick = () => {
      const button = $("#tr-sort-dir");
      button.dataset.direction = button.dataset.direction === "1" ? "-1" : "1";
      button.textContent = button.dataset.direction === "1" ? "↑" : "↓";
      renderHistory(latestStatus.history || []);
    };
    fillInitialModels();
    const output = $("#tr-output");
    output.oninput = () => { customOutputName = true; };
    const updateOutputName = () => {
      if (!customOutputName) output.value = `rl_${selectedPresets().join("_") || "training"}_top2.npz`;
    };
    document.querySelectorAll("[data-tr-preset]").forEach((input) => { input.onchange = updateOutputName; });
    updateOutputName();
  }

  async function loadCheckpoints() {
    const data = await api("/api/train/checkpoints").catch(() => ({ checkpoints: [] }));
    checkpoints = data.checkpoints || [];
    fillInitialModels();
    if (activeTab === "checkpoints") renderCheckpoints();
  }

  function fillInitialModels() {
    const select = $("#tr-init");
    if (!select) return;
    const old = select.value;
    select.innerHTML = checkpoints.map((c) => `<option value="${esc(c.path)}">${esc(c.name)}</option>`).join("");
    if (old && checkpoints.some((c) => c.path === old)) select.value = old;
    else if (checkpoints.length) select.value = checkpoints.find((c) => c.name.includes("f26_a20_h512"))?.path || checkpoints[0].path;
  }

  function selectedPresets() {
    return [...document.querySelectorAll("[data-tr-preset]:checked")].map((el) => el.dataset.trPreset);
  }

  async function start() {
    const selected = selectedPresets();
    if (!selected.length) return;
    const body = {
      presets: selected.join(","), epochs: Number($("#tr-epochs").value), batch: Number($("#tr-batch").value),
      init: $("#tr-init").value, output: $("#tr-output").value,
      resume: $("#tr-resume").checked,
      rate_weight: Number($("#tr-rate").value), speed_weight: Number($("#tr-speed").value),
      temperature: Number($("#tr-temp").value), entropy_weight: Number($("#tr-entropy").value), kl_weight: Number($("#tr-kl").value),
    };
    try { await api("/api/train/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); refresh(); } catch (error) { $("#tr-status").textContent = error.message; }
  }

  async function uploadCheckpoint() {
    const input = $("#tr-upload");
    const file = input.files?.[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    try {
      const data = await api("/api/train/upload_checkpoint", { method: "POST", body: form });
      await loadCheckpoints();
      $("#tr-init").value = data.path;
      $("#tr-resume").checked = false;
      const stem = file.name.replace(/\\.npz$/i, "").replace(/[^A-Za-z0-9_-]+/g, "_");
      $("#tr-output").value = "rl_" + (selectedPresets().join("_") || "training") + "_from_" + stem + "_top2.npz";
      customOutputName = true;
      $("#tr-status").textContent = "Uploaded " + data.name + ". Resume is disabled so training starts from this checkpoint.";
    } catch (error) {
      $("#tr-status").textContent = error.message;
    } finally {
      input.value = "";
    }
  }

  function renderHistory(history) {
    const root = $("#tr-history");
    if (!root) return;
    const newBaseline = (history.find((row) => row.validation)?.validation || history.find((row) => row.baseline_mse != null) || {});
    const newTrainBaseline = history.find((row) => row.train_baseline_mse != null) || {};
    const newObserved = (field, training = false) => history.map((row) => training ? row[field] : (row.validation || row)[field]);
    const newSorted = history.filter((row) => row.epoch || row.validation).slice();
    const newKey = $("#tr-sort")?.value || "epoch";
    const newDirection = Number($("#tr-sort-dir")?.dataset.direction || (newKey === "epoch" ? "1" : "-1"));
    newSorted.sort((a, b) => {
      const value = (row) => newKey === "epoch" ? (row.epoch ?? -1) : (newKey.startsWith("train_") ? row : (row.validation || row))[newKey];
      return (Number(value(a) ?? -Infinity) - Number(value(b) ?? -Infinity)) * newDirection;
    });
    const newRows = newSorted.map((row) => {
      const v = row.validation || row;
      return `<tr><td>${row.epoch ? `Epoch ${row.epoch}` : "Initial"}</td><td>${fmt(row.train_reward, 5)}</td><td>${fmt(row.reward ?? v.reward, 5)}</td><td>${fmt(v.aggregate_score, 5)}</td><td>${metricValue("mse", row.train_mse, row.train_baseline_mse, newObserved("train_mse", true), 3)}</td><td>${metricValue("mse", v.mse, v.baseline_mse, newObserved("mse"), 3)}</td><td>${metricValue("bpp", row.train_bpp, row.train_baseline_bpp, newObserved("train_bpp", true), 6)}</td><td>${metricValue("bpp", v.bpp, v.baseline_bpp, newObserved("bpp"), 6)}</td><td>${metricValue("work", row.train_work, row.train_baseline_work, newObserved("train_work", true), 0)}</td><td>${metricValue("work", v.work, v.baseline_work, newObserved("work"), 0)}</td><td>${fmt(row.train_encode_seconds, 3)}</td><td>${fmt(v.encode_seconds, 3)}</td></tr>`;
    });
    if (newRows.length) {
      root.innerHTML = `<div class="tr-baseline-line">Training reference: MSE ${fmt(newTrainBaseline.train_baseline_mse, 3)} · bpp ${fmt(newTrainBaseline.train_baseline_bpp, 6)} · grid coefficients ${fmt(newTrainBaseline.train_baseline_work, 0)}. Validation reference: MSE ${fmt(newBaseline.baseline_mse, 3)} · bpp ${fmt(newBaseline.baseline_bpp, 6)}. Speed is measured rollout encode time; colors fade from green (best observed) through yellow (reference) to red (worst observed).</div><div class="tr-table-wrap"><table class="tr-table"><thead><tr><th>Checkpoint</th><th>Train reward</th><th>Val reward</th><th>Aggregate score</th><th>Train MSE</th><th>Val MSE</th><th>Train bpp</th><th>Val bpp</th><th>Train grid coefficients</th><th>Val grid coefficients</th><th>Train encode s</th><th>Val encode s</th></tr></thead><tbody>${newRows.join("")}</tbody></table></div>`;
    } else {
      root.innerHTML = `<p class="card-date">No validation rows yet. The first baseline pass can take a few minutes.</p>`;
    }
    return;
    const legacyKey = $("#tr-sort")?.value || "epoch";
    const legacyDirection = Number($("#tr-sort-dir")?.dataset.direction || (legacyKey === "epoch" ? "1" : "-1"));
    const legacySorted = history.filter((row) => row.epoch || row.validation).slice().sort((a, b) => {
      const value = (row) => legacyKey === "epoch" ? (row.epoch ?? -1) : (legacyKey.startsWith("train_") ? row : (row.validation || row))[legacyKey];
      const av = value(a);
      const bv = value(b);
      return (Number(av ?? -Infinity) - Number(bv ?? -Infinity)) * legacyDirection;
    });
    const observed = (field, training = false) => sorted.map((row) => training ? row[field] : (row.validation || row)[field]);
    const valMse = observed("mse");
    const valBpp = observed("bpp");
    const valWork = observed("work");
    const trainMse = observed("train_mse", true);
    const trainBpp = observed("train_bpp", true);
    const trainWork = observed("train_work", true);
    const legacyRows = legacySorted.map((row) => {
      const v = row.validation || row;
      return `<tr><td>${row.epoch ? `Epoch ${row.epoch}` : "Initial"}</td><td>${fmt(row.train_reward, 5)}</td><td>${fmt(row.reward ?? v.reward, 5)}</td><td>${fmt(v.aggregate_score, 5)}</td><td>${metricValue("mse", row.train_mse, row.train_baseline_mse, trainMse, 3)}</td><td>${metricValue("mse", v.mse, v.baseline_mse, valMse, 3)}</td><td>${metricValue("bpp", row.train_bpp, row.train_baseline_bpp, trainBpp, 6)}</td><td>${metricValue("bpp", v.bpp, v.baseline_bpp, valBpp, 6)}</td><td>${metricValue("work", row.train_work, row.train_baseline_work, trainWork, 0)}</td><td>${metricValue("work", v.work, v.baseline_work, valWork, 0)}</td></tr>`;
    });
    const baseline = (history.find((r) => r.validation)?.validation || {});
    root.innerHTML = legacyRows.length ? `<div class="tr-baseline-line">PBC3.0 baseline: MSE ${fmt(baseline.baseline_mse, 3)} · bpp ${fmt(baseline.baseline_bpp, 6)}. <span class="tr-better">Green = better</span> · <span class="tr-close">yellow = close</span> · <span class="tr-worse">red = worse</span>. Work uses the starting policy as its reference.</div><div class="tr-table-wrap"><table class="tr-table"><thead><tr><th>Checkpoint</th><th>Train reward</th><th>Val reward</th><th>Aggregate score</th><th>Train MSE</th><th>Val MSE</th><th>Train bpp</th><th>Val bpp</th><th>Train work</th><th>Val work</th></tr></thead><tbody>${legacyRows.join("")}</tbody></table></div>` : `<p class="card-date">No validation rows yet. The first baseline pass can take a few minutes.</p>`;
  }

  function resultText() {
    const history = latestStatus.history || [];
    return history.map((row) => {
      const v = row.validation || row;
      return [row.epoch ? `Epoch ${row.epoch}` : "Initial", fmt(row.train_reward, 5), fmt(row.reward ?? v.reward, 5), fmt(v.aggregate_score, 5), fmt(row.train_mse, 3), fmt(v.mse, 3), fmt(row.train_bpp, 6), fmt(v.bpp, 6), fmt(row.train_work, 0), fmt(v.work, 0), fmt(row.train_encode_seconds, 3), fmt(v.encode_seconds, 3)].join("\t");
    }).join("\n");
  }

  async function copyResults() {
    const text = ["Checkpoint\tTrain reward\tVal reward\tAggregate score\tTrain MSE\tVal MSE\tTrain bpp\tVal bpp\tTrain grid coefficients\tVal grid coefficients\tTrain encode s\tVal encode s", resultText()].filter(Boolean).join("\n");
    try { await navigator.clipboard.writeText(text); $("#tr-pause").checked = true; } catch { window.prompt("Copy training results", text); }
  }

  function downloadResults() {
    const blob = new Blob([JSON.stringify(latestStatus, null, 2)], { type: "application/json" });
    const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = "pbc_training_results.json"; link.click(); URL.revokeObjectURL(link.href);
  }

  async function refresh() {
    const data = await api("/api/train/status").catch((error) => ({ running: false, log_tail: error.message, history: [] }));
    latestStatus = data;
    const status = $("#tr-status");
    if (!status) return;
    const state = data.running ? "Running" : data.return_code == null ? "Idle" : data.return_code === 0 ? "Finished" : `Stopped / failed (${data.return_code})`;
    const output = data.output ? `<code>${esc(data.output)}</code>` : "No output selected";
    status.innerHTML = `<b>${state}</b> · ${output}${data.spec?.presets ? ` · ${esc(data.spec.presets)}` : ""}`;
    if (!$("#tr-pause")?.checked) {
      $("#tr-log-tail").textContent = data.log_tail || "";
      renderHistory(data.history || []);
      drawCharts(data.history || []);
    }
    renderBestCheckpoints(data.checkpoint_groups || {});
    if (data.running && !poll) poll = setInterval(refresh, 3000);
    if (!data.running && poll) { clearInterval(poll); poll = null; loadCheckpoints(); }
  }

  function chartLayout(title, yTitle, extra = {}) {
    return Object.assign({ title, height: 290, margin: { l: 52, r: 48, t: 38, b: 42 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: "#cfd3dc", size: 11 }, xaxis: { title: "Epoch", gridcolor: "rgba(255,255,255,.07)" }, yaxis: { title: yTitle, gridcolor: "rgba(255,255,255,.07)" }, legend: { orientation: "h" } }, extra);
  }

  function drawCharts(history) {
    if (!window.Plotly) return;
    const points = history.filter((row) => row.epoch || row.validation);
    if (!points.length) return;
    const x = points.map((row) => row.epoch ?? 0);
    const values = points.map((row) => row.validation || row);
    const color = points.map((_, i) => i);
    Plotly.react($("#tr-chart-reward"), [
      { x, y: points.map((row) => row.train_reward), name: "train reward", mode: "lines", line: { color: "#60a5fa" } },
      { x, y: points.map((row) => row.reward ?? (row.validation || {}).reward), name: "validation reward", mode: "lines", line: { color: "#f59e0b" } },
      { x, y: values.map((row) => row.aggregate_score), name: "aggregate score", mode: "lines", line: { color: "#22c55e" } },
    ], chartLayout("Reward and aggregate score", "score"), { responsive: true, displayModeBar: false });
    Plotly.react($("#tr-chart-metrics"), [
      { x, y: points.map((row) => row.train_mse), name: "train MSE", mode: "lines", line: { color: "#93c5fd" } },
      { x, y: values.map((row) => row.mse), name: "val MSE", mode: "lines", line: { color: "#ef4444" } },
      { x, y: values.map((row) => row.baseline_mse), name: "PBC3.0 MSE", mode: "lines", line: { color: "#fca5a5", dash: "dot" } },
    ], chartLayout("MSE", "MSE"), { responsive: true, displayModeBar: false });
    Plotly.react($("#tr-chart-work"), [
      { x, y: points.map((row) => row.train_work), name: "train grid coefficients", mode: "lines", line: { color: "#a78bfa" } },
      { x, y: values.map((row) => row.work), name: "val grid coefficients", mode: "lines", line: { color: "#c084fc" } },
    ], chartLayout("Grid coefficients", "coefficients"), { responsive: true, displayModeBar: false });
    Plotly.react($("#tr-chart-train"), [
      { x, y: points.map((row) => row.train_bpp), name: "train bpp", mode: "lines", line: { color: "#93c5fd" } },
      { x, y: values.map((row) => row.bpp), name: "val bpp", mode: "lines", line: { color: "#38bdf8" } },
      { x, y: values.map((row) => row.baseline_bpp), name: "PBC3.0 bpp", mode: "lines", line: { color: "#7dd3fc", dash: "dot" } },
    ], chartLayout("Bitrate", "bits / pixel"), { responsive: true, displayModeBar: false });
    Plotly.react($("#tr-chart-speed"), [
      { x, y: points.map((row) => row.train_encode_seconds), name: "train encode seconds", mode: "lines", line: { color: "#f59e0b" } },
      { x, y: values.map((row) => row.encode_seconds), name: "val encode seconds", mode: "lines", line: { color: "#ef4444" } },
      { x, y: values.map((row) => row.baseline_speed_seconds), name: "reference seconds", mode: "lines", line: { color: "#fca5a5", dash: "dot" } },
    ], chartLayout("Encode speed", "seconds"), { responsive: true, displayModeBar: false });
    const baseline = values[0];
    Plotly.react($("#tr-chart-rd"), [
      { x: values.map((row) => row.bpp), y: values.map((row) => row.mse), text: x.map((epoch) => `Epoch ${epoch}`), mode: "markers", name: "validation RD", marker: { size: 9, color, colorscale: "YlOrRd", showscale: true, colorbar: { title: "newer" } } },
      { x: [baseline.baseline_bpp], y: [baseline.baseline_mse], text: ["PBC3.0 baseline"], mode: "markers+text", name: "PBC3.0 baseline", textposition: "top center", marker: { size: 13, color: "#fff", symbol: "diamond", line: { color: "#ef4444", width: 2 } } },
    ], chartLayout("Rate–distortion history", "MSE", { xaxis: { title: "bpp", gridcolor: "rgba(255,255,255,.07)" } }), { responsive: true, displayModeBar: false });
  }

  function renderBestCheckpoints(groups) {
    const root = $("#tr-best-checkpoints");
    if (!root) return;
    const normalized = groups.validation || groups.training ? groups : { validation: groups, training: {} };
    const labels = { mse: "Best MSE checkpoints", bpp: "Best bpp checkpoints", work: "Best grid-coefficient checkpoints", rd: "Best RD checkpoints", overall: "Best overall checkpoint" };
    root.innerHTML = ["validation", "training"].map((split) => {
      const splitGroups = normalized[split] || {};
      const allEntries = Object.values(splitGroups).flat();
      const observed = (metric) => allEntries.map((entry) => entry[metric]);
      const content = Object.entries(labels).map(([metric, label]) => {
        const entries = (splitGroups[metric] || []).filter((entry, index, all) => index === all.findIndex((other) => other.mse === entry.mse && other.bpp === entry.bpp && other.work === entry.work));
        const rows = entries.map((entry) => `<div class="tr-best-row"><span><b>Epoch ${entry.epoch}</b> · MSE ${metricValue("mse", entry.mse, entry.baseline_mse, observed("mse"), 3)} · bpp ${metricValue("bpp", entry.bpp, entry.baseline_bpp, observed("bpp"), 6)} · grid ${metricValue("work", entry.work, entry.baseline_work, observed("work"), 0)} · speed ${fmt(entry.speed_seconds, 3)}s${metric === "rd" ? ` · RD ${fmt(entry.rd_score, 5)}` : ""}${metric === "overall" ? ` · score ${fmt(entry.overall_score, 5)}` : ""}</span><a class="hold-btn" download href="/api/train/download_checkpoint?path=${encodeURIComponent(entry.path)}">Download</a></div>`).join("");
        return `<div class="tr-best-group"><h4>${label}</h4>${rows || `<p class="card-date">No checkpoint yet.</p>`}</div>`;
      }).join("");
      return `<div class="tr-best-split"><h3>${split === "validation" ? "Validation results" : "Training results"}</h3>${content}</div>`;
    }).join("");
  }

  function renderCheckpoints() {
    checkpointRoot.innerHTML = `<div class="params"><div class="params-head"><span class="field-label">Downloadable model checkpoints</span></div><p class="card-date">Download an exported <code>.npz</code> after reviewing its validation history, then upload it into the main project when you want to test or promote it.</p><input id="tr-ckpt-filter" class="tr-filter" placeholder="Filter checkpoints…"><div id="tr-checkpoint-list" class="tr-checkpoint-grid"></div></div>`;
    $("#tr-ckpt-filter").oninput = renderCheckpointList;
    renderCheckpointList();
  }

  function renderCheckpointList() {
    const root = $("#tr-checkpoint-list");
    if (!root) return;
    const filter = ($("#tr-ckpt-filter").value || "").toLowerCase();
    const visible = checkpoints.filter((c) => c.name.toLowerCase().includes(filter));
    root.innerHTML = visible.length ? visible.map((c) => `<div class="tr-checkpoint"><div><b>${esc(c.name)}</b><small>${(c.bytes / 1024).toFixed(1)} KB</small></div><a class="hold-btn" download href="/api/train/download_checkpoint?path=${encodeURIComponent(c.path)}">Download</a></div>`).join("") : `<p class="card-date">No matching checkpoints.</p>`;
  }

  setInterval(() => { if (!$("#train-lab").hidden && activeTab === "run") refresh(); }, 5000);
})();

"use strict";
/* PBC3 Sweep Lab: landing→lab navigation, Benchmark/Runner/Analyzer tabs,
   and the Sweep Runner UI. BenchmarkPage and SweepAnalyzer own their views. */
(function () {
  const $ = (id) => document.getElementById(id);
  let PARAMS = [], rows = [], pollTimer = null;

  function openLab() {
    if (typeof window.enterApp === "function") window.enterApp();
    $("app").hidden = true;
    $("sweep-lab").hidden = false;
    tab(document.querySelector("#sweep-lab .nav-item.active")?.dataset.sweepTab || "quickrd");
  }
  function closeLab() {
    $("sweep-lab").hidden = true;
    if (typeof window.exitToLanding === "function") window.exitToLanding();
    else $("landing").hidden = false;
    stopPolling();
    if (window.BenchmarkPage) window.BenchmarkPage.onHide();
    if (window.QuickRDPage) window.QuickRDPage.onHide();
    if (window.QuickRDPage) window.QuickRDPage.onHide();
  }
  function tab(which) {
    document.querySelectorAll("#sweep-lab .nav-item")
      .forEach((n) => n.classList.toggle("active", n.dataset.sweepTab === which));
    if ($("view-benchmark")) $("view-benchmark").hidden = which !== "benchmark";
    $("view-quick-rd").hidden = which !== "quickrd";
    $("view-runner").hidden = which !== "runner";
    $("view-sweep").hidden = which !== "analyzer";
    if (window.BenchmarkPage) window.BenchmarkPage.onHide();
    if (window.QuickRDPage) window.QuickRDPage.onHide();
    if (window.QuickRDPage) window.QuickRDPage.onHide();
    if (which === "benchmark") { stopPolling(); if (window.BenchmarkPage) window.BenchmarkPage.onShow(); }
    else if (which === "quickrd") { stopPolling(); if (window.QuickRDPage) window.QuickRDPage.onShow(); setTimeout(() => window.dispatchEvent(new Event("resize")), 80); }
    else if (which === "runner") { build(); refreshStatus(); }
    else { stopPolling(); if (window.SweepAnalyzer) window.SweepAnalyzer.onShow(); }
  }

  let built = false;
  function build() {
    if (built) { refreshStatus(); return; }
    $("view-runner").innerHTML = `
    <div class="sw-wrap">
      <div class="sw-card">
        <h3>Database<span class="sub">pbc3_sweep.db on the Space</span></h3>
        <div class="sw-row">
          <label class="sw-btn">Load last database<input id="run-upload" type="file" accept=".db" hidden></label>
          <button id="run-download" class="sw-btn">Download current database</button>
        </div>
        <div id="run-dbmsg" class="sw-muted" style="margin-top:8px"></div>
      </div>

      <div class="sw-card" style="margin-top:18px">
        <h3>Dataset<span class="sub">hpt_data/ · fixed on the Space</span></h3>
        <div id="run-dataset" class="sw-muted">loading…</div>
      </div>

      <div class="sw-card" style="margin-top:18px">
        <h3>New trials</h3>
        <div class="seg" id="run-mode">
          <button class="seg-btn active" data-m="optimizer">Continue with optimizer</button>
          <button class="seg-btn" data-m="grid">Continue with these parameters</button>
        </div>
        <div id="run-grid" hidden style="margin-top:14px">
          <div class="sw-muted" style="margin-bottom:8px">
            One row per parameter. Comma-separate values for a grid axis; a single value pins it.
            Everything you don't list uses PBC3 defaults.</div>
          <div id="run-grid-rows" style="display:flex;flex-direction:column;gap:8px"></div>
          <div class="sw-row" style="margin-top:10px">
            <select id="run-addparam" class="sw-select sm"><option value="">+ Add parameter…</option></select>
            <span id="run-grid-count" class="sw-muted"></span>
          </div>
        </div>
        <div class="sw-row" style="margin-top:14px;gap:10px">
          <button id="run-start" class="sw-btn primary">Start sweep</button>
          <button id="run-stop" class="sw-btn" disabled>Stop sweep</button>
        </div>
      </div>

      <div class="sw-card" style="margin-top:18px">
        <h3>Status<span id="run-badge" class="sub">idle</span></h3>
        <div id="run-status" class="sw-muted"></div>
        <pre id="run-log" class="run-log"></pre>
      </div>
    </div>`;

    $("run-upload").onchange = (e) => e.target.files[0] && uploadDb(e.target.files[0]);
    $("run-download").onclick = () => window.open("/api/sweeps/download_db", "_blank");
    $("run-mode").querySelectorAll(".seg-btn").forEach((b) => b.onclick = () => {
      $("run-mode").querySelectorAll(".seg-btn").forEach((x) => x.classList.toggle("active", x === b));
      $("run-grid").hidden = b.dataset.m !== "grid";
    });
    $("run-addparam").onchange = (e) => { if (e.target.value) { addRow(e.target.value); e.target.value = ""; } };
    $("run-start").onclick = start;
    $("run-stop").onclick = stop;

    Promise.all([
      fetch("/api/sweeps/run/params").then((r) => r.json()),
      fetch("/api/sweeps/run/dataset").then((r) => r.json()),
    ]).then(([p, ds]) => {
      PARAMS = p.params || [];
      $("run-addparam").innerHTML = `<option value="">+ Add parameter…</option>` + PARAMS.map((x) => `<option value="${x.name}">${x.name}</option>`).join("");
      $("run-dataset").innerHTML = ds.count ? `<b>${ds.count}</b> images · ` + ds.images.map((i) => `${i.name} (${i.mp} MP)`).join(" · ") : "No images found in hpt_data/.";
    });
    built = true;
  }

  const meta = (name) => PARAMS.find((p) => p.name === name);

  function addRow(name) {
    if (rows.some((r) => r.name === name)) return;
    rows.push({ name, value: "" });
    renderRows();
  }
  function renderRows() {
    const wrap = $("run-grid-rows");
    wrap.innerHTML = rows.map((r, i) => {
      const m = meta(r.name);
      const hint = m.kind === "cat" ? m.options.join(", ") : `${m.min} … ${m.max}`;
      return `<div class="sw-row tight" data-i="${i}" style="gap:10px">
        <span class="sw-tag">${r.name}</span>
        <input class="sw-input mono sw-grow" data-v placeholder="${hint}" value="${r.value}">
        <button class="sw-btn sm" data-rm>✕</button></div>`;
    }).join("");
    wrap.querySelectorAll("[data-i]").forEach((el) => {
      const i = +el.dataset.i;
      el.querySelector("[data-v]").oninput = (e) => { rows[i].value = e.target.value; updateCount(); };
      el.querySelector("[data-rm]").onclick = () => { rows.splice(i, 1); renderRows(); updateCount(); };
    });
    updateCount();
  }
  function parseVal(m, raw) {
    const s = raw.trim();
    if (m.kind === "int") return parseInt(s, 10);
    if (m.kind === "float") return parseFloat(s);
    if (m.kind === "cat" && (s === "true" || s === "false")) return s === "true";
    return s;
  }
  function buildSpec() {
    const grid = {};
    for (const r of rows) {
      const m = meta(r.name);
      const vals = r.value.split(",").map((v) => v.trim()).filter(Boolean).map((v) => parseVal(m, v));
      if (vals.length) grid[r.name] = vals;
    }
    return { grid };
  }
  function updateCount() {
    const grid = buildSpec().grid;
    const n = Object.values(grid).reduce((a, v) => a * v.length, 1);
    $("run-grid-count").textContent = Object.keys(grid).length ? `${n} trial combination(s)` : "";
  }

  async function start() {
    const mode = $("run-mode").querySelector(".seg-btn.active").dataset.m;
    const body = { mode, spec: mode === "grid" ? buildSpec() : {} };
    const d = await (await fetch("/api/sweeps/run/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
    if (d.error) return window.toast ? window.toast(d.error) : alert(d.error);
    startPolling();
    refreshStatus();
  }
  async function stop() { await fetch("/api/sweeps/run/stop", { method: "POST" }); refreshStatus(); }
  async function uploadDb(file) {
    const fd = new FormData(); fd.append("file", file);
    const d = await (await fetch("/api/sweeps/run/upload_db", { method: "POST", body: fd })).json();
    $("run-dbmsg").textContent = d.error || "Database loaded. Continuing a sweep will resume from it.";
  }

  function startPolling() { if (!pollTimer) pollTimer = setInterval(refreshStatus, 5000); }
  function stopPolling() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }
  async function refreshStatus() {
    if (!built) return;
    let s; try { s = await (await fetch("/api/sweeps/run/status")).json(); } catch { return; }
    $("run-badge").textContent = s.running ? `running · ${s.mode}` : (s.error ? "error" : "idle");
    $("run-badge").style.color = s.running ? "var(--red)" : "";
    $("run-start").disabled = s.running;
    $("run-stop").disabled = !s.running;
    const prog = s.total ? ` · ${s.done}/${s.total}` : ` · ${s.done} trials`;
    $("run-status").innerHTML = (s.running ? `Sweeping${prog}` : (s.error ? `Error: ${s.error}` : "Idle.")) +
      (s.current ? `<br><span class="sw-muted">current: ${escapeHtml(JSON.stringify(s.current))}</span>` : "");
    $("run-log").textContent = (s.log || []).slice(-25).join("\n");
    if (s.running) startPolling();
    else stopPolling();
  }
  const escapeHtml = (t) => t.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

  function labToApp() {
    $("sweep-lab").hidden = true;
    if (typeof window.enterApp === "function") window.enterApp();
    stopPolling();
    if (window.BenchmarkPage) window.BenchmarkPage.onHide();
    if (window.QuickRDPage) window.QuickRDPage.onHide();
    if (window.QuickRDPage) window.QuickRDPage.onHide();
  }
  window.SweepLab = { open: openLab, toApp: labToApp };

  $("enter-sweep").addEventListener("click", openLab);
  $("sweep-back").addEventListener("click", closeLab);
  document.querySelectorAll("#sweep-lab .nav-item")
    .forEach((n) => n.addEventListener("click", () => tab(n.dataset.sweepTab)));
})();
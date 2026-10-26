"use strict";
/* ============================================================
   PBC3 SWEEP ANALYZER
   Reads the live `pbc3` Optuna study via /api/sweeps/study, ranks by MSE
   (positive, lower = better), filters by megapixel range (server-side
   re-aggregation), plots 2D tradeoffs (incl. efficiency vs speed) and an
   opt-in 3D view, and overlays JPEG/AVIF/WEBP/preset baselines that are
   already stored in the DB. The metric filters (MSE/bpp/seconds) apply to
   the codec curves too. The Pareto front is recomputed client-side over all
   PBC trials (presets included) so it always matches what's plotted.
   Exposes window.SweepAnalyzer.onShow() for the Sweep Lab tab.
============================================================ */
(function () {
  const PLOT = window.Plotly;
  const $ = (id) => document.getElementById(id);

  const S = {
    study: null, trials: [], filtered: [],
    selected: null, codec: null, built: false, fsKind: null,
    sort: { key: "mse", dir: 1 },   // MSE ascending = best first
    show: { pbcAll: true, pbcPareto: true, presets: true, jpeg: false, jp2: false, avif: false, webp: false, jxl: false, downsample: false, png: false },
    dyn: [], mpMin: null, mpMax: null, logx: true, logy: false, psnr: false,
    fnum: { mse: NaN, bpp: NaN, sp: NaN },   // numeric metric filters (apply to PBC + codec curves)
  };

  const COLORS = {
    gray: "#5c3839", pareto: "#ff3b41", sel: "#ffce2e", base: "#ff280c",
    jpeg: "#7d8cff", jp2: "#4fa3ff", avif: "#34d39a", webp: "#54c0ff", jxl: "#c084fc", downsample: "#e6e6e6", png: "#ef19ef",
  };
  const TOGGLES = [["pbcAll", "PBC All"], ["pbcPareto", "PBC Pareto"], ["presets", "PBC Presets"], ["jpeg", "JPEG"], ["jp2", "JPEG2000"], ["webp", "WebP"], ["jxl", "JXL"], ["avif", "AVIF"], ["downsample", "Downsample"], ["png", "PNG"]];

  // 2D plot defs. efficiency = 1/(bpp·mse) (higher is better).
  const P2D = {
    qc:  { x: "bpp",   y: "mse", xt: "bpp",     yt: "MSE", note: "↙ lower-left is best", title: "MSE vs Compression" },
    qs:  { x: "speed", y: "mse", xt: "seconds", yt: "MSE", note: "↙ lower-left is best", title: "MSE vs Speed" },
    cs:  { x: "speed", y: "bpp", xt: "seconds", yt: "bpp", note: "↙ lower-left is best", title: "Compression vs Speed" },
    qcs: { x: "speed", y: "eff", xt: "seconds", yt: "efficiency  1/(bpp·MSE)", note: "↖ upper-left is best", title: "Efficiency vs Speed" },
  };

  const FS_STYLE = `
    .sw-plot-box{position:relative;}
    .sw-fs-btn{position:absolute;top:10px;right:10px;z-index:5;background:rgba(18,18,20,.72);
      border:1px solid var(--border2);color:var(--text);border-radius:7px;width:28px;height:28px;
      cursor:pointer;font-size:15px;line-height:1;display:flex;align-items:center;justify-content:center;transition:.15s;}
    .sw-fs-btn:hover{border-color:var(--red-dim);color:#fff;background:rgba(18,18,20,.9);}
    #sw-fs{position:fixed;inset:0;z-index:1000;background:rgba(0,0,0,.82);backdrop-filter:blur(6px);
      display:flex;padding:24px;animation:swfs .14s ease;}
    #sw-fs[hidden]{display:none;}
    @keyframes swfs{from{opacity:0}to{opacity:1}}
    .sw-fs-inner{margin:auto;width:96vw;height:92vh;background:var(--surface);border:1px solid var(--border);
      border-radius:16px;display:flex;flex-direction:column;padding:14px 16px;box-shadow:0 24px 70px rgba(0,0,0,.55);}
    .sw-fs-bar{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}
    .sw-fs-bar span{font-size:13px;font-weight:600;color:var(--text);}
    #sw-fs-plot{flex:1;min-height:0;}`;

  const fmt = (v, d = 4) => (v === null || v === undefined || isNaN(v)) ? "—" : (+v).toFixed(d);
  const efficiency = (t) => (t.bpp > 0 && t.mse > 0) ? 1 / (t.bpp * t.mse) : 0;
  const baseLabel = (t) => t.preset ? `${t.preset} preset` : (t.codec ? `${t.codec} q${t.q}` : "");

  /* -------------------- shell -------------------- */
  function build() {
    const st = document.createElement("style");
    st.textContent = FS_STYLE;
    document.head.appendChild(st);

    $("view-sweep").innerHTML = `
    <div class="sw-wrap">
      <div class="sw-card">
        <h3>Tuning database<span class="sub">pbc3 · live on the Space</span></h3>
        <div class="sw-row" style="gap:16px">
          <button id="sw-load" class="sw-btn primary">Refresh</button>
          <div class="sw-f"><label>MP ≥</label><input id="f-mpmin" class="sw-input" type="number" step="0.1" placeholder="any"></div>
          <div class="sw-f"><label>MP ≤</label><input id="f-mpmax" class="sw-input" type="number" step="0.1" placeholder="any"></div>
          <button id="sw-mpreset" class="sw-btn sm">Reset MP</button>
        </div>
        <div id="sw-msg" class="sw-muted" style="margin-top:8px"></div>
      </div>

      <div id="sw-summary-card" class="sw-card" hidden>
        <h3>Summary</h3>
        <div id="sw-warn"></div>
        <div id="sw-summary" class="sw-summary"></div>
      </div>

      <div id="sw-main" hidden>
        <div class="sw-card">
          <h3>Filters<span class="sub">applied to charts, table & baselines</span></h3>
          <div class="sw-row" style="gap:16px">
            <div class="sw-f"><label>MSE ≤</label><input id="f-msemax" class="sw-input" type="number" step="1" placeholder="any"></div>
            <div class="sw-f"><label>BPP ≤</label><input id="f-bppmax" class="sw-input" type="number" step="0.01" placeholder="any"></div>
            <div class="sw-f"><label>seconds ≤</label><input id="f-spmax" class="sw-input" type="number" step="0.1" placeholder="any"></div>
          </div>
          <div id="sw-dyn" style="margin-top:12px;display:flex;flex-direction:column;gap:8px"></div>
          <div class="sw-row" style="margin-top:10px">
            <select id="sw-addfilter" class="sw-select sm"><option value="">+ Add parameter filter…</option></select>
          </div>
        </div>

        <div class="sw-card" style="margin-top:18px">
          <h3>Tradeoffs<span class="sub">red = selected · ◆ = baseline</span></h3>
          <div class="sw-row" id="sw-toggles" style="margin-bottom:10px"></div>
          <div class="sw-row" id="sw-scale" style="margin-bottom:10px">
            <span class="sw-muted" style="margin-right:4px">Scale:</span>
            <label class="sw-muted" style="text-transform:none;letter-spacing:0"><input type="checkbox" id="sw-logx" checked> log x</label>
            <label class="sw-muted" style="text-transform:none;letter-spacing:0"><input type="checkbox" id="sw-logy"> log y</label>
            <label class="sw-muted" style="text-transform:none;letter-spacing:0"><input type="checkbox" id="sw-psnr"> PSNR</label>
            <button id="sw-deselect" class="sw-btn sm" style="margin-left:auto">Deselect trial</button>
          </div>
          <div class="sw-plots">
            <div class="sw-plot-box"><button class="sw-fs-btn" data-fs="qc" title="Fullscreen">⤢</button><div id="sw-p-qc" class="sw-plot"></div></div>
            <div class="sw-plot-box"><button class="sw-fs-btn" data-fs="qs" title="Fullscreen">⤢</button><div id="sw-p-qs" class="sw-plot"></div></div>
            <div class="sw-plot-box"><button class="sw-fs-btn" data-fs="cs" title="Fullscreen">⤢</button><div id="sw-p-cs" class="sw-plot"></div></div>
            <div class="sw-plot-box"><button class="sw-fs-btn" data-fs="qcs" title="Fullscreen">⤢</button><div id="sw-p-qcs" class="sw-plot"></div></div>
          </div>
          <div id="sw-readout" class="sw-muted" style="margin-top:8px"></div>
        </div>

        <div class="sw-card" style="margin-top:18px">
          <h3>3D tradeoff</h3>
          <div class="sw-row" style="margin-bottom:8px">
            <label class="sw-muted" style="text-transform:none;letter-spacing:0"><input type="checkbox" id="sw-3d-show"> Show 3D Graph</label>
            <span class="sw-muted">color by</span>
            <select id="sw-3dcolor" class="sw-select sm">
              <option value="mse">mse</option>
              <option value="seconds">seconds</option>
              <option value="bpp">bpp</option>
              <option value="pareto">pareto</option>
            </select>
            <button class="sw-fs-btn" data-fs="3d" title="Fullscreen" style="position:static;margin-left:auto">⤢</button>
          </div>
          <div id="sw-3d-wrap" style="height:460px;overflow:hidden;position:relative;border-radius:10px" hidden>
            <div id="sw-3d" style="width:100%;height:100%"></div>
          </div>
        </div>

        <div class="sw-card" style="margin-top:18px">
          <h3>Trials<span class="sub" id="sw-tcount"></span></h3>
          <div class="sw-table-wrap"><table class="sw-table" id="sw-table"></table></div>
        </div>

        <div class="sw-card" style="margin-top:18px">
          <h3>Selected trial<span class="sub" id="sw-sel-sub">none</span></h3>
          <div class="sw-sel-grid">
            <div>
              <div id="sw-sel-metrics" class="sw-kv"></div>
              <div class="sw-copybar" id="sw-copybar"></div>
            </div>
            <div>
              <div class="sw-muted" style="margin-bottom:6px">Parameters</div>
              <div id="sw-sel-params" class="sw-params"></div>
            </div>
          </div>
        </div>
      </div>

      <div id="sw-fs" hidden>
        <div class="sw-fs-inner">
          <div class="sw-fs-bar"><span id="sw-fs-title"></span><button id="sw-fs-close" class="sw-btn sm">Close ✕</button></div>
          <div id="sw-fs-plot"></div>
        </div>
      </div>
    </div>`;

    $("sw-load").onclick = load;
    ["f-mpmin", "f-mpmax"].forEach((id) => $(id).addEventListener("change", () => {
      S.mpMin = $("f-mpmin").value === "" ? null : +$("f-mpmin").value;
      S.mpMax = $("f-mpmax").value === "" ? null : +$("f-mpmax").value;
      load();
    }));
    $("sw-mpreset").onclick = () => {
      $("f-mpmin").value = ""; $("f-mpmax").value = "";
      S.mpMin = null; S.mpMax = null;
      load();
    };
    ["f-msemax", "f-bppmax", "f-spmax"].forEach((id) => $(id).addEventListener("input", applyFilters));
    $("sw-addfilter").onchange = (e) => { if (e.target.value) { addDynFilter(e.target.value); e.target.value = ""; } };

    $("sw-logx").onchange = (e) => { S.logx = e.target.checked; render2d(); };
    $("sw-logy").onchange = (e) => { S.logy = e.target.checked; render2d(); };
    $("sw-psnr").onchange = (e) => { S.psnr = e.target.checked; render2d(); };
    $("sw-deselect").onclick = () => select(null);

    $("sw-3d-show").onchange = (e) => { $("sw-3d-wrap").hidden = !e.target.checked; if (e.target.checked) render3d(); };
    $("sw-3dcolor").onchange = render3d;

    $("view-sweep").querySelectorAll(".sw-fs-btn").forEach((b) => b.onclick = () => openFs(b.dataset.fs));
    $("sw-fs-close").onclick = closeFs;
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("sw-fs").hidden) closeFs(); });

    buildToggles();
    S.built = true;
  }

  function buildToggles() {
    $("sw-toggles").innerHTML = `<span class="sw-muted" style="margin-right:4px">Show:</span>` +
      TOGGLES.map(([k, l]) =>
        `<label class="sw-muted" style="text-transform:none;letter-spacing:0">
          <input type="checkbox" data-show="${k}" ${S.show[k] ? "checked" : ""}> ${l}</label>`).join("");
    $("sw-toggles").querySelectorAll("input").forEach((el) =>
      el.onchange = () => { S.show[el.dataset.show] = el.checked; render2d(); codecReadout(); });
  }

  /* -------------------- data loading -------------------- */
  async function load() {
    msg("Loading…", true);
    const qs = [];
    if (S.mpMin != null) qs.push("mp_min=" + S.mpMin);
    if (S.mpMax != null) qs.push("mp_max=" + S.mpMax);
    try {
      const d = await (await fetch("/api/sweeps/study?" + qs.join("&"))).json();
      if (!d.exists) { $("sw-summary-card").hidden = true; $("sw-main").hidden = true; return msg("No sweep database yet — run a sweep first."); }
      const all = (d.trials || []).map((t) => ({ ...t, eff: efficiency(t) }));
      S.study = d;
      S.trials = all.filter((t) => t.kind === "pbc3");
      computePareto(S.trials);
      buildCodec(all);
      S.selected = null; S.dyn = [];
      msg(`${S.trials.length} PBC trials · ${S.trials.filter((t) => t.pareto).length} on the Pareto front`);
      $("sw-summary-card").hidden = false;
      $("sw-main").hidden = false;
      renderSummary();
      buildAddFilterOptions();
      renderDyn();
      applyFilters();
    } catch (e) { msg("Failed to load: " + e.message); }
  }

  // Client-side Pareto front over all PBC trials (presets included) on the three
  // minimized objectives. Overrides the server flag so the front always matches
  // the (MP-re-aggregated) values we actually plot.
  function computePareto(rows) {
    rows.forEach((a) => {
      a.pareto = !rows.some((b) => b !== a &&
        b.speed <= a.speed && b.bpp <= a.bpp && b.mse <= a.mse &&
        (b.speed < a.speed || b.bpp < a.bpp || b.mse < a.mse));
    });
  }

  function buildCodec(all) {
    const pick = (c) => all.filter((t) => t.codec === c)
      .map((t) => ({ q: t.q, bpp: t.bpp, mse: t.mse, speed: t.speed, eff: t.eff })).sort((a, b) => a.q - b.q);
    const jpeg = pick("JPEG"), jp2 = pick("JPEG2000"), avif = pick("AVIF"), webp = pick("WEBP"), jxl = pick("JXL"), downsample = pick("DOWNSAMPLE"), png = pick("PNG");
    S.codec = (jpeg.length || jp2.length || avif.length || webp.length || jxl.length || downsample.length || png.length) ? { jpeg, jp2, avif, webp, jxl, downsample, png } : null;
  }

  function renderSummary() {
    const d = S.study;
    $("sw-warn").innerHTML = d.completed ? "" :
      `<div class="sw-warn">No completed trials in this study yet.</div>`;
    const cells = [
      ["Study", d.study_name],
      ["MP filter", `${S.mpMin ?? "any"} – ${S.mpMax ?? "any"}`],
      ["PBC trials", S.trials.length, "hl"],
      ["Pareto trials", S.trials.filter((t) => t.pareto).length, "hl"],
      ["Metrics", (d.metric_names || []).join(" · ")],
    ];
    $("sw-summary").innerHTML = cells.map(([k, v, c]) =>
      `<div class="cell"><span class="k">${k}</span><span class="v ${c || ""}">${v}</span></div>`).join("");
  }

  /* -------------------- dynamic param filters -------------------- */
  function paramKeys() {
    const keys = new Set();
    S.trials.forEach((t) => Object.keys(t.params || {}).forEach((k) => keys.add(k)));
    return [...keys].sort();
  }
  function isNumericParam(k) {
    let seen = false;
    for (const t of S.trials) {
      const v = t.params[k];
      if (v === undefined || v === null) continue;
      seen = true;
      if (isNaN(+v)) return false;
    }
    return seen;
  }
  function buildAddFilterOptions() {
    $("sw-addfilter").innerHTML = `<option value="">+ Add parameter filter…</option>` +
      paramKeys().map((k) => `<option value="${k}">${k}</option>`).join("");
  }
  function addDynFilter(param) {
    if (S.dyn.some((d) => d.param === param)) return;
    if (isNumericParam(param)) S.dyn.push({ param, kind: "num", min: null, max: null });
    else {
      const values = [...new Set(S.trials.map((t) => String(t.params[param])).filter((v) => v !== "undefined"))].sort();
      S.dyn.push({ param, kind: "cat", values: new Set(values), all: values });
    }
    renderDyn();
    applyFilters();
  }
  function renderDyn() {
    const wrap = $("sw-dyn");
    if (!S.dyn.length) { wrap.innerHTML = ""; return; }
    wrap.innerHTML = S.dyn.map((d, i) => {
      if (d.kind === "num") {
        return `<div class="sw-row tight" data-i="${i}" style="gap:10px">
          <span class="sw-tag">${d.param}</span>
          <label class="sw-muted">min <input class="sw-input sm" type="number" data-dyn="min" value="${d.min ?? ""}"></label>
          <label class="sw-muted">max <input class="sw-input sm" type="number" data-dyn="max" value="${d.max ?? ""}"></label>
          <button class="sw-btn sm" data-dyn="rm">✕</button></div>`;
      }
      return `<div class="sw-row tight" data-i="${i}" style="gap:10px;flex-wrap:wrap">
        <span class="sw-tag">${d.param}</span>
        ${d.all.map((v) => `<label class="sw-muted" style="text-transform:none;letter-spacing:0">
          <input type="checkbox" data-dyn="cat" data-v="${v}" ${d.values.has(v) ? "checked" : ""}> ${v}</label>`).join("")}
        <button class="sw-btn sm" data-dyn="rm">✕</button></div>`;
    }).join("");
    wrap.querySelectorAll("[data-i]").forEach((row) => {
      const i = +row.dataset.i, d = S.dyn[i];
      row.querySelectorAll("[data-dyn]").forEach((el) => {
        const kind = el.dataset.dyn;
        if (kind === "rm") el.onclick = () => { S.dyn.splice(i, 1); renderDyn(); applyFilters(); };
        else if (kind === "min" || kind === "max") el.oninput = () => { d[kind] = el.value === "" ? null : +el.value; applyFilters(); };
        else if (kind === "cat") el.onchange = () => { el.checked ? d.values.add(el.dataset.v) : d.values.delete(el.dataset.v); applyFilters(); };
      });
    });
  }

  function applyFilters() {
    if (!S.study) return;
    const msemax = parseFloat($("f-msemax").value),
      bppmax = parseFloat($("f-bppmax").value),
      spmax = parseFloat($("f-spmax").value);
    S.fnum = { mse: msemax, bpp: bppmax, sp: spmax };
    S.filtered = S.trials.filter((t) => {
      if (!isNaN(msemax) && t.mse > msemax) return false;
      if (!isNaN(bppmax) && t.bpp > bppmax) return false;
      if (!isNaN(spmax) && t.speed > spmax) return false;
      for (const d of S.dyn) {
        const v = t.params[d.param];
        if (d.kind === "num") {
          const n = +v;
          if (d.min !== null && !(n >= d.min)) return false;
          if (d.max !== null && !(n <= d.max)) return false;
        } else if (d.values.size && !d.values.has(String(v))) return false;
      }
      return true;
    });
    $("sw-tcount").textContent = ` ${S.filtered.length} / ${S.trials.length} shown`;
    renderAll();
  }

  function renderAll() { render2d(); render3d(); renderTable(); }

  /* -------------------- codec baselines (from DB) -------------------- */
  // Metric filters apply to codec points too — a JPEG q95 above the bpp cap is hidden.
  function codecPoints(key) {
    if (!S.codec || !S.codec[key]) return null;
    const f = S.fnum || {};
    return S.codec[key].filter((p) =>
      (isNaN(f.mse) || p.mse <= f.mse) &&
      (isNaN(f.bpp) || p.bpp <= f.bpp) &&
      (isNaN(f.sp) || p.speed <= f.sp));
  }

  function interp(xs, ys, x) {
    for (let i = 1; i < xs.length; i++) {
      if (x >= xs[i - 1] && x <= xs[i]) {
        const f = (x - xs[i - 1]) / (xs[i] - xs[i - 1] || 1);
        return ys[i - 1] + f * (ys[i] - ys[i - 1]);
      }
    }
    return null;
  }

  function codecReadout() {
    const jp = codecPoints("jpeg");
    const sorted = jp ? [...jp].sort((a, b) => a.mse - b.mse) : [];
    if (!S.show.jpeg || !sorted.length) { $("sw-readout").textContent = ""; return; }
    const par = S.filtered.filter((t) => t.pareto);
    const lo = sorted[0].mse, hi = sorted[sorted.length - 1].mse;
    let line = `JPEG baselines span MSE ${fmt(lo, 1)}–${fmt(hi, 1)}.`;
    const mids = par.filter((t) => t.mse >= lo && t.mse <= hi);
    if (mids.length) {
      const mid = (lo + hi) / 2;
      const t = mids.reduce((a, b) => (Math.abs(b.mse - mid) < Math.abs(a.mse - mid) ? b : a));
      const jbpp = interp(sorted.map((p) => p.mse), sorted.map((p) => p.bpp), t.mse);
      if (jbpp) {
        const pct = (t.bpp - jbpp) / jbpp * 100;
        line += ` At MSE ${fmt(t.mse, 1)} (trial #${t.number}), PBC is ${Math.abs(pct).toFixed(0)}% ${pct < 0 ? "smaller" : "larger"} than JPEG.`;
      }
    }
    $("sw-readout").textContent = line;
  }

  /* -------------------- 2D plots -------------------- */
  const hover = (t) => {
    const p = t.params || {};
    return `#${t.number}${t.pareto ? " · Pareto" : ""}${t.baseline ? " · " + baseLabel(t) : ""}<br>` +
      `MSE ${fmt(t.mse, 1)}<br>bpp ${fmt(t.bpp)}<br>seconds ${fmt(t.speed, 3)}<br>eff ${fmt(t.eff, 3)}<br>` +
      `patches ${p.patch_count ?? "—"} · ${p.color_space ?? ""}`;
  };

  const floorVal = (v, isLog, floor) => (isLog && (v == null || v <= 0)) ? floor : v;

  const mseToPsnr = (v) => (v != null && v > 0) ? 10 * Math.log10(65025 / v) : null;

  function ptTrace(rows, xKey, yKey, style, fx = 0, fy = 0, yfn = null, logy = S.logy) {
    return {
      x: rows.map((t) => floorVal(t[xKey], S.logx, fx)),
      y: rows.map((t) => yfn ? yfn(t[yKey]) : floorVal(t[yKey], logy, fy)),
      customdata: rows.map((t) => t.number), text: rows.map(hover),
      hovertemplate: "%{text}<extra></extra>", mode: "markers", type: "scatter", ...style,
    };
  }

  function codecTrace(pts, xKey, yKey, name, color, fx = 0, fy = 0, yfn = null, logy = S.logy) {
    return {
      x: pts.map((p) => floorVal(p[xKey], S.logx, fx)),
      y: pts.map((p) => yfn ? yfn(p[yKey]) : floorVal(p[yKey], logy, fy)),
      mode: "lines+markers", type: "scatter", name,
      line: { color }, marker: { size: 5, color },
      text: pts.map((p) => {
        const tag = name === "DOWNSAMPLE" ? `f${p.q}` : name === "PNG" ? "lossless" : `q${p.q}`;
        return `${name} ${tag}<br>bpp ${fmt(p.bpp)}<br>MSE ${fmt(p.mse, 1)}<br>seconds ${fmt(p.speed, 3)}`;
      }),
      hovertemplate: "%{text}<extra></extra>",
    };
  }

  function scatter2d(xKey, yKey, usePsnr = false) {
    const all = S.filtered, traces = [];
    const CODECS = ["jpeg", "jp2", "webp", "jxl", "avif", "downsample", "png"];
    const yfn = usePsnr ? mseToPsnr : null;
    const logy = S.logy && !usePsnr;
    const plotFloor = (key) => {
      let m = Infinity;
      const scan = (rows) => rows && rows.forEach((t) => { const v = t[key]; if (v != null && v > 0 && v < m) m = v; });
      scan(all);
      CODECS.forEach((k) => { if (S.show[k]) scan(codecPoints(k)); });
      return (m === Infinity ? 1 : m) / 2;       // half a step below the smallest real value
    };
    const fx = S.logx ? plotFloor(xKey) : 0;
    const fy = logy ? plotFloor(yKey) : 0;

    if (S.show.pbcAll) traces.push(ptTrace(all, xKey, yKey, { name: "PBC all", marker: { size: 5, color: COLORS.gray, opacity: .55 } }, fx, fy, yfn, logy));
    if (S.show.pbcPareto) traces.push(ptTrace(all.filter((t) => t.pareto), xKey, yKey, { name: "PBC Pareto", marker: { size: 8, color: COLORS.pareto } }, fx, fy, yfn, logy));

    [["jpeg", COLORS.jpeg], ["jp2", COLORS.jp2], ["webp", COLORS.webp], ["jxl", COLORS.jxl], ["avif", COLORS.avif], ["downsample", COLORS.downsample], ["png", COLORS.png]].forEach(([k, c]) => {
      if (!S.show[k]) return;
      const pts = codecPoints(k);
      if (pts && pts.length) traces.push(codecTrace([...pts].sort((a, b) => a[xKey] - b[xKey]), xKey, yKey, k.toUpperCase(), c, fx, fy, yfn, logy));
    });

    if (S.show.presets) {
      const pre = all.filter((t) => t.preset);
      if (pre.length) traces.push(ptTrace(pre, xKey, yKey, { name: "presets", marker: { size: 12, color: COLORS.base, symbol: "diamond", line: { color: "#fff", width: 1 } } }, fx, fy, yfn, logy));
    }

    const sel = all.filter((t) => t.number === S.selected);
    if (sel.length) traces.push(ptTrace(sel, xKey, yKey, { name: "selected", marker: { size: 13, color: COLORS.sel, line: { color: "#fff", width: 1 } } }, fx, fy, yfn, logy));
    return traces;
  }

  function layout2d(xt, yt, note, usePsnr = false) {
    const logy = S.logy && !usePsnr;
    return {
      margin: { l: 56, r: 10, t: note ? 26 : 12, b: 38 }, showlegend: false,
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#a6a6ad", size: 10 },
      title: note ? { text: note, font: { size: 10, color: "#82828a" }, x: 0, xanchor: "left" } : undefined,
      // ~g formats log-axis minor ticks as their real value (0.2 instead of "2").
      xaxis: { title: xt, gridcolor: "#262629", zeroline: false, type: S.logx ? "log" : "linear", tickformat: S.logx ? "~g" : undefined },
      yaxis: { title: yt, gridcolor: "#262629", zeroline: false, type: logy ? "log" : "linear", tickformat: logy ? "~g" : undefined },
    };
  }

  function bindClick(id) {
    const el = $(id);
    if (el._swbound) return;
    el._swbound = true;
    el.on("plotly_click", (ev) => { const cd = ev.points[0] && ev.points[0].customdata; if (cd != null) select(cd); });
  }

  function plot2d(elId, key) {
    const c = P2D[key];
    const usePsnr = S.psnr && (key === "qc" || key === "qs");
    const yt = usePsnr ? "PSNR (dB)" : c.yt;
    const note = usePsnr ? "↖ upper-left is best" : c.note;
    PLOT.react(elId, scatter2d(c.x, c.y, usePsnr), layout2d(c.xt, yt, note, usePsnr), { displayModeBar: false, responsive: true });
    bindClick(elId);
  }

  function render2d() {
    if (typeof PLOT === "undefined" || !S.study) return;
    ["qc", "qs", "cs", "qcs"].forEach((k) => plot2d("sw-p-" + k, k));
    codecReadout();
    if (S.fsKind && S.fsKind !== "3d") { plot2d("sw-fs-plot", S.fsKind); resizeFs(); }
  }

  /* -------------------- 3D plot -------------------- */
  function render3dInto(elId) {
    try {
      const by = $("sw-3dcolor").value, all = S.filtered;
      const colorVals = by === "pareto" ? all.map((t) => (t.pareto ? 1 : 0))
        : ["mse", "bpp"].includes(by) ? all.map((t) => t[by])
          : by === "seconds" ? all.map((t) => t.speed)
            : all.map((t) => +t.params[by]);
      const traces = [{
        x: all.map((t) => t.bpp), y: all.map((t) => t.speed), z: all.map((t) => t.mse),
        customdata: all.map((t) => t.number), text: all.map(hover), hovertemplate: "%{text}<extra></extra>",
        mode: "markers", type: "scatter3d",
        marker: { size: 3, color: colorVals, colorscale: "Viridis", opacity: .85, showscale: true, colorbar: { title: by, thickness: 10, len: .6 } },
      }];
      const par = all.filter((t) => t.pareto);
      traces.push({
        x: par.map((t) => t.bpp), y: par.map((t) => t.speed), z: par.map((t) => t.mse),
        text: par.map(hover), hovertemplate: "%{text}<extra></extra>", mode: "markers", type: "scatter3d",
        name: "Pareto", marker: { size: 5, color: COLORS.pareto },
      });
      const sel = all.find((t) => t.number === S.selected);
      if (sel) traces.push({
        x: [sel.bpp], y: [sel.speed], z: [sel.mse], mode: "markers", type: "scatter3d",
        text: [hover(sel)], hovertemplate: "%{text}<extra></extra>",
        marker: { size: 7, color: COLORS.sel, line: { color: "#fff", width: 1 } },
      });
      const layout = {
        margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: false, autosize: true,
        paper_bgcolor: "rgba(0,0,0,0)", font: { color: "#a6a6ad", size: 10 },
        scene: {
          xaxis: { title: "bpp", gridcolor: "#262629", backgroundcolor: "rgba(0,0,0,0)" },
          yaxis: { title: "seconds", gridcolor: "#262629", backgroundcolor: "rgba(0,0,0,0)" },
          zaxis: { title: "MSE", gridcolor: "#262629", backgroundcolor: "rgba(0,0,0,0)" },
        },
      };
      PLOT.react(elId, traces, layout, { displayModeBar: false, responsive: true });
      PLOT.Plots.resize($(elId));
      bindClick(elId);
    } catch (e) {
      $(elId).innerHTML = `<div class="sw-empty">3D render failed: ${e.message}</div>`;
    }
  }

  function render3d() {
    if (typeof PLOT === "undefined" || !S.study) return;
    if ($("sw-3d-show") && $("sw-3d-show").checked) render3dInto("sw-3d");
    if (S.fsKind === "3d") { render3dInto("sw-fs-plot"); resizeFs(); }
  }

  /* -------------------- fullscreen -------------------- */
  function openFs(kind) {
    if (typeof PLOT === "undefined") return toastMsg("Plotly unavailable");
    S.fsKind = kind;
    $("sw-fs-title").textContent = kind === "3d" ? "3D tradeoff" : P2D[kind].title;
    $("sw-fs").hidden = false;
    if (kind === "3d") render3dInto("sw-fs-plot"); else plot2d("sw-fs-plot", kind);
    resizeFs();
  }
  function closeFs() {
    if (S.fsKind) { try { PLOT.purge("sw-fs-plot"); } catch (e) { /* noop */ } }
    S.fsKind = null;
    $("sw-fs-plot")._swbound = false;
    $("sw-fs").hidden = true;
    resizeAll();
  }
  function resizeFs() {
    const el = $("sw-fs-plot");
    if (el && el.data && !$("sw-fs").hidden) PLOT.Plots.resize(el);
  }

  /* -------------------- table -------------------- */
  const COLS = [
    ["number", "Trial"], ["pareto", "Par"], ["baseline", "Base"], ["mse", "MSE"],
    ["bpp", "BPP"], ["speed", "sec"], ["eff", "eff"],
    ["patch_count", "patches"], ["q_init", "q_init"], ["color_space", "color"], ["_actions", ""],
  ];
  const METRIC_KEYS = ["mse", "bpp", "speed", "eff"];
  const cellVal = (t, k) =>
    k === "number" ? t.number :
      k === "pareto" ? (t.pareto ? 1 : 0) :
        k === "baseline" ? (t.baseline ? 1 : 0) :
          METRIC_KEYS.includes(k) ? t[k] : t.params[k];

  function renderTable() {
    const table = $("sw-table");
    const rows = [...S.filtered].sort((a, b) => {
      const va = cellVal(a, S.sort.key), vb = cellVal(b, S.sort.key);
      const na = +va, nb = +vb;
      if (!isNaN(na) && !isNaN(nb)) return (na - nb) * S.sort.dir;
      return String(va).localeCompare(String(vb)) * S.sort.dir;
    });
    const head = COLS.map(([k, l]) => `<th data-k="${k}">${l}${S.sort.key === k ? (S.sort.dir > 0 ? " ▲" : " ▼") : ""}</th>`).join("");
    const body = rows.map((t) => {
      const cells = COLS.map(([k]) => {
        if (k === "number") return `<td>#${t.number}</td>`;
        if (k === "pareto") return `<td>${t.pareto ? '<span class="sw-tag par">P</span>' : ""}</td>`;
        if (k === "baseline") return `<td>${t.baseline ? '<span class="sw-tag base" title="' + baseLabel(t) + '">B</span>' : ""}</td>`;
        if (k === "_actions") return `<td><button class="sw-btn sm" data-act="copy" data-n="${t.number}">Copy</button></td>`;
        if (k === "mse") return `<td>${fmt(t.mse, 1)}</td>`;
        if (k === "speed") return `<td>${fmt(t.speed, 3)}</td>`;
        if (k === "eff") return `<td>${fmt(t.eff, 3)}</td>`;
        if (k === "bpp") return `<td>${fmt(t.bpp)}</td>`;
        return `<td>${t.params[k] ?? "—"}</td>`;
      }).join("");
      return `<tr data-n="${t.number}" class="${t.number === S.selected ? "sel" : ""}">${cells}</tr>`;
    }).join("");
    table.innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>`;

    table.querySelectorAll("th").forEach((th) => th.onclick = () => {
      const k = th.dataset.k; if (k === "_actions") return;
      S.sort = { key: k, dir: S.sort.key === k ? -S.sort.dir : (METRIC_KEYS.includes(k) ? 1 : -1) };
      renderTable();
    });
    table.querySelectorAll("tbody tr").forEach((tr) => tr.onclick = (e) => {
      const btn = e.target.closest("button");
      if (btn) { copy(optunaText(byNum(+btn.dataset.n)), "Params copied"); return; }
      select(+tr.dataset.n);
    });
  }

  /* -------------------- selection -------------------- */
  const byNum = (n) => S.trials.find((t) => t.number === n);

  function select(n) {
    S.selected = n;
    renderTable(); render2d(); render3d(); renderSelected();
  }

  function renderSelected() {
    const t = byNum(S.selected);
    $("sw-sel-sub").textContent = t ? `#${t.number}` : "none";
    if (!t) {
      $("sw-sel-metrics").innerHTML = "";
      $("sw-sel-params").textContent = "";
      $("sw-copybar").innerHTML = "";
      return;
    }
    $("sw-sel-metrics").innerHTML = [
      ["MSE", fmt(t.mse, 1)], ["BPP", fmt(t.bpp)], ["Seconds", fmt(t.speed, 3)],
      ["Efficiency", fmt(t.eff, 3)], ["Pareto", t.pareto ? "yes" : "no"],
      ["Baseline", t.baseline ? baseLabel(t) : "no"],
    ].map(([k, v]) => `<div class="k">${k}</div><div class="v">${v}</div>`).join("");

    $("sw-sel-params").textContent = optunaText(t);

    $("sw-copybar").innerHTML = `
      <button class="sw-btn sm" id="cp-opt">Copy params</button>
      <button class="sw-btn sm" id="cp-cfg">Copy PBC3Config</button>
      <button class="sw-btn sm" id="cp-json">Copy JSON</button>
      <button class="sw-btn sm primary" id="cp-load">Load into Demo</button>`;
    $("cp-opt").onclick = () => copy(optunaText(t), "Params copied");
    $("cp-cfg").onclick = () => copy(configPython(t), "PBC3Config copied");
    $("cp-json").onclick = () => copy(JSON.stringify(t.params, null, 2), "JSON copied");
    $("cp-load").onclick = () => loadIntoDemo(t);
  }

  /* -------------------- copy / load -------------------- */
  function optunaText(t) {
    return Object.entries(t.params).map(([k, v]) =>
      `${k} ${v === true ? "True" : v === false ? "False" : v}`).join("\n");
  }

  function configPython(t) {
    const lines = Object.entries(t.params).map(([k, v]) => {
      if (typeof v === "boolean") return `${k}=${v ? "True" : "False"}`;
      if (typeof v === "number") return `${k}=${v}`;
      if (Array.isArray(v)) return `${k}=(${v.join(", ")})`;
      return `${k}="${v}"`;
    });
    return `PBC3Config(\n    ${lines.join(",\n    ")},\n)`;
  }

  function loadIntoDemo(t) {
    const text = optunaText(t);
    if (typeof window.applyOptuna === "function") {
      window.applyOptuna(text);
      if (window.SweepLab && typeof window.SweepLab.toApp === "function") window.SweepLab.toApp();
      else if (typeof window.gotoView === "function") window.gotoView("compress");
      toastMsg("Loaded trial #" + t.number + " into the demo");
    } else copy(text, "Copied — paste into the demo's params");
  }

  function copy(text, ok) {
    navigator.clipboard.writeText(text).then(() => toastMsg(ok)).catch(() => toastMsg("Copy failed"));
  }

  /* -------------------- misc -------------------- */
  function msg(text, busy) { $("sw-msg").innerHTML = (busy ? `<span class="sw-spin"></span> ` : "") + text; }
  function toastMsg(m) { typeof window.toast === "function" ? window.toast(m) : msg(m); }

  function resizeAll() {
    if (!S.built || typeof PLOT === "undefined") return;
    ["sw-p-qc", "sw-p-qs", "sw-p-cs", "sw-p-qcs"].forEach((id) => { const el = $(id); if (el && el.data) PLOT.Plots.resize(el); });
    const td = $("sw-3d");
    if (td && td.data && $("sw-3d-show").checked) PLOT.Plots.resize(td);
  }

  window.SweepAnalyzer = {
    onShow() {
      if (!S.built) build();
      if (typeof PLOT === "undefined") msg("Plotly failed to load (offline / blocked). Charts disabled.");
      if (!S.study) load(); else resizeAll();
    },
  };
})();
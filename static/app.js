"use strict";

/* ============================================================
   SAMPLE IMAGES  ──  swap these for your own files in static/samples/
   The teaser cycles these first, then pulls random photos from Lorem
   Picsum (picsum.photos → Unsplash-sourced, keyless, CORS-enabled).
   Set USE_REMOTE=false to only ever loop your local samples.
   Anything that fails to load/CORS falls back to a procedural pattern.
============================================================ */
const SAMPLES = [
  "/samples/1.jpg",
  "/samples/2.jpg",
  "/samples/3.jpg",
  "/samples/4.jpg",
  "/samples/5.jpg",
];
const USE_REMOTE = true;
// const remoteURL = () => `https://picsum.photos/seed/${Math.floor(Math.random() * 1e9)}/${CANVAS}/${CANVAS}`;
const remoteURL = () => `/api/random_photo?seed=${Math.floor(Math.random() * 1e9)}`;

/* ============================================================
   LANDING ROSTER ANIMATION
============================================================ */
const CANVAS = 256;                 // internal canvas resolution
const RADIUS = 50;                  // click brush radius
const GLOBAL_START = 128;           // global auto strokes: 128x128 …
const GLOBAL_END = 4;               // … down to 4x4
const GLOBAL_T = 2.5;               // … over 2.5s (then stays at GLOBAL_END)
const CLICK_END = 2;                // size clicks shrink down to
const CLICK_T = 2;                  // … over 2 seconds
const STROKES_PER_FRAME = 18;
const CYCLE_MS = 5000;
const N = 5;                        // pairs in the ring (3 visible)
const BG = [22, 22, 24];            // canvas seed color (#161618)

const SLOTS = [
  { y: 0,    op: 1,  scale: 1,   z: 3 }, // r0  mid  · active (on top)
  { y: 185,  op: .5, scale: .92, z: 2 }, // r1  bottom · next
  { y: 380,  op: 0,  scale: .85, z: 1 }, // r2  hidden below (entering)
  { y: -380, op: 0,  scale: .85, z: 1 }, // r3  hidden above (exiting)
  { y: -185, op: .5, scale: .92, z: 2 }, // r4  top · prev
];

let pairs = [], offset = 0, activePair = null, activeStart = 0, pressStart = 0, srcCursor = 0;
let firstCycle = true;              // hide the "previously seen" top slot until a real one rotates up
let landingActive = true;           // only run the teaser while the landing is on screen
let actualAlgo = false;             // landing demo runs the real PBC algorithm (streamed) when true
let streamAbort = null;             // aborts the in-flight stream when rotating / toggling / leaving
let mouse = { down: false, x: 0, y: 0, target: null, startSize: GLOBAL_START };
const cycleFill = document.getElementById("cycle-fill");

const isLive = () => landingActive && !document.hidden;

// Exponential (fast-then-slow) interpolation from s0 to s1 over T seconds.
const expSize = (s0, s1, t, T) => Math.max(1, Math.round(s0 * Math.pow(s1 / s0, Math.min(t, T) / T)));
const globalSizeNow = () => expSize(GLOBAL_START, GLOBAL_END, (performance.now() - activeStart) / 1000, GLOBAL_T);

function resetBuf(p) {
  for (let i = 0; i < p.buf.length; i += 3) { p.buf[i] = BG[0]; p.buf[i + 1] = BG[1]; p.buf[i + 2] = BG[2]; }
}

function clearCanvas(p) {
  p.ctx.fillStyle = "#161618";
  p.ctx.fillRect(0, 0, CANVAS, CANVAS);
}

function makePair() {
  const el = document.createElement("div");
  el.className = "pair";
  el.innerHTML = `
    <div class="cell src-cell"><span class="tag">source</span><img></div>
    <div class="cell canvas-cell"><span class="tag">canvas</span><canvas width="${CANVAS}" height="${CANVAS}"></canvas></div>`;
  const cv = el.querySelector("canvas");
  const ctx = cv.getContext("2d");
  const p = { el, img: el.querySelector("img"), canvas: cv, ctx, srcData: null, buf: new Uint8ClampedArray(CANVAS * CANVAS * 3) };
  clearCanvas(p);
  resetBuf(p);
  return p;
}

function proceduralFallback(p, seed) {
  const off = document.createElement("canvas");
  off.width = off.height = CANVAS;
  const c = off.getContext("2d");
  const g = c.createLinearGradient(0, 0, CANVAS, CANVAS);
  const h = (seed * 67) % 360;
  g.addColorStop(0, `hsl(${h},65%,55%)`);
  g.addColorStop(.5, `hsl(${(h + 60) % 360},60%,40%)`);
  g.addColorStop(1, `hsl(${(h + 160) % 360},70%,30%)`);
  c.fillStyle = g;
  c.fillRect(0, 0, CANVAS, CANVAS);
  for (let i = 0; i < 18; i++) {
    c.fillStyle = `hsla(${(h + i * 30) % 360},70%,${30 + (i % 5) * 10}%,.5)`;
    c.beginPath();
    c.arc(Math.random() * CANVAS, Math.random() * CANVAS, 20 + Math.random() * 70, 0, 7);
    c.fill();
  }
  p.img.src = off.toDataURL();
  p.srcData = c.getImageData(0, 0, CANVAS, CANVAS).data;
  if (actualAlgo && isLive() && p === activePair) startStream(p);
}

function loadSource(p) {
  const remote = USE_REMOTE && srcCursor >= SAMPLES.length;
  const url = remote ? remoteURL() : SAMPLES[srcCursor % SAMPLES.length];
  const seed = srcCursor++;
  clearCanvas(p);
  resetBuf(p);
  const im = new Image();
  im.crossOrigin = "anonymous";
  im.onload = () => {
    try {
      const off = document.createElement("canvas");
      off.width = off.height = CANVAS;
      const c = off.getContext("2d");
      const s = Math.max(CANVAS / im.width, CANVAS / im.height);
      const w = im.width * s, h = im.height * s;
      c.drawImage(im, (CANVAS - w) / 2, (CANVAS - h) / 2, w, h);
      p.srcData = c.getImageData(0, 0, CANVAS, CANVAS).data; // throws if CORS-tainted
      p.img.src = off.toDataURL();
      if (actualAlgo && isLive() && p === activePair) startStream(p);
    } catch (e) {
      proceduralFallback(p, seed); // CORS-tainted or decode issue
    }
  };
  im.onerror = () => proceduralFallback(p, seed);
  im.src = url;
}

function rOf(p) { return ((pairs.indexOf(p) - offset) % N + N) % N; }

function layoutAll() {
  pairs.forEach(p => {
    const r = rOf(p), slot = SLOTS[r];
    p.el.style.transform = `translate(-50%, calc(-50% + ${slot.y}px)) scale(${slot.scale})`;
    p.el.style.opacity = (firstCycle && r === 4) ? 0 : slot.op;
    p.el.style.zIndex = slot.z; // active stays above the faded prev/next tiles
    p.el.classList.toggle("active", r === 0);
    p.el.querySelector(".canvas-cell .tag").textContent = r === 0 ? "canvas · live" : "canvas";
  });
}

function tick() {
  if (!isLive()) return; // paused while off the landing page / off-tab
  offset++;
  firstCycle = false; // the pair now arriving at the top has actually been painted
  const recycled = pairs.find(p => rOf(p) === 2);
  recycled.el.classList.add("no-anim");
  loadSource(recycled);
  layoutAll();
  requestAnimationFrame(() => requestAnimationFrame(() => recycled.el.classList.remove("no-anim")));
  activePair = pairs.find(p => rOf(p) === 0);
  activeStart = performance.now();
  if (actualAlgo) startStream(activePair);
}

/* ---- Real algorithm: stream PBC frames onto the active canvas ---- */
async function startStream(p) {
  if (!actualAlgo || !isLive() || !p || !p.img.src) return;
  if (streamAbort) streamAbort.abort();
  const ctrl = new AbortController();
  streamAbort = ctrl;
  clearCanvas(p);
  try {
    const blob = await (await fetch(p.img.src)).blob();
    const fd = new FormData();
    fd.append("image", blob, "src.png");
    fd.append("downsample_initialize", "true");
    const res = await fetch("/api/stream_compress", { method: "POST", body: fd, signal: ctrl.signal });
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl); buf = buf.slice(nl + 1);
        if (!line) continue;
        let obj; try { obj = JSON.parse(line); } catch { continue; }
        if (obj.image && !ctrl.signal.aborted) drawFrame(p, obj.image, ctrl);
      }
    }
  } catch (e) { /* aborted or network error → ignore */ }
}

function drawFrame(p, dataurl, ctrl) {
  const im = new Image();
  im.onload = () => { if (!ctrl.signal.aborted) p.ctx.drawImage(im, 0, 0, CANVAS, CANVAS); };
  im.src = dataurl;
}

function fillStroke(p, x, y, size) {
  x = Math.max(0, Math.min(CANVAS - size, Math.round(x)));
  y = Math.max(0, Math.min(CANVAS - size, Math.round(y)));
  const src = p.srcData, buf = p.buf;
  if (!src) return;

  let sR = 0, sG = 0, sB = 0, sq = 0, before = 0;
  for (let yy = y; yy < y + size; yy++) {
    for (let xx = x; xx < x + size; xx++) {
      const si = (yy * CANVAS + xx) * 4, bi = (yy * CANVAS + xx) * 3;
      const r = src[si], g = src[si + 1], b = src[si + 2];
      const cr = buf[bi], cg = buf[bi + 1], cb = buf[bi + 2];
      sR += r; sG += g; sB += b;
      sq += r * r + g * g + b * b;
      before += (r - cr) * (r - cr) + (g - cg) * (g - cg) + (b - cb) * (b - cb);
    }
  }
  const n = size * size;
  // error after filling with the area average == variance of the source area
  const after = sq - (sR * sR + sG * sG + sB * sB) / n;
  if (after >= before) return; // would not improve → skip

  const mr = Math.round(sR / n), mg = Math.round(sG / n), mb = Math.round(sB / n);
  for (let yy = y; yy < y + size; yy++) {
    for (let xx = x; xx < x + size; xx++) {
      const bi = (yy * CANVAS + xx) * 3;
      buf[bi] = mr; buf[bi + 1] = mg; buf[bi + 2] = mb;
    }
  }
  p.ctx.fillStyle = `rgb(${mr},${mg},${mb})`;
  p.ctx.fillRect(x, y, size, size);
}

function paintLoop() {
  const p = activePair;
  const live = isLive();
  if (live && !actualAlgo && p && p.srcData) {
    const clicking = mouse.down && mouse.target === p;
    if (clicking) {
      const size = expSize(mouse.startSize, CLICK_END, (performance.now() - pressStart) / 1000, CLICK_T);
      for (let i = 0; i < STROKES_PER_FRAME * 2; i++) {
        const a = Math.random() * Math.PI * 2;
        const d = Math.random() * Math.max(0, RADIUS - size);
        fillStroke(p, mouse.x + Math.cos(a) * d - size / 2, mouse.y + Math.sin(a) * d - size / 2, size);
      }
    } else {
      const size = globalSizeNow();
      for (let i = 0; i < STROKES_PER_FRAME; i++) {
        fillStroke(p, Math.random() * (CANVAS - size), Math.random() * (CANVAS - size), size);
      }
    }
  }
  if (cycleFill && live) {
    const rem = Math.max(0, 1 - (performance.now() - activeStart) / CYCLE_MS);
    cycleFill.style.transform = `scaleX(${rem})`;
  }
  requestAnimationFrame(paintLoop);
}

function initRoster() {
  const roster = document.getElementById("roster");
  for (let i = 0; i < N; i++) {
    const p = makePair();
    pairs.push(p);
    roster.appendChild(p.el);
    loadSource(p);
  }
  layoutAll();
  activePair = pairs.find(p => rOf(p) === 0);
  activeStart = performance.now();

  roster.addEventListener("mousedown", e => {
    if (actualAlgo) return; // no click-painting in real-algorithm mode
    const cv = e.target.closest(".canvas-cell canvas");
    if (!cv) return;
    const p = pairs.find(x => x.canvas === cv);
    if (p !== activePair) return;
    mouse.down = true; mouse.target = p; pressStart = performance.now();
    mouse.startSize = Math.min(globalSizeNow(), RADIUS);
    moveMouse(e, cv);
  });
  roster.addEventListener("mousemove", e => {
    if (mouse.down && mouse.target) moveMouse(e, mouse.target.canvas);
  });
  window.addEventListener("mouseup", () => { mouse.down = false; });

  const useReal = document.getElementById("use-real");
  const landingRight = document.getElementById("landing-right");
  useReal.addEventListener("change", () => {
    actualAlgo = useReal.checked;
    landingRight.classList.toggle("use-real", actualAlgo);
    if (actualAlgo) {
      startStream(activePair);
    } else {
      if (streamAbort) streamAbort.abort();
      if (activePair) { clearCanvas(activePair); resetBuf(activePair); } // resume simulation cleanly
    }
  });

  // Pause everything (paint, rotation, streaming) when the tab is hidden.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      if (streamAbort) streamAbort.abort();
    } else if (landingActive) {
      activeStart = performance.now();
      if (actualAlgo) startStream(activePair);
    }
  });

  setInterval(tick, CYCLE_MS);
  requestAnimationFrame(paintLoop);
}

function moveMouse(e, cv) {
  const rect = cv.getBoundingClientRect();
  mouse.x = (e.clientX - rect.left) / rect.width * CANVAS;
  mouse.y = (e.clientY - rect.top) / rect.height * CANVAS;
}

/* ============================================================
   NAVIGATION
============================================================ */
function enterApp() {
  document.getElementById("landing").hidden = true;
  document.getElementById("app").hidden = false;
  landingActive = false;
  if (streamAbort) streamAbort.abort();
  gotoView("compress");
}
function exitToLanding() {
  document.getElementById("app").hidden = true;
  document.getElementById("landing").hidden = false;
  landingActive = true;
  activeStart = performance.now();   // resume the active pair's size curve
  if (actualAlgo) startStream(activePair);
}

document.getElementById("enter-btn").addEventListener("click", enterApp);
document.getElementById("back-home").addEventListener("click", exitToLanding);
document.querySelectorAll(".nav-item").forEach(n => n.addEventListener("click", () => gotoView(n.dataset.view)));

function gotoView(v) {
  document.querySelectorAll(".nav-item").forEach(x => x.classList.toggle("active", x.dataset.view === v));
  document.getElementById("view-compress").hidden = v !== "compress";
  document.getElementById("view-decode").hidden = v !== "decode";
  document.getElementById("view-registry").hidden = v !== "registry";
  if (v === "registry") hideLastResult();
}

/* ============================================================
   PARAMETERS  (PBC3.0)
   Auto   → pick one of the PBC3Config presets.
   Semi   → the high-impact parameters as sliders/inputs.
   Manual → every PBC3Config field.
============================================================ */
const AUTO_CONFIGS = [
  { id: "compression", label: "Compression" },
  { id: "balanced", label: "Balanced" },
  { id: "quality", label: "Quality" },
  { id: "high_quality", label: "High Quality" },
];
const PRESET_LEARNED_Q = { compression: 0.4, balanced: 0.6, quality: 0.8, high_quality: 0.95 };

const PARAMS = [
  // ---- Neural network filler (all modes) ----
  { id: "learned_filler_enabled", label: "Neural network filler", hint: "tiny learned policy — faster on heavy presets, adds one quality dial", group: "Neural network filler", modes: ["Semi", "Manual"], type: "check", value: true },
  { id: "learned_filler_q", label: "Quality dial", hint: "learned-filler quality, relative to the preset", group: "Neural network filler", modes: ["Semi", "Manual"], type: "slider", min: 0.1, max: 0.95, step: 0.05, value: 0.35, full: true },

  // ---- Search (Semi shows the first four; the rest are Manual-only) ----
  { id: "patch_count", label: "Patch count", hint: "more patches = higher quality, larger files, slower", group: "Search", modes: ["Semi", "Manual"], type: "slider", min: 0, max: 500, step: 1, value: 20, full: true },
  { id: "search_depth", label: "Search depth", group: "Search", modes: ["Semi", "Manual"], type: "slider", min: 0, max: 2000, step: 10, value: 200 },
  { id: "proposal_depth", label: "Proposal depth", group: "Search", modes: ["Semi", "Manual"], type: "slider", min: 0, max: 500, step: 5, value: 50 },
  { id: "exact_depth", label: "Exact depth", group: "Search", modes: ["Semi", "Manual"], type: "slider", min: 0, max: 200, step: 1, value: 10 },
  { id: "top_k", label: "Top-k", group: "Search", modes: ["Manual"], type: "slider", min: 1, max: 200, step: 1, value: 20 },
  { id: "anchor_block_size", label: "Anchor block size", group: "Search", modes: ["Manual"], type: "slider", min: 1, max: 64, step: 1, value: 8 },
  { id: "cell_sizes_per_candidate", label: "Cell sizes / candidate", group: "Search", modes: ["Manual"], type: "slider", min: 1, max: 16, step: 1, value: 3 },

  // ---- Quality schedule (Semi + Manual) ----
  { id: "search_q_start", label: "Search q start", group: "Quality schedule", modes: ["Semi", "Manual"], type: "slider", min: 0, max: 1, step: 0.01, value: 0.4 },
  { id: "search_q_end", label: "Search q end", group: "Quality schedule", modes: ["Semi", "Manual"], type: "slider", min: 0, max: 1, step: 0.01, value: 0.1 },
  { id: "q_init", label: "Q init", group: "Quality schedule", modes: ["Semi", "Manual"], type: "slider", min: 0, max: 1, step: 0.01, value: 0.7 },
  { id: "q_start", label: "Q start", group: "Quality schedule", modes: ["Semi", "Manual"], type: "slider", min: 0, max: 1, step: 0.01, value: 0.9 },
  { id: "q_end", label: "Q end", group: "Quality schedule", modes: ["Semi", "Manual"], type: "slider", min: 0, max: 1, step: 0.01, value: 0.9 },

  // ---- Downsampling (Semi + Manual; Manual adds the init-layer knobs) ----
  { id: "downsample_rate", label: "Downsample rate", hint: "auto = derive from max pixels", group: "Downsampling", modes: ["Semi", "Manual"], type: "slider", min: 1, max: 16, step: 0.1, value: 2, autoToggle: { value: -1 } },
  { id: "auto_downsample_max_pixels", label: "Auto downsample max pixels", group: "Downsampling", modes: ["Semi", "Manual"], type: "slider", min: 10000, max: 4000000, step: 10000, value: 250000, full: true },
  { id: "warmup_ratio", label: "Warmup ratio", hint: "off = no warmup. Otherwise: fraction of patches applied on the initial low-res canvas before switching up to a higher-res one", group: "Downsampling", modes: ["Manual"], type: "slider", min: 0, max: 1, step: 0.01, value: 0.6, full: true, autoToggle: { value: -1, label: "off" } },
  { id: "warm_downsample_max_pixels", label: "Warm downsample max pixels", hint: "post-warmup canvas size; 'full res' = no limit. Ignored unless it exceeds the initial canvas size", group: "Downsampling", modes: ["Manual"], type: "slider", min: 10000, max: 8000000, step: 10000, value: 750000, full: true, autoToggle: { value: -1, label: "full res" } },
  { id: "auto_downsample_init", label: "Auto downsample init", group: "Downsampling", modes: ["Manual"], type: "check", value: true },
  { id: "init_search_depth", label: "Init search depth", group: "Downsampling", modes: ["Manual"], type: "slider", min: 0, max: 50, step: 1, value: 7 },
  { id: "downsample_init_cell_size", label: "Init cell size", group: "Downsampling", modes: ["Manual"], type: "slider", min: 1, max: 64, step: 1, value: 12 },
  { id: "downsample_palette_bitcount", label: "Init palette bitcount", group: "Downsampling", modes: ["Manual"], type: "slider", min: 1, max: 9, step: 1, value: 6 },

  // ---- Patch & cell sizing (Manual) ----
  { id: "min_patch_size", label: "Min patch size", group: "Patch & cell sizing", modes: ["Manual"], type: "slider", min: 1, max: 1024, step: 1, value: 16 },
  { id: "max_patch_size", label: "Max patch size", group: "Patch & cell sizing", modes: ["Manual"], type: "slider", min: 1, max: 2000, step: 1, value: 400 },
  { id: "min_cell_size", label: "Min cell size", group: "Patch & cell sizing", modes: ["Manual"], type: "slider", min: 1, max: 64, step: 1, value: 1 },
  { id: "max_cell_size", label: "Max cell size", group: "Patch & cell sizing", modes: ["Manual"], type: "slider", min: 1, max: 256, step: 1, value: 64 },

  // ---- Color (Manual) ----
  { id: "color_space", label: "Color space", group: "Color", modes: ["Manual"], type: "select", options: ["RGB", "YCbCr"], value: "YCbCr" },
  { id: "channel_cycle", label: "Channel cycle", group: "Color", modes: ["Manual"], type: "select", options: ["Off", "Sum", "Max"], value: "Sum" },

  // ---- Palette & bit allocation (Manual) ----
  { id: "patch_palette_bitcount", label: "Patch palette bitcount", group: "Palette & bits", modes: ["Manual"], type: "slider", min: 1, max: 9, step: 1, value: 2 },
  { id: "patch_bitcount_mode", label: "Patch bitcount mode", group: "Palette & bits", modes: ["Manual"], type: "select", options: ["constant", "dynamic"], value: "constant" },
  { id: "dynamic_patch_bitcount_min", label: "Dynamic bitcount min", group: "Palette & bits", modes: ["Manual"], type: "slider", min: 1, max: 9, step: 1, value: 2 },
  { id: "dynamic_patch_bitcount_max", label: "Dynamic bitcount max", group: "Palette & bits", modes: ["Manual"], type: "slider", min: 1, max: 9, step: 1, value: 3 },
  { id: "palette_mode", label: "Palette mode", group: "Palette & bits", modes: ["Manual"], type: "select", options: ["generated", "explicit", "auto"], value: "generated" },
  { id: "explicit_palette_max_bitcount", label: "Explicit palette max bitcount", group: "Palette & bits", modes: ["Manual"], type: "slider", min: 1, max: 9, step: 1, value: 3 },
  { id: "palette_difference_threshold", label: "Palette diff threshold", group: "Palette & bits", modes: ["Manual"], type: "slider", min: 0, max: 255, step: 1, value: 0 },
  { id: "palette_difference_threshold_mode", label: "Palette diff threshold mode", group: "Palette & bits", modes: ["Manual"], type: "select", options: ["constant", "linear"], value: "constant" },
  { id: "mask_size", label: "Mask size", group: "Palette & bits", modes: ["Manual"], type: "slider", min: 1, max: 1023, step: 1, value: 4 },
  { id: "positive_bias", label: "Positive bias", group: "Palette & bits", modes: ["Manual"], type: "check", value: true },

  // ---- Advanced (Manual) ----
  { id: "quality_target_mae", label: "Quality target MAE", hint: "0 = off (stop early once MAE drops below this)", group: "Advanced", modes: ["Manual"], type: "slider", min: 0, max: 50, step: 0.1, value: 0, full: true },
  { id: "use_lzma", label: "Use LZMA", group: "Advanced", modes: ["Manual"], type: "check", value: true },
  { id: "random_seed", label: "Random seed", group: "Advanced", modes: ["Manual"], type: "slider", min: 0, max: 1000000, step: 1, value: 2003 },
  { id: "debug_mode", label: "Debug mode", group: "Advanced", modes: ["Manual"], type: "check", value: false },
  { id: "debug_print", label: "Debug print", group: "Advanced", modes: ["Manual"], type: "check", value: false },
];

const paramState = {};
PARAMS.forEach(p => paramState[p.id] = p.value);
paramState.preset = "quality";
let PRESET_VALUES = {};
fetch("/api/presets").then(r => r.json()).then(d => {
  PRESET_VALUES = d;
  applyPreset(paramState.preset);
  if (paramsBody.innerHTML) renderParams();
}).catch(() => {
  if (paramState.preset in PRESET_LEARNED_Q) paramState.learned_filler_q = PRESET_LEARNED_Q[paramState.preset];
});

function applyPreset(name) {
  paramState.preset = name;
  if (name in PRESET_LEARNED_Q) paramState.learned_filler_q = PRESET_LEARNED_Q[name];
  const cfg = PRESET_VALUES[name];
  if (!cfg) return;
  PARAMS.forEach(p => {
    if (p.id.startsWith("learned_filler") || !(p.id in cfg)) return;
    paramState[p.id] = (p.type === "check") ? !!cfg[p.id] : cfg[p.id];
    if (p.autoToggle) paramState[p.id + "__auto"] = (cfg[p.id] === p.autoToggle.value);
  });
}
function markCustom() {
  if (paramState.preset === "custom") return;
  paramState.preset = "custom";
  const sel = paramsBody.querySelector('[data-pid="__preset"]');
  if (sel) sel.value = "custom";
}
PARAMS.filter(p => p.autoToggle).forEach(p => paramState[p.id + "__auto"] = true);

let paramMode = "Auto";
const paramsBody = document.getElementById("params-body");

document.querySelectorAll("#param-mode .seg-btn").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll("#param-mode .seg-btn").forEach(x => x.classList.remove("active"));
  b.classList.add("active");
  paramMode = b.dataset.mode;
  if (paramMode === "Auto") {
    const preset = (paramState.preset && paramState.preset !== "custom") ? paramState.preset : "quality";
    applyPreset(preset);
  }
  renderParams();
}));

function presetSelectHtml() {
  const opts = paramMode === "Auto" ? AUTO_CONFIGS : [...AUTO_CONFIGS, { id: "custom", label: "Custom" }];
  return `<div class="param-grid"><div class="param-group-title">Preset</div>
      <div class="param full"><label>Configuration</label>
        <select data-pid="__preset">${opts.map(c => `<option value="${c.id}" ${c.id === paramState.preset ? "selected" : ""}>${c.label}</option>`).join("")}</select>
      </div></div>`;
}

function renderParams() {
  const items = PARAMS.filter(p => p.modes.includes(paramMode));
  let html = presetSelectHtml();
  if (paramMode === "Auto")
    html += `<p class="card-date">Auto — pick a preset; structural parameters are derived from the image. The neural network filler and its quality dial still apply.</p>`;
  let group = null;
  html += `<div class="param-grid">`;
  items.forEach(p => {
    if (p.group !== group) { group = p.group; html += `<div class="param-group-title">${group}</div>`; }
    html += control(p);
  });
  html += `</div>`;
  if (paramMode !== "Auto")
    html += `<p class="card-date" style="opacity:.7;margin-top:.6rem">Changing a parameter switches the preset to “Custom”; pick a preset to reset all parameters.</p>`;
  paramsBody.innerHTML = html;

  paramsBody.querySelector('[data-pid="__preset"]').addEventListener("change", e => {
    if (e.target.value === "custom") { paramState.preset = "custom"; return; }
    applyPreset(e.target.value);
    renderParams();
  });

  paramsBody.querySelectorAll("[data-pid]").forEach(el => {
    if (el.dataset.pid === "__preset") return;
    el.addEventListener("input", () => {
      const id = el.dataset.pid;
      paramState[id] = el.type === "checkbox" ? el.checked : el.value;
      if (el.type === "range" || el.type === "number")
        paramsBody.querySelectorAll(`[data-pid="${id}"]`).forEach(o => { if (o !== el) o.value = el.value; });
      if (!id.startsWith("learned_filler")) markCustom();
    });
  });

  paramsBody.querySelectorAll("[data-auto]").forEach(cb => {
    const id = cb.dataset.auto;
    cb.checked = !!paramState[id + "__auto"];
    const sync = () => {
      paramState[id + "__auto"] = cb.checked;
      paramsBody.querySelectorAll(`[data-pid="${id}"]`).forEach(el => el.disabled = cb.checked);
    };
    cb.addEventListener("change", () => { sync(); markCustom(); });
    sync();
  });
}

function control(p) {
  const v = paramState[p.id];
  const hint = p.hint ? ` <i>${p.hint}</i>` : "";
  if (p.type === "select")
    return `<div class="param"><label>${p.label}${hint}</label><select data-pid="${p.id}">${p.options.map(o => `<option ${o == v ? "selected" : ""}>${o}</option>`).join("")}</select></div>`;
  if (p.type === "check")
    return `<div class="param"><label>${p.label}${hint}</label><label class="check-row"><input type="checkbox" data-pid="${p.id}" ${v ? "checked" : ""}> enabled</label></div>`;
  return `<div class="param ${p.full ? "full" : ""}">
      <label>${p.label}${hint}</label>
      <div class="slider-row">
        <input type="range" data-pid="${p.id}" min="${p.min}" max="${p.max}" step="${p.step}" value="${v}">
        <input type="number" class="num" data-pid="${p.id}" min="${p.min}" max="${p.max}" step="${p.step}" value="${v}">
        ${p.autoToggle ? `<label class="check-inline"><input type="checkbox" data-auto="${p.id}"> ${p.autoToggle.label || "auto"}</label>` : ""}
      </div></div>`;
}

/* ---- Sweep Analyzer "Load into Demo" hook -------------------------------
   The analyzer still emits PBC2.4-style param dumps; PBC3 sweep integration
   is pending. For now this parses the dump and applies any key that matches a
   current parameter id, switching to Manual without erroring on the rest. */
function parseOptuna(text) {
  const out = {};
  text.trim().split(/\r?\n/).forEach(line => {
    const m = line.trim().match(/^([A-Za-z_]+)[\s:=]+(.+)$/);
    if (m) out[m[1]] = m[2].trim();
  });
  return out;
}
async function applyOptuna(text) {
  const p = parseOptuna(text);
  if (!Object.keys(p).length) { toast("Couldn't parse anything"); return; }
  for (const [k, v] of Object.entries(p)) if (k in paramState) paramState[k] = v;
  paramMode = "Manual";
  document.querySelectorAll("#param-mode .seg-btn").forEach(b => b.classList.toggle("active", b.dataset.mode === "Manual"));
  renderParams();
  toast("Loaded config");
}

renderParams();

/* ============================================================
   IMAGE INPUT  (compress)
============================================================ */
let currentFile = null;
const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const preview = document.getElementById("preview");
const compressBtn = document.getElementById("compress-btn");

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => fileInput.files[0] && setFile(fileInput.files[0]));
["dragover", "dragenter"].forEach(ev => dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach(ev => dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) setFile(f); });

function setFile(f) {
  if (!f.type.startsWith("image/")) return toast("Please drop an image file");
  currentFile = f;
  preview.src = URL.createObjectURL(f);
  preview.hidden = false;
  document.getElementById("dropzone-empty").hidden = true;
  compressBtn.disabled = false;
}

/* ============================================================
   PBC INPUT  (decode)
============================================================ */
let decodeFile = null;
const decodeInput = document.getElementById("decode-input");
const decodeZone = document.getElementById("decode-zone");
const decodeBtn = document.getElementById("decode-btn");

decodeZone.addEventListener("click", () => decodeInput.click());
decodeInput.addEventListener("change", () => decodeInput.files[0] && setDecodeFile(decodeInput.files[0]));
["dragover", "dragenter"].forEach(ev => decodeZone.addEventListener(ev, e => { e.preventDefault(); decodeZone.classList.add("drag"); }));
["dragleave", "drop"].forEach(ev => decodeZone.addEventListener(ev, e => { e.preventDefault(); decodeZone.classList.remove("drag"); }));
decodeZone.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) setDecodeFile(f); });

function setDecodeFile(f) {
  if (!f.name.toLowerCase().endsWith(".pbc")) return toast("Please provide a .pbc file");
  decodeFile = f;
  document.getElementById("decode-name").textContent = f.name;
  decodeBtn.disabled = false;
}

/* ============================================================
   GLOBAL JOB QUEUE  (compress / decode / codec-match)
   Body-level container → visible from any section. Compress runs
   sequentially (CPU-bound); decode + codec matches run immediately.
============================================================ */
const orbStatus = document.getElementById("orb-status");

const jobQueueEl = document.createElement("div");
jobQueueEl.className = "job-queue";
document.body.appendChild(jobQueueEl);

const JOB_LABELS = { queued: "Queued", run: "Working…", done: "Done", error: "Failed" };

function renderJob(job) {
  const el = document.createElement("div");
  el.className = "job queued";
  el.innerHTML = `
    <div class="job-head"><span class="job-title">${job.name}</span><span class="job-state">Queued</span></div>
    <div class="job-bar"><span></span></div>
    <div class="job-meta">waiting…</div>`;
  return el;
}
function createJob(name) {
  const job = { name, status: "queued" };
  job.el = renderJob(job);
  jobQueueEl.appendChild(job.el);
  return job;
}
function setJobState(job, state, label) {
  job.status = state;
  job.el.className = "job " + state;
  job.el.querySelector(".job-state").textContent = label || JOB_LABELS[state];
}
function setJobMeta(job, html) { job.el.querySelector(".job-meta").innerHTML = html; }
function finishJob(job) {
  setTimeout(() => { job.el.classList.add("out"); setTimeout(() => job.el.remove(), 400); },
    job.status === "error" ? 6000 : 4000);
}
function fileToDataURL(file) {
  return new Promise(r => { const fr = new FileReader(); fr.onload = () => r(fr.result); fr.readAsDataURL(file); });
}

function buildCompressForm(file) {
  const fd = new FormData();
  const custom = paramState.preset === "custom";
  const effectiveMode = custom && paramMode === "Auto" ? "Manual" : paramMode;

  fd.append("image", file);
  fd.append("mode", effectiveMode);

  if (effectiveMode === "Auto") {
    fd.append("auto_config", paramState.preset || "quality");
  } else {
    PARAMS.forEach(p => {
      if (p.autoToggle && paramState[p.id + "__auto"]) { fd.append(p.id, p.autoToggle.value); return; }
      const v = paramState[p.id];
      fd.append(p.id, p.type === "check" ? (v ? "true" : "false") : v);
    });
  }
  return fd;
}

/* ---- Compress: sequential queue ---- */
const compressJobs = [];
let compressRunning = false;

compressBtn.addEventListener("click", () => {
  if (!currentFile) return;
  const job = createJob(currentFile.name);
  job.file = currentFile;
  job.fd = buildCompressForm(currentFile);
  compressJobs.push(job);
  orbStatus.textContent = `queued · ${compressJobs.length} pending`;
  runCompressQueue();
});

async function runCompressQueue() {
  if (compressRunning) return;
  const job = compressJobs.find(j => j.status === "queued");
  if (!job) return;
  compressRunning = true;
  await runCompress(job);
  compressRunning = false;
  runCompressQueue();
}

async function runCompress(job) {
  setJobState(job, "run", "Encoding…");
  const t0 = performance.now();
  const timer = setInterval(() => setJobMeta(job, `${((performance.now() - t0) / 1000).toFixed(1)}s`), 100);
  orbStatus.textContent = "compressing…";
  try {
    const res = await fetch("/api/compress", { method: "POST", body: job.fd });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const total = (performance.now() - t0) / 1000;
    if (job.file && !data.original_image) data.original_image = await fileToDataURL(job.file);
    compressJobs.splice(compressJobs.indexOf(job), 1);
    setJobState(job, "done", "Done");
    let s = `total ${total.toFixed(1)}s · encode ${data.time_seconds.toFixed(2)}s · overhead ${Math.max(0, total - data.time_seconds).toFixed(1)}s`;
    if (data.stage_timings) {
      const top = Object.entries(data.stage_timings).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, v]) => `${k} ${v}s`).join(" · ");
      s += `<br><span style="opacity:.65">server: ${top}</span>`;
    }
    setJobMeta(job, s);
    addToRegistry(data); showLastResult(data);
    orbStatus.textContent = `done · ${data.compression_rate}x`;
  } catch (e) {
    compressJobs.splice(compressJobs.indexOf(job), 1);
    setJobState(job, "error", "Failed");
    setJobMeta(job, "failed — see console");
    orbStatus.textContent = "failed";
    console.error(e);
  } finally {
    clearInterval(timer);
    finishJob(job);
  }
}

/* ---- Decode: runs immediately ---- */
decodeBtn.addEventListener("click", () => { if (decodeFile) runDecode(decodeFile); });

async function runDecode(file) {
  const job = createJob(file.name);
  setJobState(job, "run", "Decoding…");
  const t0 = performance.now();
  const timer = setInterval(() => setJobMeta(job, `${((performance.now() - t0) / 1000).toFixed(1)}s`), 100);
  document.getElementById("decode-status").textContent = "decoding…";
  const fd = new FormData();
  fd.append("file", file);
  try {
    const res = await fetch("/api/decode", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const total = (performance.now() - t0) / 1000;
    setJobState(job, "done", "Done");
    setJobMeta(job, `decoded ${data.compression_rate}× in ${total.toFixed(1)}s`);
    addToRegistry(data); showLastResult(data);
    document.getElementById("decode-status").textContent = `done · ${data.compression_rate}x`;
  } catch (e) {
    setJobState(job, "error", "Failed");
    setJobMeta(job, "failed — invalid .pbc?");
    document.getElementById("decode-status").textContent = "failed";
    console.error(e);
  } finally {
    clearInterval(timer);
    finishJob(job);
  }
}

/* ============================================================
   LAST-RESULT FLY-IN
============================================================ */
const lastResult = document.getElementById("last-result");
document.getElementById("lr-close").addEventListener("click", hideLastResult);
document.getElementById("lr-img").addEventListener("click", () => gotoView("registry"));

function showLastResult(d) {
  document.getElementById("lr-img").src = d.reconstructed_image;
  document.getElementById("lr-rate").textContent = d.compression_rate + "×";
  document.getElementById("lr-size").textContent = d.compressed_kb + " KB";
  lastResult.hidden = false;
  lastResult.classList.remove("out");
  void lastResult.offsetWidth;
}

function hideLastResult() {
  if (lastResult.hidden) return;
  lastResult.classList.add("out");
  setTimeout(() => { lastResult.hidden = true; }, 460);
}

/* ============================================================
   REGISTRY
============================================================ */
const registry = [];
const grid = document.getElementById("registry-grid");

function addToRegistry(d) {
  d.id = Date.now();
  d.date = new Date().toLocaleString();
  registry.unshift(d);
  document.getElementById("reg-count").textContent = registry.length;
  renderRegistry();
}

function renderRegistry() {
  document.getElementById("registry-empty").hidden = registry.length > 0;
  grid.innerHTML = "";
  registry.forEach(d => grid.appendChild(card(d)));
}

// PBC reports compression as N× (original_raw / compressed). To keep codec comparisons
// in the same terms, convert a codec's bpp to the equivalent rate: 24 bits-per-pixel
// raw (3 bytes RGB) / bpp.
const bppToRate = (bpp) => 24 / bpp;

const CODEC_COLOR = { jpeg: "#7d8cff", jp2: "#4fa3ff", avif: "#34d39a", webp: "#54c0ff", jxl: "#c084fc" };
const CODEC_LABEL = { jpeg: "JPEG", jp2: "JPEG2000", webp: "WebP", jxl: "JPEG XL", avif: "AVIF" };
const PBC_COLOR = "#ff3b41";

function card(d) {
  const el = document.createElement("div");
  el.className = "card";
  const stats = d.decoded ? `
        <div class="stat"><span class="k">Dimensions</span><span class="v">${d.width}×${d.height}px</span></div>
        <div class="stat"><span class="k">Decoded</span><span class="v">${d.original_raw_kb} KB</span></div>
        <div class="stat"><span class="k">File (.pbc)</span><span class="v hl">${d.compressed_kb} KB</span></div>
        <div class="stat"><span class="k">Rate</span><span class="v hl">${d.compression_rate}× · ${d.compression_percent}%</span></div>
        <div class="stat"><span class="k">Decode time</span><span class="v">${d.time_seconds}s</span></div>`
    : `
        <div class="stat"><span class="k">Dimensions</span><span class="v">${d.width}×${d.height}px</span></div>
        <div class="stat"><span class="k">Original</span><span class="v">${d.original_raw_kb} KB</span></div>
        <div class="stat"><span class="k">Compressed</span><span class="v hl">${d.compressed_kb} KB</span></div>
        <div class="stat"><span class="k">Rate</span><span class="v hl">${d.compression_rate}× · ${d.compression_percent}%</span></div>
        <div class="stat"><span class="k">MSE</span><span class="v">${d.mse}</span></div>
        <div class="stat"><span class="k">Time</span><span class="v">${d.time_seconds}s</span></div>`;
  const badge = d.decoded
    ? `<span class="fs-btn" style="left:8px;right:auto;width:auto;padding:0 8px;opacity:1;color:var(--red);font-size:9px;letter-spacing:.12em;text-transform:uppercase;cursor:default">decoded</span>`
    : "";

  // Only show codecs that haven't been generated yet; drop the whole row once both exist.
  const haveJ = !!d.jpeg_image, haveP = !!d.jp2_image, haveA = !!d.avif_image, haveW = !!d.webp_image, haveX = !!d.jxl_image;
  const compareBlock = (!d.decoded && d.original_image && !(haveJ && haveP && haveA && haveW && haveX))
    ? `<div class="compare-codecs">
          <span class="k">Compare to:</span>
          <div class="compare-codec-row">
            ${haveJ ? "" : `<button class="chip" data-act="cmp-jpeg" style="color:#7d8cff;border-color:#7d8cff">JPEG</button>`}
            ${haveP ? "" : `<button class="chip" data-act="cmp-jp2" style="color:#4fa3ff;border-color:#4fa3ff">JPEG2000</button>`}
            ${haveW ? "" : `<button class="chip" data-act="cmp-webp" style="color:#54c0ff;border-color:#54c0ff">WebP</button>`}
          </div>
          <div class="compare-codec-row">
            ${haveX ? "" : `<button class="chip" data-act="cmp-jxl" style="color:#c084fc;border-color:#c084fc">JPEG XL</button>`}
            ${haveA ? "" : `<button class="chip" data-act="cmp-avif" style="color:#34d39a;border-color:#34d39a">AVIF</button>`}
          </div>
        </div>`
    : "";
  const haveAnim = !!d.animation_url;
  const animBlock = d.pbc_base64
    ? `<div class="foot-row">
          <button class="chip" data-act="anim"
            style="${haveAnim ? "color:#3fd6c0;border-color:#3fd6c0" : "color:#ff9d42;border-color:#ff9d42"}">
            ${haveAnim ? "View animation" : "Create animation"}</button>
        </div>`
    : "";
  el.innerHTML = `
    <div class="card-media">
      ${badge}
      <button class="fs-btn" data-act="fs" title="Full screen">⤢</button>
      <img src="${d.reconstructed_image}">
    </div>
    <div class="card-body">
      <div class="stat-grid">${stats}</div>
      <div class="card-date">${d.date}${d.params ? " · " + escapeParams(d.params) : ""}</div>
      <div class="card-foot">
        <div class="foot-row">
          <button class="chip" data-act="png">.png</button>
          <button class="chip" data-act="pbc">.pbc</button>
        </div>
        ${compareBlock}
        ${animBlock}
        <button class="chip del" data-act="delete">Delete</button>
      </div>
    </div>`;
  el.querySelector('[data-act="fs"]').onclick = () => openViewer(d);
  el.querySelector(".card-media img").ondblclick = () => openViewer(d);
  el.querySelector('[data-act="png"]').onclick = () => savePng(d);
  el.querySelector('[data-act="pbc"]').onclick = () => savePbc(d);
  const cmpJ = el.querySelector('[data-act="cmp-jpeg"]');
  const cmpP = el.querySelector('[data-act="cmp-jp2"]');
  const cmpA = el.querySelector('[data-act="cmp-avif"]');
  const cmpW = el.querySelector('[data-act="cmp-webp"]');
  const cmpX = el.querySelector('[data-act="cmp-jxl"]');
  if (cmpJ) cmpJ.onclick = () => compareCodec(d, "jpeg", cmpJ);
  if (cmpP) cmpP.onclick = () => compareCodec(d, "jp2", cmpP);
  if (cmpA) cmpA.onclick = () => compareCodec(d, "avif", cmpA);
  if (cmpW) cmpW.onclick = () => compareCodec(d, "webp", cmpW);
  if (cmpX) cmpX.onclick = () => compareCodec(d, "jxl", cmpX);
  const animBtn = el.querySelector('[data-act="anim"]');
  if (animBtn) animBtn.onclick = () => createAnimation(d, animBtn);
  el.querySelector('[data-act="delete"]').onclick = () => {
    registry.splice(registry.indexOf(d), 1);
    document.getElementById("reg-count").textContent = registry.length;
    renderRegistry();
  };
  return el;
}

// Encode the registry entry's original image with JPEG/AVIF at PBC3's bpp, then expose
// it as a "Hold for …" / side-by-side option in the fullscreen viewer. Once generated the
// codec is cached on the entry, so the chip is removed and a repeat click is impossible.
async function compareCodec(d, codec, btn) {
  if (!d.original_image || d[codec + "_image"]) return;
  const label = CODEC_LABEL[codec] || codec.toUpperCase();
  btn.disabled = true;
  btn.textContent = label + "…";
  const job = createJob(`${label} match`);
  setJobState(job, "run", `${label} matching…`);
  setJobMeta(job, "starting…");
  try {
    const blob = await (await fetch(d.original_image)).blob();
    const target = (d.compressed_kb * 1024 * 8) / (d.width * d.height);
    const fd = new FormData();
    fd.append("image", blob, "original.png");
    fd.append("codec", codec);
    fd.append("target_bpp", target);
    const res = await fetch("/api/match_codec", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "", done = null;
    while (true) {
      const { value, done: rd } = await reader.read();
      if (rd) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl); buf = buf.slice(nl + 1);
        if (!line) continue;
        const o = JSON.parse(line);
        if (o.done) done = o;
        else if (o.error) throw new Error(o.error);
        else if (o.q !== undefined) {
          setJobState(job, "run", `${label} q${o.q}`);
          setJobMeta(job, `bpp ${o.bpp.toFixed(4)} · target ${target.toFixed(4)}`);
        }
      }
    }
    if (!done) throw new Error("no result");
    d[codec + "_image"] = done.image;
    d[codec + "_bpp"] = done.bpp;
    d[codec + "_q"] = done.q;
    d[codec + "_mse"] = done.mse;
    d[codec + "_size_kb"] = done.size_kb;
    setJobState(job, "done", `${label} q${done.q}`);
    setJobMeta(job, `${done.size_kb} KB · ${done.bpp.toFixed(4)} bpp · MSE ${done.mse}`);
    renderRegistry();
    if (!viewer.hidden && registry[viewIndex] === d) renderViewer();
  } catch (e) {
    setJobState(job, "error", `${label} failed`);
    setJobMeta(job, "see console");
    toast(`${label} comparison failed`);
    console.error(e);
    btn.disabled = false;
    btn.textContent = label;
  } finally {
    finishJob(job);
  }
}

/* ============================================================
   ANIMATION  (create via /api/animate, view in the animation viewer)
============================================================ */
async function createAnimation(d, btn) {
  if (d.animation_url) { openAnimViewer(d); return; }
  btn.disabled = true;
  btn.textContent = "Rendering…";
  const job = createJob("Animation");
  setJobState(job, "run", "Rendering animation…");
  setJobMeta(job, "decoding frames + encoding video…");
  const t0 = performance.now();
  const timer = setInterval(() => setJobMeta(job, `${((performance.now() - t0) / 1000).toFixed(1)}s`), 200);
  try {
    const bin = atob(d.pbc_base64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    const fd = new FormData();
    fd.append("file", new Blob([arr], { type: "application/octet-stream" }), "image.pbc");
    if (d.original_image) {
      const ob = await (await fetch(d.original_image)).blob();
      fd.append("original", ob, "original.png");
    }
    const res = await fetch("/api/animate", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    const blob = await res.blob();
    d.animation_blob = blob;
    d.animation_type = blob.type;
    d.animation_url = URL.createObjectURL(blob);
    setJobState(job, "done", "Animation ready");
    setJobMeta(job, `${(blob.size / 1048576).toFixed(2)} MB · ${((performance.now() - t0) / 1000).toFixed(1)}s`);
    renderRegistry();
    openAnimViewer(d);
  } catch (e) {
    setJobState(job, "error", "Animation failed");
    setJobMeta(job, "see console");
    toast("Animation failed");
    console.error(e);
    btn.disabled = false;
    btn.textContent = "Create animation";
  } finally {
    clearInterval(timer);
    finishJob(job);
  }
}

/* ============================================================
   ANIMATION VIEWER
============================================================ */
const animViewer = document.getElementById("anim-viewer");
const animVideo = document.getElementById("anim-video");
const animGif = document.getElementById("anim-gif");
let animBlob = null, animName = "pbc_animation.mp4";
document.getElementById("anim-close").addEventListener("click", closeAnimViewer);
animViewer.addEventListener("click", e => { if (e.target === animViewer) closeAnimViewer(); });
animVideo.addEventListener("click", e => {
  e.preventDefault();           // stop the browser's own click-to-toggle
  animVideo.paused ? animVideo.play() : animVideo.pause();
});
animVideo.addEventListener("seeking", () => { if (!animVideo.paused) animVideo.pause(); });
document.getElementById("anim-download").addEventListener("click", () => { if (animBlob) download(animBlob, animName); });
window.addEventListener("keydown", e => { if (!animViewer.hidden && e.key === "Escape") closeAnimViewer(); });

function openAnimViewer(d) {
  const isGif = d.animation_type === "image/gif";
  animGif.hidden = !isGif;
  animVideo.hidden = isGif;
  animBlob = d.animation_blob;
  animName = `pbc_animation_${d.id}.${isGif ? "gif" : "mp4"}`;
  if (isGif) {
    animGif.src = d.animation_url;
  } else {
    animVideo.src = d.animation_url;
    animVideo.controls = true;
    animVideo.play().catch(() => {});
  }
  animViewer.hidden = false;
}

function closeAnimViewer() {
  animViewer.hidden = true;
  try { animVideo.pause(); } catch (_) {}
}

function escapeParams(p) {
  if (!p || p.mode === "Auto") return p && p.auto_config ? `Auto · ${p.auto_config}` : "Auto";
  return Object.entries(p).filter(([k]) => k !== "mode")
    .map(([k, v]) => `${k}=${Array.isArray(v) ? v.join(",") : v}`).join(" · ") || p.mode;
}

/* ============================================================
   FULLSCREEN VIEWER
   - Hold mode (default): press-and-hold a button to swap the single image.
   - Side-by-side mode: the same buttons become toggles that place the chosen
     reference next to PBC3. Row vs. stacked is whichever fits the bigger image
     given the current window + image aspect ratio.
   The caption reserves two lines of height at all times, so swapping in the
   comparison text never shifts the image or the buttons.
============================================================ */
const viewer = document.getElementById("viewer");
const viewerStage = document.getElementById("viewer-stage");
const holdBtn = document.getElementById("hold-original");
const holdJpeg = document.getElementById("hold-jpeg");
const holdJp2 = document.getElementById("hold-jp2");
const holdAvif = document.getElementById("hold-avif");
const holdWebp = document.getElementById("hold-webp");
const holdJxl = document.getElementById("hold-jxl");
const sbsToggle = document.getElementById("viewer-sbs");
const prevBtn = document.getElementById("viewer-prev");
const nextBtn = document.getElementById("viewer-next");
const capEl = document.getElementById("viewer-caption");
let viewIndex = -1;
let sideBySide = false;
let activeCompare = null;            // side-by-side selected comparison key
let viewerDownTarget = null;         // where the last pointerdown started (backdrop-close guard)

document.getElementById("viewer-close").addEventListener("click", closeViewer);
// Only close on a genuine click that both started AND ended on the backdrop — a
// drag that releases over the backdrop (e.g. letting go of a hold button) must not close.
viewer.addEventListener("pointerdown", e => { viewerDownTarget = e.target; });
viewer.addEventListener("click", e => { if (e.target === viewer && viewerDownTarget === viewer) closeViewer(); });
prevBtn.addEventListener("click", () => step(-1));
nextBtn.addEventListener("click", () => step(1));
sbsToggle.addEventListener("change", () => { sideBySide = sbsToggle.checked; activeCompare = null; renderViewer(); });
window.addEventListener("keydown", e => {
  if (viewer.hidden) return;
  if (e.key === "Escape") closeViewer();
  else if (e.key === "ArrowLeft") step(-1);
  else if (e.key === "ArrowRight") step(1);
});
window.addEventListener("resize", () => { if (!viewer.hidden) renderViewer(); });

function openViewer(d) { viewIndex = registry.indexOf(d); activeCompare = null; renderViewer(); viewer.hidden = false; }
function closeViewer() { viewer.hidden = true; }
function step(dir) {
  const i = viewIndex + dir;
  if (i >= 0 && i < registry.length) { viewIndex = i; activeCompare = null; renderViewer(); }
}

function compareSrcOf(d, kind) {
  return kind === "original" ? d.original_image
    : kind === "jpeg" ? d.jpeg_image
    : kind === "jp2" ? d.jp2_image
    : kind === "avif" ? d.avif_image
    : kind === "webp" ? d.webp_image
    : kind === "jxl" ? d.jxl_image : null;
}

function viewerCaption(d) {
  if (d.decoded) return `Decoded in ${d.time_seconds}s  |  ${d.compression_rate}× compression`;
  return `Compressed ${d.compression_rate}× in ${d.time_seconds}s  |  MSE: ${d.mse}`;
}

// Two-line, colored comparison summary (size + MSE only; no composite quality).
// Reusable pieces of the comparison summary (size + MSE only).
function codecCompareParts(d, codec) {
  const label = CODEC_LABEL[codec] || codec.toUpperCase();
  const cc = CODEC_COLOR[codec];
  const fmtSize = d[codec + "_size_kb"], fmtMse = d[codec + "_mse"], q = d[codec + "_q"];
  const pct = (a, b) => Math.abs((a - b) / (b || 1e-9) * 100).toFixed(0);
  const red = s => `<span style="color:${PBC_COLOR};font-weight:600">${s}</span>`;
  const csp = s => `<span style="color:${cc};font-weight:600">${s}</span>`;
  return {
    pbcStat: `${red("PBC")}: ${d.compressed_kb} KB · ${d.mse} MSE`,
    codecStat: `${csp(label)}: ${fmtSize} KB · ${fmtMse} MSE`,
    sentence: `${csp(label + " (q=" + q + ")")} compressed this image with `
      + `${red((fmtSize / d.compressed_kb).toFixed(1) + "x")} the file size and `
      + `${red((fmtMse / d.mse).toFixed(1) + "x")} the information loss.`,
  };
}

// Hold mode: both lines stacked at the bottom (real <div> lines, not <br>).
function codecCompareCaptionHTML(d, codec) {
  const p = codecCompareParts(d, codec);
  return `<div>${p.pbcStat}&nbsp;&nbsp;|&nbsp;&nbsp;${p.codecStat}</div><div>${p.sentence}</div>`;
}

function updateCaption() {
  const d = registry[viewIndex];
  if (!d) return;
  if (sideBySide && (["jpeg", "jp2", "avif", "webp", "jxl"].includes(activeCompare)))
    capEl.innerHTML = `<div>${codecCompareParts(d, activeCompare).sentence}</div>`; // stats moved on top
  else
    capEl.textContent = viewerCaption(d);
}

function renderViewer() {
  const d = registry[viewIndex];
  if (!d) return;

  const compareSrc = sideBySide ? compareSrcOf(d, activeCompare) : null;
  if (compareSrc) {
    const vw = window.innerWidth, vh = window.innerHeight;
    const rowScale = Math.min(0.47 * vw / d.width, 0.72 * vh / d.height);
    const colScale = Math.min(0.90 * vw / d.width, 0.35 * vh / d.height);
    viewerStage.className = "viewer-stage dual " + (colScale > rowScale ? "col" : "row");

    let leftCap, rightCap;
    if (["jpeg", "jp2", "avif", "webp", "jxl"].includes(activeCompare)) {
      const p = codecCompareParts(d, activeCompare);
      leftCap = `<figcaption class="cmp-stat">${p.codecStat}</figcaption>`;
      rightCap = `<figcaption class="cmp-stat">${p.pbcStat}</figcaption>`;
    } else {
      leftCap = `<figcaption>Original</figcaption>`;
      rightCap = `<figcaption>PBC3</figcaption>`;
    }
    viewerStage.innerHTML =
      `<figure class="cmp-cell">${leftCap}<img src="${compareSrc}"></figure>` +
      `<figure class="cmp-cell">${rightCap}<img src="${d.reconstructed_image}"></figure>`;
  } else {
    viewerStage.className = "viewer-stage";
    viewerStage.innerHTML = `<img src="${d.reconstructed_image}">`;
  }

  holdBtn.hidden = !d.original_image;
  holdJpeg.hidden = !d.jpeg_image;
  holdJp2.hidden = !d.jp2_image;
  holdAvif.hidden = !d.avif_image;
  holdWebp.hidden = !d.webp_image;
  holdJxl.hidden = !d.jxl_image;
  holdBtn.textContent = sideBySide ? "Original" : "Hold for original";
  holdJpeg.textContent = sideBySide ? "JPEG" : "Hold for JPEG";
  holdJp2.textContent = sideBySide ? "JPEG2000" : "Hold for JPEG2000";
  holdAvif.textContent = sideBySide ? "AVIF" : "Hold for AVIF";
  holdWebp.textContent = sideBySide ? "WebP" : "Hold for WebP";
  holdJxl.textContent = sideBySide ? "JXL" : "Hold for JXL";
  holdBtn.classList.toggle("holding", sideBySide && activeCompare === "original");
  holdJpeg.classList.toggle("holding", sideBySide && activeCompare === "jpeg");
  holdJp2.classList.toggle("holding", sideBySide && activeCompare === "jp2");
  holdAvif.classList.toggle("holding", sideBySide && activeCompare === "avif");
  holdWebp.classList.toggle("holding", sideBySide && activeCompare === "webp");
  holdJxl.classList.toggle("holding", sideBySide && activeCompare === "jxl");

  updateCaption();
  prevBtn.disabled = viewIndex <= 0;
  nextBtn.disabled = viewIndex >= registry.length - 1;
}

function wireHold(btn, kind, getCaptionHTML) {
  const release = () => {
    const d = registry[viewIndex];
    const img = viewerStage.querySelector("img");
    if (d && img) img.src = d.reconstructed_image;
    if (d) capEl.textContent = viewerCaption(d);
    btn.classList.remove("holding");
  };
  btn.addEventListener("pointerdown", e => {
    e.preventDefault();
    const d = registry[viewIndex];
    if (!d) return;
    if (sideBySide) {                               // toggle this reference on/off
      activeCompare = activeCompare === kind ? null : kind;
      renderViewer();
      return;
    }
    // Capture the pointer so the hold survives the cursor drifting off the button
    // (e.g. when the caption grows) and we always get the matching pointerup.
    try { btn.setPointerCapture(e.pointerId); } catch (_) {}
    const img = viewerStage.querySelector("img");
    const src = compareSrcOf(d, kind);
    if (!img || !src) return;
    img.src = src;
    if (getCaptionHTML) capEl.innerHTML = getCaptionHTML(d);
    btn.classList.add("holding");
  });
  btn.addEventListener("pointerup", () => { if (!sideBySide) release(); });
  btn.addEventListener("pointercancel", () => { if (!sideBySide) release(); });
}
wireHold(holdBtn, "original");
wireHold(holdJpeg, "jpeg", d => codecCompareCaptionHTML(d, "jpeg"));
wireHold(holdJp2, "jp2", d => codecCompareCaptionHTML(d, "jp2"));
wireHold(holdAvif, "avif", d => codecCompareCaptionHTML(d, "avif"));
wireHold(holdWebp, "webp", d => codecCompareCaptionHTML(d, "webp"));
wireHold(holdJxl, "jxl", d => codecCompareCaptionHTML(d, "jxl"));
/* ============================================================
   DOWNLOAD
============================================================ */
function savePbc(d) {
  const bin = atob(d.pbc_base64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  download(new Blob([arr], { type: "application/octet-stream" }), `pbc_${d.id}.pbc`);
}

async function savePng(d) {
  const blob = await (await fetch(d.reconstructed_image)).blob();
  download(blob, `pbc_${d.id}.png`);
}

function download(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/* ============================================================
   TOAST
============================================================ */
let toastTimer;
function toast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 2600);
}

/* ============================================================
   BOOT
============================================================ */
initRoster();

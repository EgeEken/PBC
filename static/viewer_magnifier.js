"use strict";

(function () {
  const stage = document.getElementById("viewer-stage");
  if (!stage) return;

  const SCALE = 4;
  let activePointer = null;
  let activeImage = null;

  const clamp = (v) => Math.max(0, Math.min(1, v));

  function wrapImages() {
    stage.querySelectorAll("img").forEach((img) => {
      if (img.parentElement && img.parentElement.classList.contains("viewer-img-frame")) return;
      const frame = document.createElement("div");
      frame.className = "viewer-img-frame";
      img.parentNode.insertBefore(frame, img);
      frame.appendChild(img);
    });
  }

  function eventImage(e) {
    return e.target.closest ? e.target.closest(".viewer-img-frame img") : null;
  }

  function setZoom(e, sourceImage = activeImage || eventImage(e)) {
    if (!sourceImage) return;

    const rect = sourceImage.getBoundingClientRect();
    const x = clamp((e.clientX - rect.left) / rect.width);
    const y = clamp((e.clientY - rect.top) / rect.height);
    const origin = `${(x * 100).toFixed(2)}% ${(y * 100).toFixed(2)}%`;

    stage.querySelectorAll(".viewer-img-frame").forEach((frame) => frame.classList.add("magnifying"));
    stage.querySelectorAll(".viewer-img-frame img").forEach((img) => {
      img.style.transformOrigin = origin;
      img.style.transform = `scale(${SCALE})`;
    });
  }

  function clearZoom() {
    activePointer = null;
    activeImage = null;
    stage.querySelectorAll(".viewer-img-frame").forEach((frame) => frame.classList.remove("magnifying"));
    stage.querySelectorAll(".viewer-img-frame img").forEach((img) => {
      img.style.transform = "";
      img.style.transformOrigin = "";
    });
  }

  new MutationObserver(wrapImages).observe(stage, { childList: true, subtree: true });
  wrapImages();

  stage.addEventListener("pointerdown", (e) => {
    const img = eventImage(e);
    if (!img) return;
    e.preventDefault();
    e.stopPropagation();
    activePointer = e.pointerId;
    activeImage = img;
    try { stage.setPointerCapture(e.pointerId); } catch (_) {}
    setZoom(e, img);
  });

  stage.addEventListener("pointermove", (e) => {
    if (activePointer === e.pointerId) setZoom(e);
  });

  stage.addEventListener("pointerup", (e) => {
    if (activePointer === e.pointerId) clearZoom();
  });
  stage.addEventListener("pointercancel", (e) => {
    if (activePointer === e.pointerId) clearZoom();
  });
  window.addEventListener("blur", clearZoom);
})();
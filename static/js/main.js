/**
 * AI Interview Analyzer – Frontend Script
 */
(function () {
  "use strict";

  // ── Element refs ───────────────────────────────────────────────────────
  const overlay         = document.getElementById("loading-overlay");
  const form            = document.getElementById("upload-form");
  const submitBtn       = document.getElementById("submit-btn");
  const fileInput       = document.getElementById("video-input");
  const uploadZone      = document.getElementById("upload-zone");
  const uploadIcon      = document.getElementById("upload-icon");
  const uploadPrompt    = document.getElementById("upload-prompt");
  const uploadSub       = document.getElementById("upload-sub");
  const previewWrap     = document.getElementById("video-preview-wrap");
  const videoPreview    = document.getElementById("video-preview");
  const previewFilename = document.getElementById("preview-filename");
  const previewFilesize = document.getElementById("preview-filesize");
  const changeFileBtn   = document.getElementById("change-file-btn");

  // ── Video preview ──────────────────────────────────────────────────────
  function showPreview(file) {
    if (!file || !previewWrap || !videoPreview) return;

    // Revoke any previous object URL to free memory
    if (videoPreview.src) URL.revokeObjectURL(videoPreview.src);

    videoPreview.src = URL.createObjectURL(file);

    if (previewFilename) {
      previewFilename.innerHTML =
        `<i class="bi bi-check-circle-fill me-1"></i>${file.name}`;
    }
    if (previewFilesize) {
      previewFilesize.textContent = `Size: ${formatBytes(file.size)}`;
    }

    // Hide the drop-zone content, show the preview panel
    previewWrap.classList.remove("d-none");
    if (uploadIcon)   uploadIcon.classList.add("d-none");
    if (uploadPrompt) uploadPrompt.classList.add("d-none");
    if (uploadSub)    uploadSub.classList.add("d-none");
  }

  function resetPreview() {
    if (videoPreview && videoPreview.src) {
      URL.revokeObjectURL(videoPreview.src);
      videoPreview.src = "";
    }
    if (previewWrap)  previewWrap.classList.add("d-none");
    if (uploadIcon)   uploadIcon.classList.remove("d-none");
    if (uploadPrompt) uploadPrompt.classList.remove("d-none");
    if (uploadSub)    uploadSub.classList.remove("d-none");
    if (fileInput)    fileInput.value = "";
  }

  if (fileInput) {
    fileInput.addEventListener("change", function () {
      const file = this.files[0];
      if (file) showPreview(file);
    });
  }

  if (changeFileBtn) {
    changeFileBtn.addEventListener("click", function () {
      resetPreview();
      if (fileInput) fileInput.click();
    });
  }

  // ── Drag-and-drop visual feedback ─────────────────────────────────────
  if (uploadZone) {
    uploadZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      uploadZone.classList.add("dragover");
    });

    ["dragleave", "drop"].forEach((evt) =>
      uploadZone.addEventListener(evt, () => uploadZone.classList.remove("dragover"))
    );

    uploadZone.addEventListener("drop", (e) => {
      e.preventDefault();
      const dt = e.dataTransfer;
      if (dt && dt.files.length) {
        fileInput.files = dt.files;
        fileInput.dispatchEvent(new Event("change"));
      }
    });
  }

  // ── Show loading overlay on form submit ────────────────────────────────
  if (form) {
    form.addEventListener("submit", function (e) {
      if (!fileInput || !fileInput.files.length) {
        e.preventDefault();
        alert("Please select a video file before submitting.");
        return;
      }
      if (overlay)   overlay.classList.add("active");
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML =
          '<span class="spinner-border spinner-border-sm me-2" role="status"></span>Analyzing…';
      }
    });
  }

  // ── Animate progress bars on results page ─────────────────────────────
  // (Chart.js bars are handled in result.html inline script)
  document.querySelectorAll(".score-bar[data-score]").forEach((bar) => {
    const target = parseFloat(bar.dataset.score) || 0;
    bar.style.width = "0%";
    requestAnimationFrame(() => {
      setTimeout(() => { bar.style.width = `${Math.min(target, 100)}%`; }, 150);
    });
  });

  // ── Helpers ────────────────────────────────────────────────────────────
  function formatBytes(bytes) {
    if (bytes === 0) return "0 B";
    const k     = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i     = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  }
})();

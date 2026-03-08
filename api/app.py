import os
import sys

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

# Allow imports from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import get_logger
from src.pipeline.interview_pipeline import analyze_interview

log = get_logger(__name__)

UPLOAD_FOLDER      = os.path.join(os.path.dirname(__file__), "..", "uploads")
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "..", "static"),
)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

log.info("Flask app initialised — upload folder: %s", UPLOAD_FOLDER)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _wants_json() -> bool:
    """Return True when the client prefers a JSON response."""
    return request.is_json or request.headers.get("Accept", "").startswith("application/json")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    # ── Validate file presence ────────────────────────────────────────────
    if "video" not in request.files:
        msg = "No video file included in the request."
        log.warning("Upload rejected: %s", msg)
        if _wants_json():
            return jsonify(status="error", message="Video file missing"), 400
        flash(msg, "danger")
        return redirect(url_for("index"))

    video = request.files["video"]

    if video.filename == "":
        msg = "No video file selected. Please choose a file before submitting."
        log.warning("Upload rejected: empty filename")
        if _wants_json():
            return jsonify(status="error", message="No file selected"), 400
        flash(msg, "warning")
        return redirect(url_for("index"))

    if not allowed_file(video.filename):
        ext  = video.filename.rsplit(".", 1)[-1].upper() if "." in video.filename else "unknown"
        msg  = (f"Unsupported file type '{ext}'. "
                f"Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS)).upper()}.")
        log.warning("Upload rejected: unsupported extension for '%s'", video.filename)
        if _wants_json():
            return jsonify(status="error", message=msg), 400
        flash(msg, "danger")
        return redirect(url_for("index"))

    # ── Save uploaded file ────────────────────────────────────────────────
    filename  = secure_filename(video.filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    video.save(save_path)
    log.info("Video uploaded: %s  (%.1f MB)", filename,
             os.path.getsize(save_path) / (1024 * 1024))

    # ── Run pipeline ──────────────────────────────────────────────────────
    try:
        log.info("Pipeline start: %s", filename)
        results = analyze_interview(save_path)
        log.info("Pipeline complete: %s  score=%.1f", filename,
                 results["final_interview_score"])
    except FileNotFoundError as exc:
        log.error("Pipeline file error: %s", exc)
        if _wants_json():
            return jsonify(status="error", message=str(exc)), 400
        flash(f"File error: {exc}", "danger")
        return redirect(url_for("index"))
    except RuntimeError as exc:
        log.error("Pipeline runtime error: %s", exc, exc_info=True)
        if _wants_json():
            return jsonify(status="error", message=str(exc)), 500
        flash(f"Analysis failed: {exc}", "danger")
        return redirect(url_for("index"))
    except Exception as exc:
        log.error("Unexpected pipeline error: %s", exc, exc_info=True)
        if _wants_json():
            return jsonify(status="error", message="An unexpected error occurred."), 500
        flash(f"Unexpected error: {exc}", "danger")
        return redirect(url_for("index"))

    if _wants_json():
        return jsonify(status="ok", results=dict(results))

    return render_template("result.html", results=results, filename=filename)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(400)
def bad_request(error):
    log.warning("400 Bad Request: %s", error)
    if _wants_json():
        return jsonify(status="error", message="Bad request."), 400
    flash("Bad request — please check your input.", "danger")
    return redirect(url_for("index"))


@app.errorhandler(413)
def request_entity_too_large(error):
    log.warning("413 Payload Too Large")
    if _wants_json():
        return jsonify(status="error", message="File too large. Max 500 MB."), 413
    flash("Uploaded file is too large. Maximum allowed size is 500 MB.", "danger")
    return redirect(url_for("index")), 413


@app.errorhandler(500)
def internal_error(error):
    log.error("500 Internal Server Error: %s", error)
    if _wants_json():
        return jsonify(status="error", message="Internal server error."), 500
    flash("An internal error occurred. Please try again.", "danger")
    return redirect(url_for("index"))


if __name__ == "__main__":
    log.info("Starting development server on http://0.0.0.0:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)


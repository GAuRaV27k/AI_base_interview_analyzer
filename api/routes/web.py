from __future__ import annotations

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from api.services.interview_service import analyze_uploaded_video
from api.services.upload_service import UploadValidationError, save_upload_from_request
from api.utils.logging import get_logger
from api.utils.responses import json_error, json_success, wants_json_response

web_bp = Blueprint("web", __name__)
log = get_logger(__name__)


@web_bp.route("/")
def index():
    return render_template("index.html")


@web_bp.route("/healthz", methods=["GET"])
def healthcheck():
    return json_success({"status": "ok"})


@web_bp.route("/analyze", methods=["POST"])
def analyze():
    as_json = wants_json_response()

    try:
        upload = save_upload_from_request(
            request=request,
            upload_dir=current_app.config["UPLOAD_FOLDER"],
            allowed_extensions=current_app.config["ALLOWED_EXTENSIONS"],
            allowed_mime_types=current_app.config["ALLOWED_MIME_TYPES"],
            max_content_length=current_app.config.get("MAX_CONTENT_LENGTH"),
        )
        log.info("Upload accepted: %s (%.2f MB)", upload.filename, upload.size_bytes / (1024 * 1024))
    except UploadValidationError as exc:
        log.warning("Upload validation failed: %s", exc)
        if as_json:
            return json_error(str(exc), status=400)
        flash(str(exc), "danger")
        return redirect(url_for("web.index"))

    try:
        results = analyze_uploaded_video(upload.path, current_app.config)
        if as_json:
            return json_success({"filename": upload.filename, "results": results})
        return render_template("result.html", results=results, filename=upload.filename)
    except (RuntimeError, FileNotFoundError) as exc:
        log.error("Interview analysis failed: %s", exc, exc_info=True)
        if as_json:
            return json_error(str(exc), status=500)
        flash(f"Analysis failed: {exc}", "danger")
        return redirect(url_for("web.index"))


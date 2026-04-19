from __future__ import annotations

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from api.services.interview_service import analyze_uploaded_video
from api.services.job_service import get_analysis_job, submit_analysis_job
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


@web_bp.route("/readyz", methods=["GET"])
def readiness():
    preflight = current_app.extensions.get("model_preflight", {})
    if not preflight.get("ok", False):
        return json_error(preflight.get("message", "Model preflight failed."), status=503)
    return json_success({"message": preflight.get("message", "Ready")})


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

    async_enabled = bool(current_app.config.get("ANALYSIS_ASYNC_ENABLED", True))
    if as_json and async_enabled:
        job_id = submit_analysis_job(upload.path, upload.filename, current_app.config)
        return json_success(
            {
                "message": "Analysis job submitted.",
                "job_id": job_id,
                "status_url": url_for("web.analysis_job_status", job_id=job_id),
                "result_url": url_for("web.analysis_job_result", job_id=job_id),
            },
            status=202,
        )

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


@web_bp.route("/api/jobs/analyze", methods=["POST"])
def submit_analysis_job_api():
    try:
        upload = save_upload_from_request(
            request=request,
            upload_dir=current_app.config["UPLOAD_FOLDER"],
            allowed_extensions=current_app.config["ALLOWED_EXTENSIONS"],
            allowed_mime_types=current_app.config["ALLOWED_MIME_TYPES"],
            max_content_length=current_app.config.get("MAX_CONTENT_LENGTH"),
        )
    except UploadValidationError as exc:
        return json_error(str(exc), status=400)

    job_id = submit_analysis_job(upload.path, upload.filename, current_app.config)
    return json_success(
        {
            "message": "Analysis job submitted.",
            "job_id": job_id,
            "status_url": url_for("web.analysis_job_status", job_id=job_id),
            "result_url": url_for("web.analysis_job_result", job_id=job_id),
        },
        status=202,
    )


@web_bp.route("/api/jobs/<job_id>", methods=["GET"])
def analysis_job_status(job_id: str):
    job = get_analysis_job(job_id)
    if job is None:
        return json_error("Job not found.", status=404)

    if job["status"] == "failed":
        return json_error(job.get("error") or "Analysis failed.", status=500, details=job)
    return json_success(job)


@web_bp.route("/jobs/<job_id>/result", methods=["GET"])
def analysis_job_result(job_id: str):
    job = get_analysis_job(job_id)
    if job is None:
        flash("Analysis job not found or expired.", "danger")
        return redirect(url_for("web.index"))

    if job["status"] == "failed":
        flash(f"Analysis failed: {job.get('error') or 'Unknown error'}", "danger")
        return redirect(url_for("web.index"))

    if job["status"] != "completed" or not job.get("result"):
        flash("Analysis is still running. Please wait.", "warning")
        return redirect(url_for("web.index"))

    return render_template("result.html", results=job["result"], filename=job["filename"])


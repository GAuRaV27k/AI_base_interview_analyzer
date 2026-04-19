from __future__ import annotations

from flask import Flask, flash, redirect, url_for

from api.utils.logging import get_logger
from api.utils.responses import json_error, wants_json_response

log = get_logger(__name__)


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(400)
    def bad_request(error):
        log.warning("400 Bad Request: %s", error)
        if wants_json_response():
            return json_error("Bad request.", status=400)
        flash("Bad request - please check your input.", "danger")
        return redirect(url_for("web.index"))

    @app.errorhandler(404)
    def not_found(error):
        log.warning("404 Not Found: %s", error)
        if wants_json_response():
            return json_error("Resource not found.", status=404)
        flash("The requested page was not found.", "warning")
        return redirect(url_for("web.index"))

    @app.errorhandler(413)
    def request_entity_too_large(error):
        log.warning("413 Payload Too Large")
        if wants_json_response():
            return json_error(
                f"File too large. Maximum upload size is {app.config['MAX_UPLOAD_MB']} MB.",
                status=413,
            )
        flash(
            f"Uploaded file is too large. Maximum allowed size is {app.config['MAX_UPLOAD_MB']} MB.",
            "danger",
        )
        return redirect(url_for("web.index")), 413

    @app.errorhandler(500)
    def internal_error(error):
        log.error("500 Internal Server Error: %s", error, exc_info=True)
        if wants_json_response():
            return json_error("Internal server error.", status=500)
        flash("An internal error occurred. Please try again.", "danger")
        return redirect(url_for("web.index"))

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        log.error("Unexpected error: %s", error, exc_info=True)
        if wants_json_response():
            return json_error(f"Server error: {str(error)}", status=500)
        flash("An unexpected error occurred. Please try again.", "danger")
        return redirect(url_for("web.index")), 500


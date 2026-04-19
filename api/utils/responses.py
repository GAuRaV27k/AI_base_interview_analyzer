from __future__ import annotations

from flask import jsonify, request


def wants_json_response() -> bool:
    accepted = request.headers.get("Accept", "").lower()
    return request.is_json or "application/json" in accepted or request.path.startswith("/api/")


def json_success(data: dict | None = None, status: int = 200):
    payload = {"status": "ok"}
    if data:
        payload.update(data)
    return jsonify(payload), status


def json_error(message: str, status: int = 400, details: dict | None = None):
    payload = {"status": "error", "message": message}
    if details:
        payload["details"] = details
    return jsonify(payload), status


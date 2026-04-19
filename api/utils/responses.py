from __future__ import annotations

import json
from flask import jsonify, request


def wants_json_response() -> bool:
    """Determine if the client wants a JSON response."""
    accepted = request.headers.get("Accept", "").lower()
    is_api_route = request.path.startswith("/api/")
    return request.is_json or "application/json" in accepted or is_api_route


def _convert_to_serializable(obj):
    """Convert numpy and other non-JSON-serializable types to Python natives."""
    import numpy as np
    
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    return obj


def json_success(data: dict | None = None, status: int = 200):
    """Return a JSON success response, converting non-serializable types."""
    payload = {"status": "ok"}
    if data:
        payload.update(_convert_to_serializable(data))
    return jsonify(payload), status


def json_error(message: str, status: int = 400, details: dict | None = None):
    """Return a JSON error response."""
    payload = {"status": "error", "message": message}
    if details:
        payload["details"] = _convert_to_serializable(details)
    return jsonify(payload), status


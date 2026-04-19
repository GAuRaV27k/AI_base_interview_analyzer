from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from flask import Request
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


class UploadValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SavedUpload:
    filename: str
    path: str
    size_bytes: int
    content_type: str


def _is_allowed_extension(filename: str, allowed_extensions: set[str]) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in allowed_extensions


def _build_safe_filename(original_filename: str) -> str:
    cleaned = secure_filename(original_filename)
    if not cleaned:
        raise UploadValidationError("Invalid file name.")

    stem, ext = os.path.splitext(cleaned)
    unique = uuid.uuid4().hex[:12]
    return f"{stem}_{unique}{ext.lower()}"


def _validate_file(file_obj: FileStorage, allowed_extensions: set[str], allowed_mime_types: set[str]) -> None:
    if not file_obj or not file_obj.filename:
        raise UploadValidationError("No video file selected.")

    if not _is_allowed_extension(file_obj.filename, allowed_extensions):
        allowed = ", ".join(sorted(ext.upper() for ext in allowed_extensions))
        raise UploadValidationError(f"Unsupported file type. Allowed formats: {allowed}.")

    mime_type = (file_obj.mimetype or "").lower()
    if mime_type and mime_type not in allowed_mime_types:
        raise UploadValidationError(f"Unsupported content type: {mime_type}")


def save_upload_from_request(
    request: Request,
    upload_dir: str | Path,
    allowed_extensions: set[str],
    allowed_mime_types: set[str],
    max_content_length: int | None = None,
    form_field: str = "video",
) -> SavedUpload:
    if max_content_length and request.content_length and request.content_length > max_content_length:
        max_mb = round(max_content_length / (1024 * 1024))
        raise UploadValidationError(f"File too large. Maximum upload size is {max_mb} MB.")

    if form_field not in request.files:
        raise UploadValidationError("No video file included in request.")

    file_obj = request.files[form_field]
    _validate_file(file_obj, allowed_extensions, allowed_mime_types)

    target_dir = Path(upload_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _build_safe_filename(file_obj.filename or "upload.mp4")
    save_path = target_dir / safe_name
    file_obj.save(str(save_path))

    size_bytes = save_path.stat().st_size
    if size_bytes <= 0:
        save_path.unlink(missing_ok=True)
        raise UploadValidationError("Uploaded file is empty.")

    return SavedUpload(
        filename=safe_name,
        path=str(save_path),
        size_bytes=size_bytes,
        content_type=(file_obj.mimetype or "").lower(),
    )


from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from api.services.interview_service import analyze_uploaded_video, build_analysis_config
from api.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class AnalysisJob:
    id: str
    filename: str
    video_path: str
    config: dict[str, Any]
    status: str = "queued"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    result: dict[str, Any] | None = None
    error: str | None = None


_lock = threading.Lock()
_jobs: dict[str, AnalysisJob] = {}
_executor: ThreadPoolExecutor | None = None
_result_ttl_minutes = 120
_delete_upload_after_analysis = False


def initialize_job_queue(
    max_workers: int = 2,
    result_ttl_minutes: int = 120,
    delete_upload_after_analysis: bool = False,
) -> None:
    global _executor, _result_ttl_minutes, _delete_upload_after_analysis
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))
            log.info("Analysis job queue initialized (workers=%d)", max(1, int(max_workers)))
        _result_ttl_minutes = max(10, int(result_ttl_minutes))
        _delete_upload_after_analysis = bool(delete_upload_after_analysis)


def submit_analysis_job(video_path: str, filename: str, app_config: dict[str, Any]) -> str:
    if _executor is None:
        initialize_job_queue()

    job_id = uuid.uuid4().hex
    snapshot = build_analysis_config(app_config)
    job = AnalysisJob(id=job_id, filename=filename, video_path=video_path, config=snapshot)
    with _lock:
        _jobs[job_id] = job

    assert _executor is not None
    _executor.submit(_run_job, job_id)
    _prune_old_jobs()
    return job_id


def get_analysis_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        return {
            "job_id": job.id,
            "status": job.status,
            "filename": job.filename,
            "result": job.result,
            "error": job.error,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
        }


def _run_job(job_id: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.status = "running"
        job.updated_at = datetime.now(timezone.utc)

    try:
        result = analyze_uploaded_video(job.video_path, job.config)
        with _lock:
            job.status = "completed"
            job.result = result
            job.updated_at = datetime.now(timezone.utc)
    except Exception as exc:
        log.error("Analysis job failed (%s): %s", job_id, exc, exc_info=True)
        with _lock:
            job.status = "failed"
            job.error = str(exc)
            job.updated_at = datetime.now(timezone.utc)
    finally:
        if _delete_upload_after_analysis:
            try:
                Path(job.video_path).unlink(missing_ok=True)
            except Exception as exc:
                log.warning("Failed to remove upload file '%s': %s", job.video_path, exc)


def _prune_old_jobs() -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=_result_ttl_minutes)
    with _lock:
        expired = [job_id for job_id, job in _jobs.items() if job.updated_at < cutoff]
        for job_id in expired:
            _jobs.pop(job_id, None)


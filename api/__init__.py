from flask import Flask

from api.config import AppConfig
from api.utils.logging import configure_logging, get_logger


def _run_model_preflight(app: Flask, log) -> None:
    preflight = {
        "ok": True,
        "message": "Model assets ready.",
    }

    if not app.config.get("MODEL_PREFLIGHT_ON_STARTUP", True):
        preflight["message"] = "Model preflight disabled by configuration."
        app.extensions["model_preflight"] = preflight
        log.info(preflight["message"])
        return

    try:
        from src.pipeline.interview_pipeline import ensure_model_assets

        ensure_model_assets(
            rf_model_path=app.config["RF_MODEL_PATH"],
            landmarker_path=app.config["LANDMARKER_MODEL_PATH"],
            rf_model_url=app.config.get("RF_MODEL_URL") or None,
            landmarker_url=app.config.get("LANDMARKER_MODEL_URL") or None,
        )
        log.info("Model preflight passed: required assets are available.")
    except Exception as exc:
        preflight["ok"] = False
        preflight["message"] = f"Model preflight failed: {exc}"
        log.error(preflight["message"], exc_info=True)
        if app.config.get("MODEL_PREFLIGHT_STRICT", False):
            raise RuntimeError(preflight["message"]) from exc
    finally:
        app.extensions["model_preflight"] = preflight


def create_app(config_obj: AppConfig | None = None) -> Flask:
    config = config_obj or AppConfig()

    app = Flask(
        __name__,
        template_folder=str(config.TEMPLATE_FOLDER),
        static_folder=str(config.STATIC_FOLDER),
    )
    app.config.update(config.to_flask_config())

    configure_logging(log_dir=config.LOG_DIR, level=config.LOG_LEVEL)
    log = get_logger(__name__)
    log.info("Application initialized")
    _run_model_preflight(app, log)

    from api.errors import register_error_handlers
    from api.routes.web import web_bp

    app.register_blueprint(web_bp)
    register_error_handlers(app)
    return app


from flask import Flask

from api.config import AppConfig
from api.utils.logging import configure_logging, get_logger


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

    from api.errors import register_error_handlers
    from api.routes.web import web_bp

    app.register_blueprint(web_bp)
    register_error_handlers(app)
    return app


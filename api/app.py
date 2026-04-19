import os
import sys
from pathlib import Path

# Supports both:
# 1) python -m api.app   (preferred)
# 2) python app.py       (when run inside /api)
if __package__:
    from . import create_app
else:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from api import create_app

app = create_app()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=bool(app.config.get("DEBUG", False)),
    )


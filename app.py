from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trispectra.web import APP_CSS, APP_THEME, build_app


app = build_app()


if __name__ == "__main__":
    app.launch(css=APP_CSS, theme=APP_THEME)

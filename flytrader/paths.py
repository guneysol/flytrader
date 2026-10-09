"""Where flytrader keeps downloads, outputs and the .env file.

Defaults to the repo checkout (editable install) or the current directory,
override with FLYTRADER_HOME.
"""

import os
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("FLYTRADER_HOME") or (_repo if (_repo / "pyproject.toml").exists() else Path.cwd()))
DATA = ROOT / "data"
OUT = ROOT / "out"
ENV_FILE = ROOT / ".env"

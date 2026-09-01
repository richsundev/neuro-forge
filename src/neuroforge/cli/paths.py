from __future__ import annotations

import os
from pathlib import Path


def state_dir() -> Path:
    return Path(os.environ.get("NEUROFORGE_STATE_DIR", "./neuroforge_state"))

from __future__ import annotations

import os
from pathlib import Path


EMAS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANNING_MODEL_PATH = os.getenv(
    "PLANNING_MODEL_PATH",
    str(EMAS_ROOT / "Qwen3.5-9B"),
)

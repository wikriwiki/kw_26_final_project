"""Project-wide configuration loaded from environment variables (and optional .env)."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"


LLM_MODE: str = os.getenv("LLM_MODE", "qwen")
LLM_ENDPOINT: str = os.getenv("LLM_ENDPOINT", "http://localhost:30000/v1")
LLM_METRICS_ENDPOINT: str = os.getenv("LLM_METRICS_ENDPOINT", "http://localhost:30000/metrics")
LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "300"))
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))

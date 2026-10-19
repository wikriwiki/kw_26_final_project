"""Project-root conftest: ensure `scripts.*` imports resolve when running pytest."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 테스트가 부르는 가짜 AI 호출이 제출용 호출 로그(output/ai_call_logs)에 섞이면 안 된다.
import os
import tempfile

os.environ["AI_CALL_LOG_DIR"] = tempfile.mkdtemp(prefix="ai_call_logs_test_")

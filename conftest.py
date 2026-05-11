"""pytest 설정 — repo 루트를 sys.path에 추가해서 `src.*` import 가능하게.

pyproject.toml로 패키지 설치 없이도 테스트가 돌아가게 합니다.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

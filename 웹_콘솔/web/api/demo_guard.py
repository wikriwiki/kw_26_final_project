"""공개 시연 배포(`WEB_PUBLIC_DEMO=1`)의 문지기.

화면은 버셀에, 서버는 이 PC 에 두고 터널로 공개할 때 쓴다. 아무나 들어오므로
- 새 실행·보고서 생성·정책 수정·동기화·모델 연결 확인 같은 변경 요청은 막고,
- 인터뷰(모델을 부르는 요청)는 주소마다 한 시간에, 서버 전체로 하루에 횟수를 묶는다.
읽기(GET)와 집단 효과 계산(모델을 부르지 않음)은 그대로 둔다.
"""
from __future__ import annotations

import os
import re
import threading
import time
from collections import defaultdict, deque

# 시연에서 허용하는 쓰기 요청. 나머지 POST·PUT·DELETE 는 403
ALLOWED_POST = re.compile(
    r"^/api/(interview"
    r"|experiments/[^/]+/agents/[^/]+/interview"
    r"|experiments/[^/]+/query/(stats|effect|interviews))$"
)
# 이 중 모델을 부르는 것 — 횟수를 센다
COUNTED_POST = re.compile(
    r"^/api/(interview"
    r"|experiments/[^/]+/agents/[^/]+/interview"
    r"|experiments/[^/]+/query/interviews)$"
)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class InterviewLimiter:
    """주소별 한 시간 창 + 서버 전체 하루 창. 메모리에만 두므로 서버를 다시 켜면 비워진다."""

    def __init__(self, per_ip_hour: int | None = None, per_day: int | None = None) -> None:
        self.per_ip_hour = per_ip_hour if per_ip_hour is not None else _int_env("WEB_DEMO_PER_IP_HOUR", 20)
        self.per_day = per_day if per_day is not None else _int_env("WEB_DEMO_PER_DAY", 400)
        self.by_ip: dict[str, deque[float]] = defaultdict(deque)
        self.all: deque[float] = deque()
        self.lock = threading.Lock()

    def allow(self, client: str, now: float | None = None) -> str | None:
        """허용하면 None, 막으면 사유."""
        now = time.time() if now is None else now
        with self.lock:
            mine = self.by_ip[client]
            while mine and now - mine[0] > 3600:
                mine.popleft()
            while self.all and now - self.all[0] > 86400:
                self.all.popleft()
            if len(self.all) >= self.per_day:
                return "오늘 인터뷰 횟수를 다 썼습니다. 내일 다시 시도해 주세요."
            if len(mine) >= self.per_ip_hour:
                return "인터뷰를 너무 자주 요청했습니다. 잠시 뒤에 다시 시도해 주세요."
            mine.append(now)
            self.all.append(now)
            return None


def client_address(headers, fallback: str | None) -> str:
    """버셀·터널을 거쳐 오므로 X-Forwarded-For 의 첫 주소가 실제 방문자다."""
    forwarded = headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return fallback or "unknown"


def max_sample() -> int | None:
    """시연에서 집단 인터뷰 한 번에 묻는 최대 인원. 설정이 없으면 None(제한 없음)."""
    value = os.environ.get("WEB_DEMO_MAX_SAMPLE")
    return int(value) if value and value.isdigit() else None

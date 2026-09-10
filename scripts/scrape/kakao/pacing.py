"""휴먼 페이싱 + 차단 탐지. Poisson 분포 sleep으로 봇 패턴 회피."""
from __future__ import annotations

import datetime as dt
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Pacer:
    """Poisson 페이서. uniform 간격은 봇 신호 → exponential 분포 필수."""

    # 평균 inter-arrival time (초). 카카오는 비교적 관대 → 0.6~1.0 권장.
    mean_interval: float = 0.7
    # 최소 sleep (너무 빠른 burst 차단용)
    min_sleep: float = 0.18
    # 최대 sleep (긴 stall 방지)
    max_sleep: float = 8.0

    # 차단 감지 카운터 (최근 N개 응답 기준)
    _recent: deque = field(default_factory=lambda: deque(maxlen=50))
    _ban_streak: int = 0

    def wait(self) -> None:
        """Exponential 분포로 sleep. 차단 streak 누적 시 가중치 증가."""
        u = random.random() * 0.999 + 0.001  # avoid log(0)
        s = -self.mean_interval * math.log(u)
        if self._ban_streak > 0:
            s *= (1 + 0.5 * self._ban_streak)
        s = max(self.min_sleep, min(self.max_sleep, s))
        # 추가 jitter (±15%) — 패턴 검출 회피
        s *= 0.85 + 0.30 * random.random()
        time.sleep(s)

    def report(self, status: int) -> None:
        """응답 코드 보고. 차단 streak 갱신."""
        self._recent.append((time.time(), status))
        if status in (429, 403):
            self._ban_streak = min(8, self._ban_streak + 1)
        elif 200 <= status < 300:
            self._ban_streak = max(0, self._ban_streak - 1)

    def ban_rate(self) -> float:
        """최근 50건 중 차단(429/403) 비율."""
        if not self._recent:
            return 0.0
        bans = sum(1 for _, s in self._recent if s in (429, 403))
        return bans / len(self._recent)

    def cooldown_if_overheated(self, threshold: float = 0.1) -> bool:
        """ban_rate가 threshold 초과 시 긴 휴식. 패턴 깨기. 휴식하면 True."""
        if self.ban_rate() > threshold:
            sleep_min = random.uniform(5, 15)
            print(f"[pacer] ban_rate={self.ban_rate():.1%} → cooldown {sleep_min:.1f}min")
            time.sleep(sleep_min * 60)
            self._recent.clear()
            self._ban_streak = 0
            return True
        return False


@dataclass
class DailyQuota:
    """일일 호출 쿼터 — SQLite 등에 영속화는 store.py에서 처리."""

    limit_per_day: int = 30_000
    _count: int = 0
    _today: dt.date = field(default_factory=dt.date.today)

    def hit(self) -> None:
        today = dt.date.today()
        if today != self._today:
            self._count = 0
            self._today = today
        self._count += 1

    def remaining(self) -> int:
        if dt.date.today() != self._today:
            return self.limit_per_day
        return max(0, self.limit_per_day - self._count)

    def must_stop(self) -> bool:
        return self.remaining() <= 0

"""세부업종으로 덮어쓴 금액은 그 L1 기본값에서 빠진다 — 같은 돈을 두 업종에 세지 않는다."""
from __future__ import annotations

import json

from scripts.report import score_p012_two_arm as S


def _write(tmp_path, arm, rows):
    with open(tmp_path / ("%s.sector.ledger.jsonl" % arm), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (tmp_path / ("%s.cashback.ledger.jsonl" % arm)).write_text("", encoding="utf-8")


def test_appliance_spend_is_not_also_counted_as_retail(tmp_path):
    row = {"aid": "A", "day": "2021-10-01", "total_spent": 30000, "offline_spent": 30000,
           "online_spent": 0, "sangsaeng_eligible_offline_spent": 30000,
           "by_l1": {"쇼핑": 30000}, "by_sub": {"가전·통신": 20000, "의류": 10000},
           "eligible_by_l1": {"쇼핑": 30000}, "eligible_by_sub": {"가전·통신": 20000, "의류": 10000}}
    _write(tmp_path, "off", [row])
    sec, _ = S.load_arm(tmp_path, "off")
    assert sec["A"]["kdi:가전·가구"] == 20000
    assert sec["A"]["kdi:유통"] == 10000          # 고치기 전에는 30000 (가전 20000 을 한 번 더)
    assert sec["A"]["kdiE:유통"] == 10000
    assert sum(v for k, v in sec["A"].items() if k.startswith("kdi:")) == 30000


def test_lodging_under_other_is_not_counted_twice(tmp_path):
    row = {"aid": "B", "day": "2021-10-01", "total_spent": 50000, "offline_spent": 50000,
           "online_spent": 0, "sangsaeng_eligible_offline_spent": 50000,
           "by_l1": {"기타": 50000}, "by_sub": {"숙박": 50000},
           "eligible_by_l1": {"기타": 50000}, "eligible_by_sub": {"숙박": 50000}}
    _write(tmp_path, "off", [row])
    sec, _ = S.load_arm(tmp_path, "off")
    assert sec["B"]["kdi:여행·레저"] == 50000
    assert sec["B"]["kdi:기타"] == 0


def test_eligible_sector_keys_are_readable():
    """K3~K8 은 적립 대상분(kdiE:)으로 잰다 — series 가 그 키를 읽어야 한다(2026-10-06)."""
    import importlib
    import sys
    from collections import defaultdict
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "report"))
    s = importlib.import_module("score_p012_two_arm")
    sec = defaultdict(lambda: defaultdict(float))
    sec["a"]["kdiE:요식"] = 5.0
    assert s.series(sec, {}, ["a"], "kdiE:요식") == [5.0]
    assert s.METRIC_KEY["sector_arm_diff:요식"] == "kdiE:요식"

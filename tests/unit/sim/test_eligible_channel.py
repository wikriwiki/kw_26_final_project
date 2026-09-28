"""적립 몫을 정책에 반응하게 한 변경이 **수준을 옮기지 않는지** 못 박는다.

자를 바꾼 총합은 진전이 아니다. 그래서 이 변경의 첫 조건은 '기준 런에서 현행과
같은 값' 이고, 두 번째 조건은 '계획이 움직이면 적립분만 움직인다' 다.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "sim"))


def load(monkeypatch, **env):
    """환경변수를 세우고 consumption 을 다시 읽는다 — 상수가 모듈 로드 시 굳는다."""
    for k in ("EXP_ELIGIBLE_CHANNEL", "EXP_SPLIT_ANCHOR", "EXP_DONG_SHARE_FILE",
              "EXP_SHARE_BASE", "EXP_ANCHOR_OVERSTATE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import consumption
    return importlib.reload(consumption)


def share_file(tmp_path, rows):
    p = tmp_path / "dong.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    return str(p)


def test_median_dong_keeps_the_current_level_identical(monkeypatch, tmp_path):
    """중앙값 동에서는 적립 몫이 SHARE_BASE 그대로여야 한다 — 항등."""
    f = share_file(tmp_path, {
        "11110515": {"eligible": 0.30, "excluded": 0.30},   # 0.50
        "11140550": {"eligible": 0.40, "excluded": 0.20},   # 0.667  <- 중앙
        "11170510": {"eligible": 0.50, "excluded": 0.10},   # 0.833
    })
    c = load(monkeypatch, EXP_DONG_SHARE_FILE=f)
    s, src = c.eligible_share_for_dong("11140550")
    assert src == "bdc_dong"
    assert s == pytest.approx(c.SHARE_BASE, rel=1e-9)

    # 앵커 x 0.2535 와 (앵커/2.40) x SHARE_BASE 가 같은 값이어야 한다.
    anchor = 1_000_000
    assert (anchor / c.ANCHOR_OVERSTATE) * s == pytest.approx(
        anchor * c.ELIGIBLE_SHARE_SEOUL, rel=1e-6)


def test_share_splits_by_dong_and_stays_inside_bounds(monkeypatch, tmp_path):
    f = share_file(tmp_path, {
        "11110515": {"eligible": 0.10, "excluded": 0.90},
        "11140550": {"eligible": 0.40, "excluded": 0.20},
        "11170510": {"eligible": 0.90, "excluded": 0.10},
    })
    c = load(monkeypatch, EXP_DONG_SHARE_FILE=f)
    lo, _ = c.eligible_share_for_dong("11110515")
    mid, _ = c.eligible_share_for_dong("11140550")
    hi, _ = c.eligible_share_for_dong("11170510")
    assert lo < mid < hi                       # 동에 따라 갈린다
    assert c.SHARE_FLOOR <= lo and hi <= 1.0   # 경계를 넘지 않는다


def test_unknown_dong_falls_back_to_the_constant(monkeypatch, tmp_path):
    """**모르면 안 움직인다** — 표에 없는 동은 전국 상수를 쓴다."""
    f = share_file(tmp_path, {"11140550": {"eligible": 0.40, "excluded": 0.20}})
    c = load(monkeypatch, EXP_DONG_SHARE_FILE=f)
    s, src = c.eligible_share_for_dong("99999999")
    assert src == "share_base_constant" and s == c.SHARE_BASE
    s, src = c.eligible_share_for_dong(None)
    assert src == "share_base_constant"


def test_missing_share_file_falls_back_rather_than_raising(monkeypatch, tmp_path):
    c = load(monkeypatch, EXP_DONG_SHARE_FILE=str(tmp_path / "absent.json"))
    s, src = c.eligible_share_for_dong("11140550")
    assert src == "share_base_constant" and s == c.SHARE_BASE


def test_dong_code_is_read_from_the_agent_id(monkeypatch):
    c = load(monkeypatch)
    assert c._dong_of("AGT_11140550_M_70대이상_001") == "11140550"
    assert c._dong_of("AGT_NOTADONG_M_30대_001") is None
    assert c._dong_of(None) is None

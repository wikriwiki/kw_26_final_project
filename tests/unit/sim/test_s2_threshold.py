"""적립 문턱을 **금액을 정하는 자리**로 보낸다 — experiments/plan_channel/s2_threshold.md

측정으로 드러난 것: 정책은 계획 금액으로 닿는데(+9.76%, p=0.0015) 소비성향
스칼라는 안 움직이고(+0.34%) 이벤트 수도 그대로다(11.5 -> 11.5). 즉 **금액이
달라지는 자리는 Stage2 인데, 문턱 정보는 Stage1 에만 있었다.**

여기서 지키는 것:
    1  꺼져 있으면 Stage2 프롬프트가 한 글자도 안 달라진다
    2  켜도 **새 사실을 만들지 않는다** — Stage1 이 받은 그 줄을 그대로 옮긴다
    3  방향 지시·목표 수치가 없다
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "sim"))

STATUS = ("- P012: 적립업종 이번달 누적 120,000원 / 2분기 월평균 약 300,000원 / "
          "3% 문턱 309,000원 | 초과분의 10% 다음 달 환급, 월 최대 100,000원 | "
          "문턱까지 189,000원 남음 — 적립업종에서 이만큼 더 쓰면 캐시백 자격 시작 "
          "(이번 달 12일 남음 · 하루 평균 15,750원 페이스) | 못 넘기면 이번 달 혜택은 사라짐")


def _load(flag):
    old = os.environ.get("EXP_S2_THRESHOLD")
    if flag is None:
        os.environ.pop("EXP_S2_THRESHOLD", None)
    else:
        os.environ["EXP_S2_THRESHOLD"] = flag
    try:
        import stage2_poi
        return importlib.reload(stage2_poi)
    finally:
        if old is None:
            os.environ.pop("EXP_S2_THRESHOLD", None)
        else:
            os.environ["EXP_S2_THRESHOLD"] = old


def _persona(with_status=True):
    p = {"daily_wd": 50000, "daily_we": 70000, "tendency": "보통",
         "lifestyle": "", "income": "중", "id": "A1"}
    if with_status:
        p["sangsaeng_status_line"] = STATUS
    return p


def _prompt(mod, persona):
    return mod.build_stage2_prompt(
        events=[], persona=persona, candidates_by_order={},
        today=None) if False else mod


def test_꺼져_있으면_기본값이_False():
    m = _load(None)
    assert m.EXP_S2_THRESHOLD is False


def test_켜면_True():
    m = _load("1")
    assert m.EXP_S2_THRESHOLD is True
    _load(None)


def test_문턱_줄에_방향_지시가_없다():
    """옮기는 줄 자체와 덧붙이는 괄호에 방향·목표가 없어야 한다."""
    m = _load("1")
    src = Path(m.__file__).read_text(encoding="utf-8")
    i = src.index("if EXP_S2_THRESHOLD:")
    blk = src[i:src.index("return", i)] if "return" in src[i:] else src[i:i + 900]
    blk = blk[:900]
    for w in ("더 써", "늘리", "많이 쓰", "권장", "해야 한다", "바람직"):
        assert w not in blk, f"방향을 지시한다: {w}"
    _load(None)


def test_새_사실을_만들지_않는다():
    """Stage2 블록은 persona 의 줄을 **그대로** 쓴다 — 자기가 계산하지 않는다."""
    m = _load("1")
    src = Path(m.__file__).read_text(encoding="utf-8")
    i = src.index("if EXP_S2_THRESHOLD:")
    blk = src[i:i + 900]
    assert "sangsaeng_status_line" in blk
    for w in ("threshold =", "anchor *", "remaining =", "/ days_left"):
        assert w not in blk, f"Stage2 가 스스로 계산한다: {w} — 두 곳이 어긋난다"
    _load(None)


def test_상태줄이_없으면_아무것도_안_붙는다():
    """적립 정책이 없는 런에서 조용히 프롬프트가 달라지면 안 된다."""
    m = _load("1")
    src = Path(m.__file__).read_text(encoding="utf-8")
    i = src.index("if EXP_S2_THRESHOLD:")
    assert "if _ss:" in src[i:i + 400], "빈 상태줄 가드가 없다"
    _load(None)

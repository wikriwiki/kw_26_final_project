"""동별 적립업종 몫 — scripts/report/dong_eligible_share.py + 매핑 파일

가장 중요한 시험은 `test_모름을_한쪽에_넣지_않는다` 와
`test_매핑에_없는_이름은_모름으로_간다` 다. 근거를 못 댄 업종을 적립이나 제외에
밀어 넣으면 **그 지표는 우리가 만든 수가 된다.** 사용자가 못 박은 규칙이다 —
"없는 말을 지어내거나 임의의 값을 부여해서 결과가 나오게 하면 안 된다".
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MAP = ROOT / "data" / "sangsaeng" / "bdc_industry_arm.json"


def _mod():
    p = ROOT / "scripts" / "report" / "dong_eligible_share.py"
    spec = importlib.util.spec_from_file_location("dong_eligible_share", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


D = _mod()


# ---------------------------------------------------------------- 매핑 파일

def test_모든_항목에_근거가_붙어_있다():
    """근거 없는 분류는 추측이다. 문자열이 비어 있으면 안 된다."""
    m = json.loads(MAP.read_text(encoding="utf-8"))
    for arm, table in m.items():
        if arm.startswith("_"):
            continue
        for name, why in table.items():
            assert isinstance(why, str) and len(why) >= 4, \
                "%s/%s 에 근거가 없다" % (arm, name)


def test_규칙_문장을_파일이_담고_있다():
    m = json.loads(MAP.read_text(encoding="utf-8"))
    for word in ("대형마트", "면세점", "신차구입", "비소비성지출"):
        assert word in m["_rule"], "규칙 문장에 %s 가 없다" % word


def test_한_업종이_두_arm에_동시에_있지_않다():
    m = json.loads(MAP.read_text(encoding="utf-8"))
    seen = {}
    for arm, table in m.items():
        if arm.startswith("_"):
            continue
        for name in table:
            assert name not in seen, "%s 가 %s 와 %s 에 동시에 있다" % (name, seen.get(name), arm)
            seen[name] = arm


def test_규칙에_명시된_제외업종이_제외에_있다():
    """규칙 문장에 이름이 그대로 나오는 것은 제외여야 한다."""
    arms = D.load_arms()
    for name in ("대형마트", "백화점", "면세점", "전자상거래(다품목취급)"):
        assert arms.get(name) in D.EXCLUDED_ARMS, "%s 가 제외가 아니다" % name


def test_섞인_이름은_모름에_있다():
    """규칙이 일부만 제외하는 이름을 한쪽으로 몰면 안 된다."""
    arms = D.load_arms()
    for name in ("가전", "실내/실외골프장", "상품권/복권", "할인점/슈퍼마켓/양판점"):
        assert arms.get(name) == "unknown", \
            "%s 는 제외·적립이 묶인 이름이다 — unknown 이어야 한다" % name


def test_잔여버킷은_모름이다():
    arms = D.load_arms()
    assert arms.get("ZZ_나머지") == "unknown"
    assert arms.get("결제대행(PG)") == "unknown"


# ---------------------------------------------------------------- 산술

def test_세_몫의_합이_원본_합과_같다():
    arms = D.load_arms()
    r = D.split({"한식": 0.3, "대형마트": 0.2, "ZZ_나머지": 0.5}, arms)
    assert r["eligible"] == pytest.approx(0.3)
    assert r["excluded"] == pytest.approx(0.2)
    assert r["unknown"] == pytest.approx(0.5)
    assert r["total"] == pytest.approx(1.0)


def test_모름을_한쪽에_넣지_않는다():
    """**가장 중요한 시험.** 모름이 적립이나 제외로 새면 안 된다."""
    arms = D.load_arms()
    r = D.split({"ZZ_나머지": 0.4, "결제대행(PG)": 0.3, "가전": 0.3}, arms)
    assert r["eligible"] == 0.0 and r["excluded"] == 0.0
    assert r["unknown"] == pytest.approx(1.0)


def test_매핑에_없는_이름은_모름으로_간다():
    """새 업종이 조용히 적립으로 세지면 몫이 부풀려진다."""
    arms = D.load_arms()
    r = D.split({"듣도보도못한업종": 0.6, "한식": 0.4}, arms)
    assert r["unknown"] == pytest.approx(0.6)
    assert r["_unmapped"] == {"듣도보도못한업종": pytest.approx(0.6)}


def test_숫자가_아닌_값은_버린다():
    arms = D.load_arms()
    r = D.split({"한식": "많이", "대형마트": None, "편의점": 0.5}, arms)
    assert r["total"] == pytest.approx(0.5)


# ---------------------------------------------------------------- 자료 읽기

def test_별표_키는_쓰지_않는다():
    """'*' 의 industry_ratio 는 지출 구성이 아니다(한식 0.0003 · 세금공과금 0.125)."""
    z = ROOT / "output" / "stats" / "bdc_based_results.zip"
    if not z.exists():
        pytest.skip("BDC zip 이 없다")
    dongs = D.load_dongs(z)
    assert "*" not in dongs
    assert all(k.isdigit() for k in dongs)
    assert len(dongs) > 400, "행정동이 400개보다 적다 — 자료가 잘렸는지 보라"

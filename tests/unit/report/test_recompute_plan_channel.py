"""원장 위 다시 셈이 엔진과 같은 산술인지 — scripts/report/recompute_plan_channel.py

**가장 중요한 시험은 `test_앵커가_약분되지_않는다` 다.** 기준선을 `REF x 앵커` 로
잡으면 `앵커 x 계획/(REF x 앵커) = 계획/REF` 가 되어 앵커가 사라진다. 엔진 쪽에서
한 번 저지른 실수이고(tests/unit/sim/test_plan_drives_total.py), 다시 셈 쪽에서도
똑같이 저지를 수 있다.

그다음으로 중요한 것은 `test_기준선이_창과_겹치면_거부한다` 다. 겹치면 그 창의
계획/기준선이 1 로 못 박혀 **그 팔만 안 움직인다** — 순환이다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _mod():
    p = ROOT / "scripts" / "report" / "recompute_plan_channel.py"
    spec = importlib.util.spec_from_file_location("recompute_plan_channel", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = _mod()


def _row(anchor, plan, total=None):
    return {"cm_anchor_total": anchor, "cm_planned_total": plan,
            "cm_today_total": total if total is not None else max(anchor, plan),
            "status": "ok"}


# ---------------------------------------------------------------- 산술

def test_앵커가_약분되지_않는다():
    """기준선이 앵커에 비례하면 앵커가 사라진다. 그러면 안 된다.

    계층이 다른 두 사람(앵커 10만 / 20만)이 **같은 정책 반응**(계획이 평소의
    1.5배)을 보이면, 고침 총액의 비는 앵커의 비를 그대로 지켜야 한다.
    """
    lo, hi, scale = 0.5, 2.0, 1.0
    작은이 = R.multiplier(_row(100_000, 150_000), 100_000, lo, hi, scale)[0]
    큰이 = R.multiplier(_row(200_000, 300_000), 200_000, lo, hi, scale)[0]
    작은_총액 = max(100_000, 150_000) * 작은이
    큰_총액 = max(200_000, 300_000) * 큰이
    assert 큰_총액 == pytest.approx(2 * 작은_총액), "앵커의 계층이 지켜져야 한다"
    assert 작은_총액 == pytest.approx(150_000), "앵커 x 1.5 = 15만"


def test_평소대로면_앵커에_머문다():
    """계획이 평소와 같으면 비는 1 이고 총액은 앵커 x SCALE 이다."""
    m, hit = R.multiplier(_row(100_000, 40_000), 40_000, 0.5, 2.0, 1.0)
    assert max(100_000, 40_000) * m == pytest.approx(100_000)
    assert hit is False


def test_클램프가_양쪽에서_문다():
    아래 = R.multiplier(_row(100_000, 10_000), 100_000, 0.5, 2.0, 1.0)
    위 = R.multiplier(_row(100_000, 900_000), 100_000, 0.5, 2.0, 1.0)
    assert max(100_000, 10_000) * 아래[0] == pytest.approx(50_000)
    assert max(100_000, 900_000) * 위[0] == pytest.approx(200_000)
    assert 아래[1] is True and 위[1] is True


def test_기준선이_없으면_손대지_않는다():
    m, hit = R.multiplier(_row(100_000, 300_000), None, 0.5, 2.0, 0.88)
    assert m == 1.0, "되살릴 수 없으면 원장 그대로 둔다"
    assert hit is None, "클램프가 아니라 결손으로 세야 한다"


def test_앵커가_0이면_손대지_않는다():
    assert R.multiplier(_row(0, 50_000), 40_000, 0.5, 2.0, 1.0) == (1.0, None)


def test_SCALE_이_곱해진다():
    m, _ = R.multiplier(_row(100_000, 40_000), 40_000, 0.5, 2.0, 0.5)
    assert max(100_000, 40_000) * m == pytest.approx(50_000)


def test_곱수는_기록된_총액에_걸린다():
    """온라인 분리가 총액을 깎아 놨어도 비율은 같아야 한다 — 분자·분모에서 약분."""
    깎인총액 = 25_000          # max(앵커,계획)=10만인데 원장엔 2.5만만 남았다
    m, _ = R.multiplier(_row(100_000, 200_000, total=깎인총액), 100_000, 0.5, 2.0, 1.0)
    assert 깎인총액 * m == pytest.approx(25_000 * (100_000 * 2.0 / 200_000))


# ---------------------------------------------------------------- 기준선

def test_기준선은_중앙값이다():
    per = {"a": {"d1": _row(1, 100), "d2": _row(1, 200), "d3": _row(1, 9_000)}}
    assert R.build_baseline(per, ["d1", "d2", "d3"])["a"] == 200, "평균이면 큰 날에 끌려간다"


def test_daytype_은_평일과_주말을_가른다():
    per = {"a": {"2021-10-04": _row(1, 60_000),   # 월
                 "2021-10-05": _row(1, 70_000),   # 화
                 "2021-10-09": _row(1, 90_000)}}  # 토
    b = R.build_baseline(per, ["2021-10-04", "2021-10-05", "2021-10-09"], "daytype")
    assert b["a"]["wd"] == 65_000
    assert b["a"]["we"] == 90_000
    assert R._base_for(b, "a", "2021-10-06", "daytype") == 65_000   # 수 -> 평일
    assert R._base_for(b, "a", "2021-10-10", "daytype") == 90_000   # 일 -> 주말


def test_daytype_에_없는_요일종류는_대신하지_않는다():
    """평일 기준선밖에 없는데 주말을 재면, 평일 값으로 때우면 요일효과가 곱수로 샌다."""
    per = {"a": {"2021-10-04": _row(1, 60_000)}}
    b = R.build_baseline(per, ["2021-10-04"], "daytype")
    assert R._base_for(b, "a", "2021-10-09", "daytype") is None


def test_섞인_요일_기준선이_클램프를_문다():
    """이 시험이 '왜 daytype 이 필요한가' 를 값으로 보인다.

    금(계획 49k) · 토(계획 99k) 로 기준선을 세우면 중앙이 74k 가 되어, 평일
    비교 창(계획 49k)에서 비가 0.66 으로 내려앉는다. 평일만으로 세우면 1.0 이다.
    """
    per = {"a": {"2021-10-01": _row(94_000, 49_000),    # 금
                 "2021-10-02": _row(62_000, 99_000)}}   # 토
    섞임 = R.build_baseline(per, ["2021-10-01", "2021-10-02"], "abs")["a"]
    평일만 = R.build_baseline(per, ["2021-10-01"], "abs")["a"]
    assert 섞임 == 74_000 and 평일만 == 49_000
    비_섞임 = 49_000 / 섞임
    비_평일 = 49_000 / 평일만
    assert 비_섞임 < 0.7 and 비_평일 == pytest.approx(1.0)


# ---------------------------------------------------------------- 관문

def test_기준선이_창과_겹치면_거부한다(tmp_path, monkeypatch, capsys):
    d = tmp_path / "metrics"
    d.mkdir()
    (d / "day_2021-10-13.jsonl").write_text(
        json.dumps({"aid": "a", **_row(1, 1)}) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "x", "--metrics", str(tmp_path), "--baseline", "2021-10-13",
        "--off", "2021-10-13", "--on", "2021-10-27"])
    assert R.main() == 2
    assert "겹친다" in capsys.readouterr().out


def test_모든_날짜를_가진_사람만_센다(tmp_path):
    d = tmp_path / "metrics"
    d.mkdir()
    for day, rows in (("2021-10-04", ["a", "b"]), ("2021-10-13", ["a", "b"]),
                      ("2021-10-27", ["a"])):
        (d / ("day_%s.jsonl" % day)).write_text(
            "".join(json.dumps({"aid": x, **_row(100_000, 50_000)}) + "\n" for x in rows),
            encoding="utf-8")
    per = R.load_ledger(str(tmp_path))
    need = {"2021-10-04", "2021-10-13", "2021-10-27"}
    assert sorted(x for x, v in per.items() if need <= set(v)) == ["a"]


def test_status_가_ok_아니면_뺀다(tmp_path):
    d = tmp_path / "metrics"
    d.mkdir()
    (d / "day_2021-10-04.jsonl").write_text(
        json.dumps({"aid": "a", **_row(1, 1)}) + "\n"
        + json.dumps({"aid": "b", "status": "error"}) + "\n", encoding="utf-8")
    assert list(R.load_ledger(str(tmp_path))) == ["a"]


# ---------------------------------------------------------------- 지갑 관문

def test_잔고0_비율을_센다():
    per = {"a": {"d": {"balance": 0}}, "b": {"d": {"balance": 5}},
           "c": {"d": {"balance": -3}}, "d": {"d": {"balance": 100}}}
    assert R.broke_rate(per, ["a", "b", "c", "d"], "d") == 0.5


def test_잔고가_없는_칸은_분모에서_뺀다():
    per = {"a": {"d": {"balance": 0}}, "b": {"d": {}}}
    assert R.broke_rate(per, ["a", "b"], "d") == 1.0


def _write(tmp_path, day, rows):
    d = tmp_path / "metrics"
    d.mkdir(exist_ok=True)
    (d / ("day_%s.jsonl" % day)).write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _agent(aid, bal):
    return {"aid": aid, "status": "ok", "cm_anchor_total": 100_000,
            "cm_planned_total": 50_000, "cm_today_total": 25_000,
            "cm_today_total_incl_online": 100_000, "cm_personal_total": 25_000,
            "balance": bal}


def _argv(tmp_path, extra=()):
    return ["x", "--metrics", str(tmp_path),
            "--baseline", "2021-10-04", "--off", "2021-10-13",
            "--on", "2021-10-28", "--boot", "10", *extra]


def test_지갑이_마르면_수를_내지_않는다(tmp_path, monkeypatch, capsys):
    """**가장 중요한 관문.** 이걸 지나치면 파산을 정책 효과로 읽는다."""
    for day in ("2021-10-04", "2021-10-13"):
        _write(tmp_path, day, [_agent(x, 500_000) for x in "abcd"])
    _write(tmp_path, "2021-10-28",
           [_agent("a", 0), _agent("b", 0), _agent("c", 0), _agent("d", 500_000)])
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))
    assert R.main() == 3
    out = capsys.readouterr().out
    assert "지갑이 말랐다" in out and "75.0%" in out
    assert "현행" not in out, "거부했으면 수를 찍으면 안 된다"


def test_지갑이_멀쩡하면_통과한다(tmp_path, monkeypatch, capsys):
    for day in ("2021-10-04", "2021-10-13", "2021-10-28"):
        _write(tmp_path, day, [_agent(x, 500_000) for x in "abcd"])
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))
    assert R.main() == 0
    assert "현행" in capsys.readouterr().out


def test_한계를_올리면_읽을_수_있다(tmp_path, monkeypatch, capsys):
    """막되 잠그지는 않는다 — 올려서 읽었으면 보고에 적을 일이다."""
    for day in ("2021-10-04", "2021-10-13"):
        _write(tmp_path, day, [_agent(x, 500_000) for x in "abcd"])
    _write(tmp_path, "2021-10-28",
           [_agent("a", 0), _agent("b", 0), _agent("c", 0), _agent("d", 500_000)])
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, ["--max-broke", "0.9"]))
    assert R.main() == 0
    assert "현행" in capsys.readouterr().out


# ---------------------------------------------------------------- 통계

def test_부호검정은_동점을_따로_센다():
    up, dn, eq = R.sign_test([1, 2, 3, 4], [2, 1, 3, 9])
    assert (up, dn, eq) == (2, 1, 1), "동점을 숨기면 유효표본이 부풀려진다"


def test_부트스트랩은_사람단위로_함께_재표집한다():
    """같은 사람의 두 창이다. 따로 재표집하면 쌍이 깨져 구간이 좁아진다."""
    off = [100.0] * 50
    on = [110.0] * 50
    lo, hi = R.boot_pct(off, on, 500)
    assert lo == pytest.approx(10.0) and hi == pytest.approx(10.0), \
        "모두가 똑같이 10% 오르면 불확실성이 없어야 한다"


def test_부트스트랩이_흩어짐을_잡는다():
    off = [100.0] * 50
    on = [200.0] * 25 + [50.0] * 25
    lo, hi = R.boot_pct(off, on, 1000)
    assert lo < 25.0 < hi, "섞여 있으면 구간이 0 근처까지 벌어져야 한다"

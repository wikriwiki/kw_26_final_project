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


# ---------------------------------------------------------------- 표류

def test_평평하면_표류가_0이다():
    g, r2, _ = R.drift_per_day([("d%d" % i, 100.0) for i in range(6)])
    assert g == pytest.approx(0.0)


def test_내려가면_표류가_음수고_직선이다():
    g, r2, (h1, h2) = R.drift_per_day([("d%d" % i, 100.0 - i) for i in range(6)])
    assert g == pytest.approx(-1 / 97.5, rel=1e-6)
    assert r2 == pytest.approx(1.0)
    assert h1 == pytest.approx(h2), "진짜 직선이면 앞뒤 기울기가 같다"


def test_점이_모자라면_표류를_재지_않는다():
    assert R.drift_per_day([("a", 1.0), ("b", 2.0)])[0] == 0.0


def test_예열은_직선이_아니라고_말한다():
    """**이 시험이 오늘 잡은 함정이다.** P013 실제 값 그대로.

    첫 3일 +7.1% 오르고 그 뒤 8일 +0.2% 로 평평하다. 전 구간에 직선을 맞추면
    하루 +1.285% 가 나오고, 7일치를 덜면 고침의 +8.99% 가 −0.36% 로 지워진다.
    없는 표류로 있는 효과를 지우는 것이다.
    """
    ys = [123108, 124841, 129216, 131906, 131855, 132884, 132396]
    g, r2, (h1, h2) = R.drift_per_day([("d%d" % i, float(y)) for i, y in enumerate(ys)])
    assert g == pytest.approx(0.01285, abs=1e-4), "하루 +1.285% — 관측된 그 값"
    assert abs(h1) > 3 * abs(h2), "앞절반이 뒤절반의 12배다 — 예열이다"


def test_예열_뒤만_주면_평평하다고_말한다():
    ys = [131906, 131855, 132884, 132396, 132202, 131471, 132179]
    g, r2, (h1, h2) = R.drift_per_day([("d%d" % i, float(y)) for i, y in enumerate(ys)])
    assert abs(g) < 0.002, "예열을 빼면 하루 0.2% 미만이어야 한다"


def test_표류를_정책창에서_재려_하면_거부한다(tmp_path, monkeypatch, capsys):
    """**이걸 허용하면 정책 효과를 표류로 덜어낸다.** 자기 자신을 빼는 꼴이다."""
    for day in ("2021-10-04", "2021-10-13", "2021-10-28"):
        _write(tmp_path, day, [_agent(x, 500_000) for x in "abcd"])
    monkeypatch.setattr(sys, "argv", _argv(
        tmp_path, ["--trend-days", "2021-10-04,2021-10-28"]))
    assert R.main() == 2
    assert "정책 창이 섞였다" in capsys.readouterr().out


def test_하강_표류를_덜면_효과가_드러난다(tmp_path, monkeypatch, capsys):
    """정책 전 내리막이 있으면, 제자리인 ON 은 사실 정책이 밀어 올린 것이다."""
    for i, day in enumerate(("2021-10-04", "2021-10-05", "2021-10-06",
                             "2021-10-07", "2021-10-08")):
        rows = []
        for x in "abcd":
            r = _agent(x, 500_000)
            for f in ("cm_today_total_incl_online", "cm_personal_total", "cm_today_total"):
                r[f] = int(r[f] * (1 - 0.01 * i))      # 하루 −1%
            rows.append(r)
        _write(tmp_path, day, rows)
    for day in ("2021-10-13", "2021-10-28"):
        _write(tmp_path, day, [_agent(x, 500_000) for x in "abcd"])
    monkeypatch.setattr(sys, "argv", [
        "x", "--metrics", str(tmp_path), "--baseline", "2021-10-04,2021-10-05",
        "--off", "2021-10-13", "--on", "2021-10-28", "--boot", "10",
        "--trend-days", "2021-10-04,2021-10-05,2021-10-06,2021-10-07,2021-10-08"])
    assert R.main() == 0
    out = capsys.readouterr().out
    assert "표류" in out
    assert "직선이 아니다" not in out, "곧은 −1%/일 이면 적용돼야 한다"
    # OFF 와 ON 이 같은 수인데(변화 0%) 표류가 음수였으니 덜면 양수가 된다
    assert "를 덜면:" in out
    body = out.split("를 덜면:")[1]
    assert "현행" in body and "+" in body.split("고침")[0]


def test_예열이_섞인_표류는_적용하지_않는다(tmp_path, monkeypatch, capsys):
    """본런에 걸기 전에 **도구가 스스로 멈춰야** 한다."""
    # 실제 P013 예열 모양 그대로 — 앞이 가파르고 뒤가 평평하다
    ramp = [0.933, 0.946, 0.979, 0.999, 0.999, 1.007, 1.003]
    for mul, day in zip(ramp, ("2021-10-04", "2021-10-05", "2021-10-06",
                               "2021-10-07", "2021-10-08", "2021-10-11",
                               "2021-10-12")):
        rows = []
        for x in "abcd":
            r = _agent(x, 500_000)
            for f in ("cm_today_total_incl_online", "cm_personal_total", "cm_today_total"):
                r[f] = int(r[f] * mul)
            rows.append(r)
        _write(tmp_path, day, rows)
    for day in ("2021-10-13", "2021-10-28"):
        _write(tmp_path, day, [_agent(x, 500_000) for x in "abcd"])
    monkeypatch.setattr(sys, "argv", [
        "x", "--metrics", str(tmp_path), "--baseline", "2021-10-04,2021-10-05",
        "--off", "2021-10-13", "--on", "2021-10-28", "--boot", "10",
        "--trend-days",
        "2021-10-04,2021-10-05,2021-10-06,2021-10-07,2021-10-08,2021-10-11,2021-10-12"])
    assert R.main() == 0
    out = capsys.readouterr().out
    assert "예열이 섞였다" in out, "이유를 정확히 말해야 다음 수가 정해진다"
    assert "를 덜면:" not in out, "멈췄으면 보정한 수를 찍으면 안 된다"


def test_표류가_없으면_그렇다고_말한다(tmp_path, monkeypatch, capsys):
    """평평한 계열을 '예열' 이라고 하면 안 된다 — 날짜를 바꿔도 소용없는 경우다."""
    for day in ("2021-10-04", "2021-10-05", "2021-10-06", "2021-10-07",
                "2021-10-08", "2021-10-13", "2021-10-28"):
        _write(tmp_path, day, [_agent(x, 500_000) for x in "abcd"])
    monkeypatch.setattr(sys, "argv", [
        "x", "--metrics", str(tmp_path), "--baseline", "2021-10-04,2021-10-05",
        "--off", "2021-10-13", "--on", "2021-10-28", "--boot", "10",
        "--trend-days", "2021-10-04,2021-10-05,2021-10-06,2021-10-07,2021-10-08"])
    assert R.main() == 0
    out = capsys.readouterr().out
    assert "표류가 없다" in out
    assert "예열" not in out.split("표류가 없다")[0][-200:]


# ---------------------------------------------------------------- 가짜 ON 창

def test_가짜ON에_진짜ON이_섞이면_거부한다(tmp_path, monkeypatch, capsys):
    """섞이면 정책 효과를 '정책 아닌 몫' 으로 빼 버린다."""
    for day in ("2021-10-04", "2021-10-13", "2021-10-28"):
        _write(tmp_path, day, [_agent(x, 500_000) for x in "abcd"])
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, ["--placebo-on", "2021-10-28"]))
    assert R.main() == 2
    assert "진짜 ON 날짜가 섞였다" in capsys.readouterr().out


def test_가짜ON이_OFF와_겹치면_거부한다(tmp_path, monkeypatch, capsys):
    for day in ("2021-10-04", "2021-10-13", "2021-10-28"):
        _write(tmp_path, day, [_agent(x, 500_000) for x in "abcd"])
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, ["--placebo-on", "2021-10-13"]))
    assert R.main() == 2
    assert "OFF 창과 겹친다" in capsys.readouterr().out


def test_드리프트를_덜어_낸다(tmp_path, monkeypatch, capsys):
    """**이 시험이 P013 에서 놓친 것이다.**

    OFF -> 가짜ON 이 +10%(정책 없음), OFF -> 진짜ON 이 +21% 면 정책 몫은
    1.21/1.10 − 1 = +10% 다. 21% 를 정책 효과로 읽으면 두 배로 부풀린다.
    """
    def rows_at(mul):
        out = []
        for x in "abcd":
            r = _agent(x, 500_000)
            for f in ("cm_today_total_incl_online", "cm_personal_total", "cm_today_total"):
                r[f] = int(r[f] * mul)
            out.append(r)
        return out
    _write(tmp_path, "2021-10-04", rows_at(1.0))
    _write(tmp_path, "2021-10-13", rows_at(1.00))   # OFF
    _write(tmp_path, "2021-10-20", rows_at(1.10))   # 가짜 ON — 정책 전
    _write(tmp_path, "2021-10-28", rows_at(1.21))   # 진짜 ON
    monkeypatch.setattr(sys, "argv", [
        "x", "--metrics", str(tmp_path), "--baseline", "2021-10-04",
        "--off", "2021-10-13", "--on", "2021-10-28",
        "--placebo-on", "2021-10-20", "--boot", "10"])
    assert R.main() == 0
    out = capsys.readouterr().out
    assert "정책 아닌 몫" in out
    assert "+10.00%" in out, "1.21/1.10 − 1 = +10% 가 나와야 한다"
    # 10 은 21 의 절반(10.5)보다 작다 — 여기서는 경고가 뜨면 안 된다
    assert "절반을 넘는다" not in out


def test_드리프트가_절반을_넘으면_경고한다(tmp_path, monkeypatch, capsys):
    """P013 이 이 자리였다 — +6.36% 중 +3.70%p(58%)가 정책 전 상승이었다."""
    def rows_at(mul):
        out = []
        for x in "abcd":
            r = _agent(x, 500_000)
            for f in ("cm_today_total_incl_online", "cm_personal_total", "cm_today_total"):
                r[f] = int(r[f] * mul)
            out.append(r)
        return out
    _write(tmp_path, "2021-10-04", rows_at(1.0))
    _write(tmp_path, "2021-10-13", rows_at(1.000))   # OFF
    _write(tmp_path, "2021-10-20", rows_at(1.037))   # 가짜 ON — P013 의 +3.70%
    _write(tmp_path, "2021-10-28", rows_at(1.064))   # 진짜 ON — P013 의 +6.36%
    monkeypatch.setattr(sys, "argv", [
        "x", "--metrics", str(tmp_path), "--baseline", "2021-10-04",
        "--off", "2021-10-13", "--on", "2021-10-28",
        "--placebo-on", "2021-10-20", "--boot", "10"])
    assert R.main() == 0
    out = capsys.readouterr().out
    assert "절반을 넘는다" in out, "3.70 은 6.40 의 절반을 넘는다 — 경고해야 한다"
    assert "+2.6" in out or "+2.5" in out, "정책 몫 약 +2.57% 가 나와야 한다"


def test_가짜ON이_자를_넘어가도_안_터진다(tmp_path, monkeypatch, capsys):
    """세 자를 돌기 때문에 지역 변수가 바깥 기준선을 덮으면 두 번째 자에서 터진다.

    실제로 `base` 를 덮어써서 `'list' object has no attribute 'get'` 로 터졌다.
    세 자가 **모두** 찍히는지로 잡는다.
    """
    def rows_at(mul):
        out = []
        for x in "abcd":
            r = _agent(x, 500_000)
            for f in ("cm_today_total_incl_online", "cm_personal_total", "cm_today_total"):
                r[f] = int(r[f] * mul)
            out.append(r)
        return out
    _write(tmp_path, "2021-10-04", rows_at(1.0))
    _write(tmp_path, "2021-10-13", rows_at(1.00))
    _write(tmp_path, "2021-10-20", rows_at(1.05))
    _write(tmp_path, "2021-10-28", rows_at(1.15))
    monkeypatch.setattr(sys, "argv", [
        "x", "--metrics", str(tmp_path), "--baseline", "2021-10-04",
        "--off", "2021-10-13", "--on", "2021-10-28",
        "--placebo-on", "2021-10-20", "--boot", "10"])
    assert R.main() == 0
    out = capsys.readouterr().out
    # **이것이 회귀 관문이다** — 덮어쓰면 두 번째 자에서 터져 이 단정이 깨진다.
    for key in R.RULERS:
        assert ("자: %s" % key) in out, "%s 자가 안 찍혔다 — 중간에 터졌다" % key
    assert out.count("가짜ON") >= len(R.RULERS), "자마다 가짜ON 이 나와야 한다"


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

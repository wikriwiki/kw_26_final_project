# -*- coding: utf-8 -*-
"""2026-10-19 — 업종 예산 봉투와 필요 게이지(EXP_CATEGORY_BUDGET).

설계와 시험 목록: data/experiments/L1_BASELINE_FIX_DESIGN_20261019.md 4절 5.
꺼져 있을 때의 고정값(tests/fixtures/category_budget_20261019/off_snapshot_a99e561.json)은
출발점 엔진(P010_simulation a99e561 의 consumption.py)으로 같은 입력을 돌려 만든 것이다.
"""
import copy
import importlib
import json
import os
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
sys.path.insert(0, str(ROOT))

AID = "AGT_11140550_M_70대이상_001"
SNAPSHOT = ROOT / "tests" / "fixtures" / "category_budget_20261019" / "off_snapshot_a99e561.json"
# P010 본런과 같은 소비 설정(tools/run_ab3w.sh common_env + p010 CASE_EXPORTS 중 consumption.py 가 읽는 것)
RUN_ENV = {"EXP_PRICE_FIRST": "1", "EXP_ELIGIBLE_CHANNEL": "1", "EXP_PLAN_DRIVES_TOTAL": "1",
           "EXP_PAYMENT_CHOICE": "0", "EXP_SPREAD_DAYS": "1"}


@pytest.fixture
def mods(monkeypatch):
    """(category_budget, consumption) 을 주어진 설정으로 다시 읽는다. 끝나면 원래 환경으로 되돌려 다시 읽는다."""
    def load(on: bool):
        for k in list(os.environ):
            if k.startswith("EXP_") or k == "COUPON_DAILY_DRAW_RATE":
                monkeypatch.delenv(k)
        for k, v in RUN_ENV.items():
            monkeypatch.setenv(k, v)
        if on:
            monkeypatch.setenv("EXP_CATEGORY_BUDGET", "1")
        import category_budget
        import consumption
        return importlib.reload(category_budget), importlib.reload(consumption)
    yield load
    monkeypatch.undo()
    import category_budget
    import consumption
    importlib.reload(category_budget)
    importlib.reload(consumption)


def _day_mixed():
    return [
        {"time": "07:30", "category": "집", "sub_category": "기상", "poi_id": "HOME"},
        {"time": "09:00", "category": "건강", "sub_category": "의원", "poi_id": "H1", "actual_spent": 15000},
        {"time": "12:00", "category": "식사", "sub_category": "한식", "poi_id": "R1", "actual_spent": 9000},
        {"time": "15:00", "category": "카페", "sub_category": "카페", "poi_id": "C1", "actual_spent": 5500},
        {"time": "16:00", "category": "교육", "sub_category": "학원", "poi_id": "E1", "actual_spent": 200000},
        {"time": "18:30", "category": "마트", "sub_category": "슈퍼마켓", "poi_id": "M1", "actual_spent": 42000},
        {"time": "19:30", "category": "식사", "sub_category": "한식", "poi_id": "R2", "actual_spent": 13000},
        {"time": "21:00", "category": "주점", "sub_category": "일반주점", "poi_id": "B1", "actual_spent": 30000},
    ]


def _day_home():
    return [{"time": "08:00", "category": "집", "sub_category": "휴식", "poi_id": "HOME"}]


def _apply(c, events, **extra):
    return c.apply_consumption_model(events, daily=120000, income_tier="중", tendency=None, balance=1500000,
                                     aid=AID, is_weekend=False, weekday=2, monthly_paid=set(),
                                     budget_carry_in=-5000, **extra)


def _shares(cb, **over):
    s = {k: 0.0 for k in cb.BUDGET_L1}
    s.update(over)
    return s


def _ev(t, cat, trigger="lifestyle", anchor="residence", sub=None):
    from stage1_intent import Stage1Event
    return Stage1Event(time=t, anchor=anchor, category=cat, sub_category=sub, intent="x", trigger=trigger)


def _mk(**kw):
    from stage1_intent import Stage1Event
    return Stage1Event(**kw)


def _mins(t):
    return int(t[:2]) * 60 + int(t[3:5])


# ---------------------------------------------------------------- 꺼져 있을 때

def test_off_matches_starting_engine(mods):
    cb, c = mods(False)
    assert cb.ENABLED is False
    want = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    for name, mk in (("weekday_mixed", _day_mixed), ("no_commerce", _day_home)):
        ev = mk()
        meta = _apply(c, ev)
        got = json.loads(json.dumps({"meta": meta, "events": ev}, ensure_ascii=False, sort_keys=True))
        assert got == want[name], name
        # 업종 인자를 넘겨도 꺼져 있으면 무시한다
        ev2 = mk()
        meta2 = _apply(c, ev2, category_shares=_shares(cb, 식사=1.0), budget_carry_cat={"식사": 99999})
        assert json.loads(json.dumps({"meta": meta2, "events": ev2}, ensure_ascii=False, sort_keys=True)) == want[name]
        assert "category_budget" not in meta2


def test_on_without_shares_is_old_path(mods):
    cb, c = mods(True)
    assert cb.ENABLED is True
    want = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    ev = _day_mixed()
    meta = _apply(c, ev)       # 입력 업종 비율이 없는 사람(shares_for → None)
    assert json.loads(json.dumps({"meta": meta, "events": ev}, ensure_ascii=False, sort_keys=True)) == want["weekday_mixed"]


# ---------------------------------------------------------------- shares_for

def test_shares_sum_floor_and_daytype(mods):
    cb, _ = mods(True)
    wd_in = {"한식": 0.30, "커피전문점": 0.10, "할인점/슈퍼마켓": 0.20, "일반병원": 0.15, "의복/의류": 0.10,
             "학원": 0.05, "편의점": 0.10}
    we_in = {"한식": 0.50, "커피전문점": 0.20, "편의점": 0.30}
    p = {"cat_ratio_wd": json.dumps(wd_in, ensure_ascii=False), "cat_ratio_we": json.dumps(we_in, ensure_ascii=False)}
    wd, hints = cb.shares_for(p, False)
    we, _ = cb.shares_for(p, True)
    for s in (wd, we):
        assert set(s) == set(cb.BUDGET_L1)
        assert abs(sum(s.values()) - 1.0) < 1e-9
        assert min(s.values()) >= cb.MIN_SHARE - 1e-12
    outside = [k for k in cb.BUDGET_L1 if k not in {"식사", "카페", "마트", "건강", "쇼핑", "교육", "편의점"}]
    assert outside and all(abs(wd[k] - cb.MIN_SHARE) < 1e-12 for k in outside)
    assert wd["식사"] > wd["마트"] > wd["건강"] > wd["카페"]          # 입력 순서를 지킨다
    assert hints["건강"] == ["의원"] and hints["쇼핑"] == ["의류"]
    # 주말은 주말 입력을 쓴다 — 평일에 있던 마트가 주말에는 최소 몫이다
    assert we["식사"] > wd["식사"] and abs(we["마트"] - cb.MIN_SHARE) < 1e-12
    # 주말 입력이 없으면 평일 입력, 입력이 없거나 깨졌으면 None
    assert cb.shares_for({"cat_ratio_wd": p["cat_ratio_wd"]}, True)[0] == wd
    assert cb.shares_for({}, False) is None
    assert cb.shares_for({"cat_ratio_wd": "not json"}, False) is None


def test_carry_from_state(mods):
    cb, _ = mods(True)
    stored = cb.carry_from_state(json.dumps({"식사": -3000, "카페": 4000}), AID)
    assert stored["식사"] == -3000 and stored["카페"] == 4000
    fresh = cb.carry_from_state(None, AID)
    assert set(fresh) == set(cb.BUDGET_L1)
    assert all(0 <= fresh[k] <= cb.TICKETS[k] for k in cb.BUDGET_L1)
    assert fresh == cb.carry_from_state(None, AID)                   # 같은 사람은 같은 위상(두 갈래가 같은 게이지)
    assert fresh != cb.carry_from_state(None, AID.replace("001", "002"))
    assert stored["마트"] == fresh["마트"]                            # 저장에 없는 업종만 위상으로 채운다


# ---------------------------------------------------------------- plan_day

def _plan(cb, events, *, budget=0, shares=None, carry=None, is_weekend=True, has_work=False, day="2025-07-19"):
    return cb.plan_day(events, aid=AID, day=day, is_weekend=is_weekend, has_work=has_work, budget=budget,
                       shares=shares or _shares(cb), hints={}, carry=carry or {k: 0 for k in cb.BUDGET_L1},
                       make_event=_mk)


def test_protected_meals_never_converted(mods):
    cb, _ = mods(True)
    events = [_ev("08:00", "식사", trigger="appointment"), _ev("12:00", "식사", trigger="policy"),
              _ev("13:00", "식사", trigger="rumor"), _ev("19:00", "식사", trigger="lifestyle")]
    before = [e.model_dump() for e in events[:3]]
    out, meta = _plan(cb, events)                                   # 식사 봉투 0
    assert [e.model_dump() for e in out if e.trigger in cb.PROTECTED_TRIGGERS] == before
    routine = [e for e in out if e.trigger == "lifestyle" and e.time == "19:00"][0]
    assert routine.category == "집" and routine.anchor == "residence" and routine.sub_category == "식사"
    assert meta["meals_converted"] == 1


def test_only_unaffordable_routine_meals_go_home_or_work(mods):
    cb, _ = mods(True)
    tk = cb.TICKETS["식사"]
    events = [_ev("08:00", "식사"), _ev("12:30", "식사", anchor="workplace"), _ev("19:00", "식사")]
    out, meta = _plan(cb, events, budget=tk, shares=_shares(cb, 식사=1.0), is_weekend=False, has_work=True,
                      day="2025-07-16")
    by_time = {e.time: e for e in out}
    assert by_time["08:00"].category == "식사"                      # 감당할 수 있는 한 끼(이른 끼니)는 그대로
    assert by_time["12:30"].category == "직장" and by_time["12:30"].anchor == "workplace"   # 근무 중
    assert by_time["19:00"].category == "집" and by_time["19:00"].anchor == "residence"
    assert all(e.pinned_poi is None and "[업종 예산" in (e.reasoning or "") for e in (by_time["12:30"], by_time["19:00"]))
    assert meta["meals_converted"] == 2
    # 봉투+이월로 다 감당하면 아무것도 바꾸지 않는다
    events2 = [_ev("08:00", "식사"), _ev("12:30", "식사"), _ev("19:00", "식사")]
    out2, meta2 = _plan(cb, events2, budget=3 * tk, shares=_shares(cb, 식사=1.0))
    assert meta2["meals_converted"] == 0 and all(e.category == "식사" for e in out2)


def _busy_day():
    return [_ev("07:00", "집", sub="기상"), _ev("08:00", "식사"), _ev("12:00", "식사"), _ev("13:00", "카페"),
            _ev("18:30", "식사"), _ev("23:00", "집", sub="취침")]


def test_plan_day_deterministic_sorted_and_bounded(mods):
    cb, _ = mods(True)
    rich = {k: 10 * cb.TICKETS[k] for k in cb.BUDGET_L1}
    a, ma = _plan(cb, _busy_day(), budget=50000, shares=_shares(cb, 식사=0.5, 카페=0.5), carry=dict(rich))
    b, mb = _plan(cb, _busy_day(), budget=50000, shares=_shares(cb, 식사=0.5, 카페=0.5), carry=dict(rich))
    assert [e.model_dump() for e in a] == [e.model_dump() for e in b] and ma == mb
    assert [_mins(e.time) for e in a] == sorted(_mins(e.time) for e in a)
    assert len(ma["injected"]) == cb.MAX_INJECT
    injected = {x["category"] for x in ma["injected"]}
    assert not injected & cb.NO_INJECT and "카페" not in injected   # 이미 일정에 있는 업종·넣지 않는 업종은 넣지 않는다
    for x in ma["injected"]:
        ev = [e for e in a if e.time == x["time"] and e.category == x["category"]][0]
        assert ev.trigger == "lifestyle" and ev.anchor == "residence" and "필요 게이지" in ev.reasoning
    # 게이지가 1회 구매액에 못 미치면 넣지 않는다
    _, mc = _plan(cb, _busy_day(), budget=0, carry={k: 0 for k in cb.BUDGET_L1})
    assert mc["injected"] == []


def test_injected_times_avoid_collisions_and_work_hours(mods):
    cb, _ = mods(True)
    rich = {k: 10 * cb.TICKETS[k] for k in cb.BUDGET_L1}
    for day in ("2025-07-14", "2025-07-15", "2025-07-16", "2025-07-17", "2025-07-18"):
        for aid_n in range(1, 21):
            aid = AID.replace("001", "%03d" % aid_n)
            events = _busy_day()
            original = [_mins(e.time) for e in events]
            out, meta = cb.plan_day(events, aid=aid, day=day, is_weekend=False, has_work=True, budget=0,
                                    shares=_shares(cb), hints={}, carry=dict(rich), make_event=_mk)
            times = [_mins(x["time"]) for x in meta["injected"]]
            assert len(times) <= cb.MAX_INJECT
            for i, t in enumerate(times):
                assert not (cb.WORK_HOURS[0] <= t < cb.WORK_HOURS[1])          # 근무자 평일 근무 시간 회피
                assert all(abs(t - o) >= 30 for o in original)                # 기존 일정과 30분 이상
                assert all(abs(t - u) >= 30 for u in times[:i])                # 넣은 일정끼리도


def test_monthly_fee_not_injected_once_paid_this_month(mods):
    cb, c = mods(True)
    carry = {k: 0 for k in cb.BUDGET_L1}
    carry["교육"] = 10 * cb.TICKETS["교육"]                               # 교육 게이지만 찼다
    kw = dict(aid=AID, day="2025-07-19", is_weekend=True, has_work=False, budget=0, shares=_shares(cb),
              carry=carry, make_event=_mk, monthly_subs=c.MONTHLY_SUBS)
    _, unpaid = cb.plan_day(_busy_day(), hints={}, monthly_paid=set(), **kw)
    assert [x["sub"] for x in unpaid["injected"]] == ["학원"]
    _, paid = cb.plan_day(_busy_day(), hints={}, monthly_paid={"학원"}, **kw)
    assert paid["injected"] == []                                         # 이미 낸 달에는 넣지 않는다
    # 건너뛴 업종은 넣는 건수에 세지 않는다 — 다음으로 밀린 업종이 들어간다
    carry["미용"] = 10 * cb.TICKETS["미용"]
    _, nxt = cb.plan_day(_busy_day(), hints={}, monthly_paid={"학원"}, **kw)
    assert [x["category"] for x in nxt["injected"]] == ["미용"]
    # 같은 업종이라도 월 납부가 아닌 세부 업종(입력 힌트)은 그대로 넣는다
    carry["미용"] = 0
    carry["건강"] = 10 * cb.TICKETS["건강"]
    carry["교육"] = 0
    _, gym = cb.plan_day(_busy_day(), hints={"건강": ["헬스장"]}, monthly_paid={"헬스장"}, **kw)
    _, clinic = cb.plan_day(_busy_day(), hints={"건강": ["의원"]}, monthly_paid={"헬스장"}, **kw)
    assert gym["injected"] == [] and [x["sub"] for x in clinic["injected"]] == ["의원"]


# ---------------------------------------------------------------- allocate

def _alloc(cb, commerce, amounts, *, budget, shares, carry=None, monthly_paid=None, aid=AID):
    import consumption
    return cb.allocate(commerce, amounts, budget=budget, shares=shares,
                       carry=carry if carry is not None else {k: 0 for k in cb.BUDGET_L1}, aid=aid,
                       monthly_subs=consumption.MONTHLY_SUBS, irregular_subs=consumption.IRREGULAR_SUBS,
                       monthly_paid=monthly_paid)


def test_allocate_category_spend_within_envelope_plus_carry(mods):
    cb, c = mods(True)
    cats = ["식사", "카페", "마트", "편의점", "주점", "건강", "교육"]
    subs = {"식사": ["한식"], "카페": ["카페"], "마트": ["슈퍼마켓"], "편의점": ["편의점"], "주점": ["일반주점"],
            "건강": ["의원", "약국", "헬스장"], "교육": ["학원", "독서실"]}
    rng = random.Random(20261019)
    for trial in range(300):
        n = rng.randint(1, 9)
        commerce, amounts = [], []
        for i in range(n):
            k = rng.choice(cats)
            commerce.append({"time": "%02d:%02d" % (8 + i, rng.choice((0, 30))), "category": k,
                             "sub_category": rng.choice(subs[k]), "poi_id": "P%d" % i,
                             "trigger": rng.choice(["lifestyle", "mood", "policy"] if rng.random() < 0.2 else ["lifestyle"])})
            amounts.append(float(rng.choice([3000, 5500, 9000, 13000, 25000, 42000, 150000])))
        shares = {k: rng.random() for k in cb.BUDGET_L1}
        tot = sum(shares.values())
        shares = {k: v / tot for k, v in shares.items()}
        carry = {k: rng.randint(-20000, 20000) for k in cb.BUDGET_L1}
        budget = rng.randint(0, 80000)
        out, meta = _alloc(cb, copy.deepcopy(commerce), amounts, budget=budget, shares=shares, carry=carry,
                           aid="A%d" % trial)
        env = cb.envelopes(budget, shares)
        for k in cb.BUDGET_L1:
            must = [i for i, e in enumerate(commerce) if cb.l1_of(e["category"]) == k and (
                e["sub_category"] in c.MONTHLY_SUBS or e["sub_category"] in c.IRREGULAR_SUBS
                or e["trigger"] == "policy")]
            room = env[k] + carry[k] - sum(out[i] for i in must)
            paid = [out[i] for i, e in enumerate(commerce)
                    if cb.l1_of(e["category"]) == k and i not in must and out[i] > 0]
            # 봉투+이월(필수 지출을 뺀 것) 안에서만 결제한다. 넘는 결제는 업종마다 마지막 한 건(해시 확률 규칙)뿐이다.
            assert sum(paid[:-1]) <= max(0, room), (trial, k)
            assert meta["category"]["paid_by_cat"][k] == sum(out[i] for i, e in enumerate(commerce)
                                                             if cb.l1_of(e["category"]) == k)


def test_allocate_monthly_once_policy_full_and_irregular_full(mods):
    cb, _ = mods(True)
    commerce = [{"time": "09:00", "category": "건강", "sub_category": "의원", "poi_id": "H"},
                {"time": "12:00", "category": "식사", "sub_category": "한식", "poi_id": "R", "trigger": "policy"},
                {"time": "16:00", "category": "교육", "sub_category": "학원", "poi_id": "E"},
                {"time": "20:00", "category": "교육", "sub_category": "학원", "poi_id": "E"},
                {"time": "21:00", "category": "식사", "sub_category": "한식", "poi_id": "R2"}]
    out, meta = _alloc(cb, commerce, [15000.0, 30000.0, 200000.0, 200000.0, 9000.0], budget=0, shares=_shares(cb, 식사=1.0),
                       monthly_paid=set())
    assert out[0] == 15000 and meta["irregular_total"] == 15000          # 비정기 의료는 봉투 밖에서 전액
    assert out[1] == 30000 and meta["policy_forced_total"] == 30000      # 정책 계기 일정은 봉투가 0 이어도 전액
    assert out[2] == 200000 and out[3] == 0 and commerce[3].get("monthly_already_paid")   # 월 납부는 달에 한 번
    assert out[4] == 0 and commerce[4].get("budget_skipped")             # 일상 끼니는 봉투를 넘으면 거른다
    out2, _ = _alloc(cb, [{"time": "16:00", "category": "교육", "sub_category": "학원", "poi_id": "E"}], [200000.0],
                     budget=0, shares=_shares(cb, 교육=1.0), monthly_paid={"학원"})
    assert out2 == [0]                                                     # 이번 달 이미 냈다
    # 필수 지출은 그 업종 봉투에서 빠진다(이월이 음수로 남는다)
    assert meta["category"]["carry_out"]["식사"] < 0 and meta["category"]["paid_by_cat"]["식사"] == 30000


def test_allocate_carry_bounds(mods):
    cb, _ = mods(True)
    shares = _shares(cb, 식사=0.5, 카페=0.5)
    _, meta = _alloc(cb, [], [], budget=10000, shares=shares, carry={k: 10 ** 9 for k in cb.BUDGET_L1})
    env = cb.envelopes(10000, shares)
    for k in cb.BUDGET_L1:
        assert meta["category"]["carry_out"][k] == max(int(1.5 * cb.TICKETS[k]), 3 * env[k])
    big = [{"time": "09:00", "category": "건강", "sub_category": "의원", "poi_id": "H"}]
    _, meta2 = _alloc(cb, big, [10 ** 9], budget=10000, shares=shares, carry={k: 0 for k in cb.BUDGET_L1})
    assert meta2["category"]["carry_out"]["건강"] == -max(cb.TICKETS["건강"], 3 * env["건강"])
    assert meta2["carry_out"] == sum(meta2["category"]["carry_out"].values())


# ---------------------------------------------------------------- 엔진 안에서

def test_on_meals_stay_inside_envelope(mods):
    cb, c = mods(True)
    shares = _shares(cb, 식사=0.4, 카페=0.2, 마트=0.2, 주점=0.2)
    carry = {k: 0 for k in cb.BUDGET_L1}
    carry["식사"] = -10 ** 7                                              # 끼니 봉투가 바닥이다
    carry["카페"] = 10000
    ev = _day_mixed()
    meta = _apply(c, ev, category_shares=shares, budget_carry_cat=carry)
    by_time = {e["time"]: e for e in ev}
    assert by_time["12:00"]["actual_spent"] == 0 and by_time["19:30"]["actual_spent"] == 0   # 옛 경로는 전액 냈다
    assert by_time["12:00"]["budget_skipped"] and by_time["19:30"]["purchase_status"] == "not_purchased"
    assert by_time["09:00"]["actual_spent"] == 15000 and by_time["16:00"]["actual_spent"] == 200000
    assert by_time["15:00"]["actual_spent"] == 5500                       # 카페는 이월로 감당
    cat = meta["category_budget"]
    assert cat["carry_in"] == carry and set(cat["carry_out"]) == set(cb.BUDGET_L1)
    assert meta["price_first"]["carry_out"] == sum(cat["carry_out"].values())
    assert meta["price_first"]["carry_in"] == sum(carry.values())
    assert meta["price_first"]["routine_budget"] == meta["personal_total"]   # 합계 이월(budget_carry_in)은 더하지 않는다


def test_on_no_commerce_day_keeps_gauge(mods):
    cb, c = mods(True)
    want = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    budget = c.base_store_budget(120000, AID)
    # 계획 배수 1 일 때의 가게 예산 — 같은 입력에서 엔진이 쓴 값과 같다
    assert budget == want["weekday_mixed"]["meta"]["personal_total"]
    shares = _shares(cb, 식사=0.5, 마트=0.5)
    carry = {k: 0 for k in cb.BUDGET_L1}
    carry["마트"] = -2000
    meta = _apply(c, _day_home(), category_shares=shares, budget_carry_cat=carry)
    env = cb.envelopes(budget, shares)
    assert meta["reason"] == "no_commerce"
    assert meta["category_budget"]["carry_out"]["식사"] == env["식사"]
    assert meta["category_budget"]["carry_out"]["마트"] == env["마트"] - 2000
    rest = {k: v for k, v in meta.items() if k != "category_budget"}
    assert rest == want["no_commerce"]["meta"]                            # 나머지는 꺼졌을 때와 같다


def test_state_wiring():
    import dawn_context
    import plan_writer
    assert "s.budget_carry_cat = $budget_carry_cat" in plan_writer.NIGHT_STATE_CYPHER
    assert "s.budget_carry_cat AS budget_carry_cat" in dawn_context.STATE_CYPHER
    import inspect
    assert "budget_carry_cat" in inspect.signature(plan_writer.night_create_state).parameters

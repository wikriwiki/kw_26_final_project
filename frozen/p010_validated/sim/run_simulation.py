"""시뮬 메인 루프 — Day-by-Day 순차, agent ThreadPoolExecutor 동시 처리.

각 Day t 처리 (노션 다이어그램 정합):
  Phase 1 (낮)  : Dawn 컨텍스트 → Stage 1 → Stage 2 → satisfaction → Plan write
  Phase 1/3 (밤): 어제 visited Memory + 오늘 State CREATE
  Phase 2 (밤) : 상호작용 대상 선정 + 의도 분류 LLM + Conversation 적재
                  → 약속(target_day=t+1)은 D+1 Dawn ④에서 자동 조회

산출물 (모두 SIM_OUTPUT_DIR 환경변수 기준, 기본=~/sim_output):
  $SIM_OUTPUT_DIR/checkpoints/done_<day>.json    # 처리 완료 agent ID
  $SIM_OUTPUT_DIR/checkpoints/failed_<day>.json  # 실패 케이스
  $SIM_OUTPUT_DIR/metrics/day_<day>.jsonl        # agent별 토큰·소요·만족도·정책hits

CLI:
  python run_simulation.py --start 2026-05-01 --days 3 --workers 32
  python run_simulation.py --start 2026-05-01 --days 1 --gu 11680 --limit 100  # 강남 100명 1일

권장 workers (A100 80GB + Qwen3-32B-AWQ 기준):
  · 16: 안전 baseline, GPU 사용률 ~70%
  · 32: sweet spot, throughput +30~50%, KV cache 안전
  · 48: 최대치, throughput +50~80%, mem-fraction 0.88 거의 가득
  · 64+: SGLang 의 KV pool 한계 도달 시 cache evict 발생 (역효과)
  실제 sweet spot 은 `curl :30000/metrics` 의 `num_running_reqs` 모니터링으로 확인.

환경변수:
  SIM_OUTPUT_DIR  : 출력 디렉토리 (기본 ~/sim_output)
  LLM_MODE        : qwen32b | qwen14b | qwen9b | exaone (기본 qwen32b)
  SGLANG_BASE_URL : LLM 서버 URL (기본 http://localhost:30000/v1, vLLM 8000 폴백)
  NEO4J_POOL_SIZE : Neo4j 드라이버 connection pool 크기 (기본 100, workers 의 2~3배 권장)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import Lock

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neo4j_load._common import driver_session  # noqa: E402
from dawn_context import build_dawn_context  # noqa: E402
from stage1_intent import call_stage1, grant_style_to_use  # noqa: E402
from stage2_poi import call_stage2, merge_to_final_events  # noqa: E402
from plan_writer import (  # noqa: E402
    write_plan, track_policy_usage,
    night_finalize_yesterday, night_create_state,
    apply_grant_to_prev_state, get_grant_amount,
    _grant_for_single_policy,
    aggregate_policy_spend, validate_policy_spend,
)
from timing_metrics import (  # noqa: E402
    load_jsonl,
    slow_cases,
    write_day_timing_report,
    write_json_atomic,
)
from consumption import (  # noqa: E402
    apply_consumption_model,
    filter_active_grant_balances,
    settle_policy_spend_priority,
)


# Google Drive 동기화 폴더(G:\)는 file write 충돌 위험 → 로컬 디스크 사용
# 기본: ~/sim_output (Windows: C:\Users\<user>\sim_output, Linux: $HOME/sim_output)
_LOCAL_BASE = Path(os.environ.get("SIM_OUTPUT_DIR",
                                  os.path.expanduser("~/sim_output")))
OUT_DIR = _LOCAL_BASE
CHECK_DIR = OUT_DIR / "checkpoints"
METRICS_DIR = OUT_DIR / "metrics"
CHECK_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# 정책 효과는 임의 modifier로 가산하지 않는다. dawn_context.POLICY_CYPHER가 매일
# 활성 정책을 자연어 description으로 Stage 1 프롬프트에 주입, LLM이 자율 해석.
# subsidy 정책의 cap_per_agent 잔액만 plan_writer.track_policy_usage에서 추적 →
# 다음날 Dawn에 "남은 잔액 N원" 형태로 LLM에 노출. 만족도 가산은 없음.
# POLICY_TARGET_CATS/POLICY_DISTRICT hardcoded 폐기 (2026-05-16).


# =========================================================
# Agent 풀 (시뮬 대상)
# =========================================================
def fetch_agents(limit: int | None = None, gu_only: str | None = None) -> list[str]:
    """시뮬 대상 agent.

    limit 지정 시 **소비 10분위 비례 층화표본**을 뽑는다. 단순 `ORDER BY a.id LIMIT n`은
    id 앞자리가 행정동 코드라 지역(종로·중구…)과 그에 딸린 소득 분포가 통째로 잘려나가
    모집단 분포를 재현하지 못한다. 분위별로 (limit × 분위비중)명씩 뽑아 전체 분포를 보존.
    seed 고정 → 같은 limit이면 항상 같은 표본(재현성).
    """
    where_gu = ""
    if gu_only:
        where_gu = "AND (a)-[:LIVES_AT]->(:POI)-[:IN_DONG]->(:Dong)<-[:HAS_DONG]-(:District {code:$gu}) "
    params = {"gu": gu_only} if gu_only else {}

    with driver_session() as s:
        if not limit:
            q = f"MATCH (a:Agent) WHERE (a)-[:LIVES_AT]->() {where_gu}RETURN a.id AS id ORDER BY a.id"
            return [r["id"] for r in s.run(q, **params)]

        # 분위별 모집단 (분위 미상은 '0' 버킷으로 묶어 함께 비례 배분)
        rows = list(s.run(
            f"MATCH (a:Agent) WHERE (a)-[:LIVES_AT]->() {where_gu}"
            "RETURN coalesce(a.spending_level_wd, 0) AS d, collect(a.id) AS ids", **params))

    import random as _rnd
    pop = {int(r["d"]): sorted(r["ids"]) for r in rows}
    total = sum(len(v) for v in pop.values())
    if total <= limit:
        return sorted(x for v in pop.values() for x in v)

    # 비례 배분 (내림) 후, 잔여分은 소수부 큰 분위부터 +1 → 합계 정확히 limit
    quota, frac = {}, {}
    for d, ids in pop.items():
        exact = len(ids) * limit / total
        quota[d] = int(exact)
        frac[d] = exact - quota[d]
    for d in sorted(frac, key=lambda k: frac[k], reverse=True)[:limit - sum(quota.values())]:
        quota[d] += 1

    out: list[str] = []
    for d, ids in pop.items():
        k = min(quota[d], len(ids))
        out.extend(_rnd.Random(f"agents-{d}-{limit}").sample(ids, k) if k else [])
    return sorted(out)


# =========================================================
# 한 agent의 1일 처리 (스레드 워커)
# =========================================================
def _local_daily_spend(daily: int | float | None, spend_decile: int | None = None) -> int:
    """'며칠치인가'를 셀 때 쓰는 하루 지출 — **사용처(소상공인)에서 나가는 몫**.

    지원금은 사용처에서만 쓸 수 있으므로, 며칠 버티는지는 전체 지출이 아니라 사용처 지출로
    나눠야 실제 감각과 맞는다. 사용처 비율은 서울시 상권분석 산출값(consumption.py)을 쓴다.
    """
    from consumption import ELIGIBLE_SHARE_SEOUL
    try:
        d = int(daily or 0)
    except (TypeError, ValueError):
        return 0
    if d <= 0:
        return 0
    return max(1, int(round(d * ELIGIBLE_SHARE_SEOUL)))


def _build_policy_budget_summary(policies: list[dict] | None, prev_policy_used: dict[str, int] | None,
                                  grants_applied_today: dict[str, int] | None = None,
                                  grant_remaining: dict[str, int] | None = None,
                                  daily_spend: int | float | None = None) -> str:
    """Stage2 LLM에 노출할 정책 예산 요약 텍스트.

    - subsidy/voucher: 잔여 cap = cap - used
    - grant: 오늘 받은 지원금 + 누적 잔여액 및 시스템의 지원금 우선 결제 원칙 명시
    """
    if not policies:
        return ""
    used = prev_policy_used or {}
    grants_today = grants_applied_today or {}
    remaining = grant_remaining or {}
    lines = []
    for pol in policies:
        pid = pol.get("id") or ""
        name = pol.get("name") or pid
        ptype = pol.get("type") or ""
        # grant — 오늘 받은 지원금 + 잔여
        if ptype == "grant":
            today_amt = grants_today.get(pid, 0)
            rem = remaining.get(pid, 0)
            parts = []
            if today_amt > 0:
                parts.append(f"오늘 +{today_amt:,}원")
            parts.append(f"누적 잔여 {rem:,}원")
            # 남은 지원금이 이 사람의 평소 하루 씀씀이로 며칠치인지. 본인의 지갑과 본인의
            # 소비규모로 계산한 값이며, 이 비(比)가 '아껴 쓸 돈인지 편히 쓸 돈인지'를 가른다.
            # LLM이 프롬프트 안에서 이 나눗셈을 스스로 하지 않아(측정: 지급액과 무관하게
            # 동일한 결제 선택) 계산해서 제시한다.
            try:
                _ds = int(daily_spend or 0)
                if _ds > 0 and rem > 0:
                    parts.append(f"평소 하루 씀씀이로 약 {max(1, round(rem / _ds))}일치")
            except (TypeError, ValueError):
                pass
            lines.append(
                f"{pid} {name} [정책 지갑] — {' / '.join(parts)}. "
                "소비 필요·시점·총액·POI는 평소 습관, 자산과 일정에 따라 판단한다. "
                "쓸 수 있는 매장에서 이 지갑으로 낼지 늘 쓰던 카드로 낼지는 결제 건마다 "
                "본인이 정하며, policy_spend에 적은 금액이 실제로 이 지갑에서 나간다."
            )
            continue
        # subsidy/voucher — 잔여 cap
        cap = pol.get("cap") or 0
        rate = pol.get("rate") or 0.0
        if cap > 0 and rate > 0:
            spent = used.get(pid, 0)
            rem_sub = max(0, cap - spent)
            lines.append(f"{pid}({name}) [subsidy {int(rate*100)}%]: 한도 {cap:,}원 / 사용 {spent:,}원 / 잔여 {rem_sub:,}원")
    return " | ".join(lines)


def _read_state_json(state_dict: dict | None, key: str) -> dict[str, int]:
    """State 노드의 JSON string 필드 파싱 헬퍼 — {ID: int} dict 반환.

    grant_received, grant_remaining 등 정책별 정수 dict 필드용.
    파싱 실패·dict 아님·문자열 빈 dict면 빈 dict 반환.
    """
    if not state_dict:
        return {}
    raw = state_dict.get(key) or "{}"
    if isinstance(raw, dict):
        return {str(k): int(v) for k, v in raw.items()}
    try:
        d = json.loads(raw) if isinstance(raw, str) else {}
        return {str(k): int(v) for k, v in d.items()} if isinstance(d, dict) else {}
    except Exception:
        return {}


# 하위호환 alias
def _read_grant_received(state_dict: dict | None) -> dict[str, int]:
    """DEPRECATED. _read_state_json(state, 'grant_received') 사용 권장."""
    return _read_state_json(state_dict, "grant_received")


def _merge_policy_lifecycle(raw_lifecycle, policies: list[dict] | None) -> dict[str, bool]:
    """기존 policy_lifecycle JSON에 오늘 Dawn 정책 ID를 true로 병합."""
    lifecycle: dict[str, bool] = {}
    if raw_lifecycle:
        try:
            parsed = json.loads(raw_lifecycle) if isinstance(raw_lifecycle, str) else raw_lifecycle
            if isinstance(parsed, dict):
                lifecycle = {str(k): bool(v) for k, v in parsed.items()}
        except Exception:
            lifecycle = {}
    for pol in policies or []:
        pid = pol.get("id")
        if pid:
            lifecycle[str(pid)] = True
    return lifecycle


def process_one(aid: str, today: date, day_idx: int) -> dict:
    """1 agent 1일. 결과 메타 dict 반환 (실패 시 status='error').

    단계별 timing 측정:
      t_dawn:   build_dawn_context (Cypher 7개 read)
      t_s1:     Stage1 LLM call
      t_s2:     Stage2 LLM call(들) 합산 — review_lookup 재호출 포함
      t_write:  Plan/State Neo4j write
      t_total:  전체 (start → return)
    """
    t0 = time.time()
    timing: dict[str, float] = {}
    try:
        _t = time.time()
        ctx = build_dawn_context(aid, today)
        timing["t_dawn"] = round(time.time() - _t, 3)
        if not ctx.persona:
            return {"aid": aid, "status": "no_persona", "elapsed": time.time() - t0}

        # grant 정책 — effective_from 당일 지원금 수령.
        # ★ 정책지원금 = balance·daily_wd와 분리된 독립 지갑. grant_remaining에만 적립하고
        #   balance에는 더하지 않는다 (개인 돈과 정책 돈 완전 분리, 미사용분 누수 방지).
        # 멱등성: 어제 State.grant_received에 이미 기록된 정책은 skip (resume 시 중복 적용 방지)
        prev_grant_received = _read_state_json(ctx.state, "grant_received")
        income = ctx.persona.get("income") or ctx.persona.get("p_income_level") or ""
        spend_decile = ctx.persona.get("spend_decile")   # 소비 10분위 — grant_key='spend_decile' 정책용
        grants_applied_today: dict[str, int] = {}
        for pol in (ctx.policy or []):
            if pol.get("type") != "grant":
                continue
            pid = pol.get("id") or ""
            if pid in prev_grant_received:
                # 이미 적용된 grant — skip (resume 멱등 가드)
                continue
            eff = pol.get("effective_from", "")
            if str(today) != eff:
                continue
            # 이 정책에서 받을 금액 — grant_remaining 독립 지갑에 적립 (balance 불변)
            amt = _grant_for_single_policy(income, pol, spend_decile=spend_decile)
            if amt > 0:
                grants_applied_today[pid] = amt

        # grant는 balance에 더하지 않는다 — grant_remaining 독립 지갑으로만 관리(회계 분리).
        # windfall 인지는 정책 카드(_format_policy)의 grant_days_since 감쇠가 담당.
        # Stage2 LLM에 노출할 정책 예산 요약 (정책 쿠폰 잔액·오늘 받은 지원금 명시)
        prev_used_for_budget = ctx.get_policy_used()
        # 누적 grant_received 갱신 (이번에 적용된 것 합산)
        merged_grant_received = dict(prev_grant_received)
        for pid, amt in grants_applied_today.items():
            merged_grant_received[pid] = merged_grant_received.get(pid, 0) + amt
        # 어제 grant_remaining 중 오늘도 활성인 정책만 가용 지갑으로 승계한다.
        # Dawn 목록에서 사라진 만료·비활성 잔액은 무제한 지갑으로 추정하지 않고 소멸시킨다.
        prev_grant_remaining_raw = _read_state_json(ctx.state, "grant_remaining")
        prev_grant_remaining, inactive_grant_remaining = filter_active_grant_balances(
            prev_grant_remaining_raw,
            ctx.policy,
        )
        # Stage2 LLM 노출용 잔여 가용액 (어제까지 잔여 + 오늘 받음)
        grant_avail_today: dict[str, int] = dict(prev_grant_remaining)
        for pid, amt in grants_applied_today.items():
            grant_avail_today[pid] = grant_avail_today.get(pid, 0) + int(amt)

        # Stage1이 보는 개인 정책 상태를 '어제 State'가 아니라 오늘 지급까지 반영한
        # 현재 상태로 맞춘다. DB에는 Night 단계에서 기록하므로 여기서는 컨텍스트만 갱신한다.
        # 이 갱신이 없으면 지급 당일에도 Stage1 프롬프트가 "지급 전 / 잔액 0원"으로 보인다.
        if ctx.state is None:
            ctx.state = {}
        ctx.state["grant_received"] = merged_grant_received
        ctx.state["grant_remaining"] = grant_avail_today

        ctx.persona["policy_budget_summary"] = _build_policy_budget_summary(
            ctx.policy, prev_used_for_budget, grants_applied_today, grant_avail_today,
            # '며칠치'는 전체 지출이 아니라 **동네 가게에서 나가는 지출** 기준으로 센다.
            # 지원금은 그 자리에서만 쓸 수 있으므로, 그 금액으로 며칠이 버티는지가 실제 감각이다.
            # 동네 가게 지출 = 평소 지출 × (1 − BDC 실측 대형·제외업종 비중).
            daily_spend=_local_daily_spend(
                ctx.persona.get("daily_we") if today.weekday() >= 5
                else ctx.persona.get("daily_wd"),
                ctx.persona.get("spend_decile"),
            ),
        )

        # 사용처 제한 정책(민생회복 소비쿠폰류, poi_restricted=true) 감지
        # → 쿠폰 잔액이 있으면 Stage2에 [쿠폰] 사실 표시 + 정책사용 하드검증.
        # 후보 정렬 가점은 POLICY_POI_SORT_BOOST=1인 별도 민감도 실험에서만 활성화한다.
        restricted_pids = {
            p["id"] for p in (ctx.policy or [])
            if p.get("poi_restricted") and grant_avail_today.get(p["id"], 0) > 0
        }
        ctx.persona["coupon_poi_restricted"] = bool(restricted_pids)
        if restricted_pids and ctx.persona.get("policy_budget_summary"):
            ctx.persona["policy_budget_summary"] += " (사용처 제한: [쿠폰] 표시 매장에서만 사용 가능)"

        # 지원금 수령 경과일 — windfall 감쇠 렌더용 (지급 당일만이 아니라
        # 이후에도 '며칠 전 받은 여윳돈'으로 계속, 그러나 점점 희미하게 인지)
        if ctx.state is not None:
            _gdays: dict[str, int] = {}
            for _p in (ctx.policy or []):
                if _p.get("type") == "grant" and _p.get("from_"):
                    try:
                        _d = (today - date.fromisoformat(str(_p["from_"])[:10])).days
                        if _d >= 0:
                            _gdays[_p["id"]] = _d
                    except (ValueError, TypeError):
                        pass
            if _gdays:
                ctx.state["grant_days_since"] = _gdays

        _t = time.time()
        s1, m1 = call_stage1(aid, today, ctx=ctx)
        timing["t_s1"] = round(time.time() - _t, 3)

        # state 전달 — 잔액(가용 자산)이 가격대(₩~₩₩₩) 선택의 예산 근거로 프롬프트에 노출.
        # active_policies/grant_remaining은 호출부 호환 인자(정책 정보는 persona로 전달됨).
        _t = time.time()
        s2, _cands, m2 = call_stage2(
            aid, s1, ctx.persona, today, state=ctx.state,
            active_policies=ctx.policy,
            grant_remaining=grant_avail_today,
        )
        timing["t_s2"] = round(time.time() - _t, 3)

        events = merge_to_final_events(
            s1, s2, ctx.persona,
            price_by_poi=m2.get("price_by_poi"),
            coupon_by_poi=m2.get("coupon_by_poi"),
            review_lookup_used=m2.get("review_lookup_used"),
            pre_review_picks=m2.get("pre_review_picks"),
        )

        # ── 소비성향(propensity) 모델 — Problem B (EconAgent 방식) ──
        # Stage2 절대 계획금액·POI 가격대를 보존하고, Stage1의 평소 대비 오늘 소비의향을
        # 곱한다. 지원금 잔액은 총액에 더하지 않으며, 계획된 사용 가능 거래액 범위의
        # 정책결제만 유동성을 완화한다.
        # 총소비 모델은 legacy로 복귀할 수 있지만 정책지갑 우선 정산은 공통 회계 규칙이다.
        cm_meta = {"applied": False}
        # 정책 유형 일반화: 어떤 grant든 속성(poi_restricted·target_l1s)이 제약이 된다.
        _pol_by_id = {p["id"]: p for p in (ctx.policy or [])}
        _envelopes = []
        _unrestricted_wallets: dict[str, int] = {}
        for pid, amt in grant_avail_today.items():
            pol = _pol_by_id[pid]  # 활성 grant만 필터링되어 반드시 정책 정의가 존재한다.
            scoped = bool(pol.get("poi_restricted")) or bool(pol.get("target_l1s"))
            if scoped and int(amt) > 0:
                _envelopes.append({
                    "pid": pid, "amount": int(amt),
                    "require_poi_eligible": bool(pol.get("poi_restricted")),
                    "categories": (pol.get("target_l1s") or None),
                })
            elif int(amt) > 0:
                _unrestricted_wallets[pid] = int(amt)

        if os.environ.get("CONSUMPTION_MODEL", "propensity") != "legacy":
            _is_weekend = today.weekday() >= 5
            # 제한 grant의 봉투는 사용처·업종 제약 메타데이터다.
            # 잔액을 p와 곱해 소비액으로 만들지 않으며, 소비 총액 확정 후 사용 가능한
            # 거래에서 결제수단만 정책지갑 우선으로 정산한다.
            cm_meta = apply_consumption_model(
                events,
                daily=ctx.persona.get("daily_we") if _is_weekend else ctx.persona.get("daily_wd"),
                income_tier=income,
                tendency=ctx.persona.get("tendency"),
                balance=(ctx.state or {}).get("balance"),
                grant_avail=_unrestricted_wallets,
                llm_propensity=getattr(s1, "daily_propensity", None),
                restricted_envelopes=_envelopes,
                # grant_use(강도 배급)는 폐기. 측정 결과 고소비 tier가 폭주해
                # 전체 hazard 37%/day·역진 −54%p로 붕괴했다(R38). 배급은 spread 경로로 일원화.
                # 결제 선택 모드에서 grant_use는 옛 강도 배급이 아니라 하루 태세다
                # (consumption.py가 건별 선택을 이 태세에 맞춰 비례 조정한다).
                # 낱말 태세(grant_style)가 있으면 그것을 비율로 옮겨 쓴다.
                grant_use=(
                    grant_style_to_use(getattr(s1, "grant_style", None))
                    or getattr(s1, "grant_use", None)
                ),
                grant_spread_days=getattr(s1, "grant_spread_days", None),
                grant_extra_spend=getattr(s1, "grant_extra_spend", None),
                grant_kept_share=getattr(s1, "grant_kept_share", None),
                grant_carry=(ctx.state or {}).get("grant_carry"),
                grant_plan_days=(ctx.state or {}).get("grant_plan_days"),
                online_share=getattr(s1, "online_share", None),
                # BDC 실측 소비수준 → 대형·제외업종 지출 비중(우리 소비패턴의 사실).
                spending_level=ctx.persona.get("spend_decile"),
            )
        else:
            # legacy는 총소비액을 건드리지 않되 결제수단은 동일한 우선 정산을 적용한다.
            _settlement = settle_policy_spend_priority(
                events,
                grant_avail=_unrestricted_wallets,
                restricted_envelopes=_envelopes,
                grant_use=getattr(s1, "grant_use", None),
            )
            _eligible_spend = int(_settlement["eligible_spend_total"])
            _allocated = int(_settlement["total"])
            cm_meta.update({
                "selected_policy_liquidity": _allocated,
                "selected_policy_liquidity_by_pid": _settlement["by_pid"],
                "policy_spend_allocated": _settlement["by_pid"],
                "policy_spend_allocated_total": _allocated,
                "policy_eligible_spend_total": _eligible_spend,
                "policy_eligible_event_count": _settlement["eligible_event_count"],
                "policy_payment_coverage": (
                    round(_allocated / _eligible_spend, 4) if _eligible_spend > 0 else 0.0
                ),
                "policy_liquidity_relief": 0,
                "policy_wallet_available": _settlement["wallet_available"],
                "policy_spend_requested": _settlement["requested_by_pid"],
                "mechanical_policy_uplift": 0,
            })

        # LLM policy_spend 환각 검증 — 사용처 제한 + 거래 단위(sum>actual) + 정책 단위(잔여액 초과)
        policy_spend_corrected = validate_policy_spend(
            events, policy_remaining=grant_avail_today, restricted_pids=restricted_pids,
        )

        # 오늘 거래별 policy_spend 집계 → 정책별 오늘 사용액
        today_policy_spend = aggregate_policy_spend(events)

        # 정책 cap 잔액 추적 (subsidy/voucher만 — grant는 policy_spend로 따로 추적)
        prev_policy_used = ctx.get_policy_used()
        updated_policy_used = track_policy_usage(
            events, ctx.persona,
            active_policies=ctx.policy,
            policy_used=prev_policy_used,
        )

        # grant_remaining = 어제 잔여 + 오늘 받음 − 오늘 사용 (음수 방지)
        merged_grant_remaining: dict[str, int] = dict(grant_avail_today)
        for pid, amt in today_policy_spend.items():
            merged_grant_remaining[pid] = max(0, merged_grant_remaining.get(pid, 0) - int(amt))

        # 활성 정책 카테고리 셋 (오늘 Dawn 컨텍스트에서 추출) — 사후 측정용 라벨
        active_policy_cats: set[str] = set()
        for pol in ctx.policy:
            for l1 in (pol.get("target_l1s") or []):
                if l1:
                    active_policy_cats.add(l1)

        day_type = "weekend" if today.weekday() >= 5 else "weekday"
        tokens_in = m1["tokens_in"] + (m2.get("tokens_in") or 0)
        tokens_out = m1["tokens_out"] + (m2.get("tokens_out") or 0)
        _t = time.time()
        _, n_inc = write_plan(
            aid, today, events, day_type, tokens_in, tokens_out,
            reviews_seen=m2.get("review_lookup_used"),
            review_lookup_count=m2.get("review_lookup_count", 0),
        )
        timing["t_write_plan"] = round(time.time() - _t, 3)

        # Night Phase — Day 1 새벽엔 어제(Day 0) Plan 없으므로 finalize는 Day 2 이상에서만
        n_mem = 0
        if day_idx >= 1:
            _t = time.time()
            n_mem = night_finalize_yesterday(aid, today)
            timing["t_night_finalize"] = round(time.time() - _t, 3)
        # 정책 인지 상태 — 어제 lifecycle에 오늘 Dawn 정책 ID를 true로 병합
        merged_policy_lifecycle = _merge_policy_lifecycle(
            (ctx.state or {}).get("policy_lc"),
            ctx.policy,
        )
        state = night_create_state(
            aid, today,
            policy_used=updated_policy_used,
            policy_lifecycle=merged_policy_lifecycle,
            grant_received=merged_grant_received,
            grant_remaining=merged_grant_remaining,
            today_policy_spent=sum(today_policy_spend.values()),
            grant_carry=int((cm_meta or {}).get("grant_carry_out") or 0),
            grant_plan_days=int((cm_meta or {}).get("grant_plan_days_effective") or 0),
            # 배송 주문은 INCLUDES 엣지가 없어 today_spent 합계에 잡히지 않는다. 별도로 차감한다.
            today_online_spent=int((cm_meta or {}).get("online_total") or 0),
        )

        # 만족도 평균
        sats = [e["actual_satisfaction"] for e in events if e["actual_satisfaction"] is not None]
        avg_sat = sum(sats) / len(sats) if sats else None

        # 정책 적용 이벤트 카운트 (사후 분석용 라벨 — modifier 아님)
        # _policy_match로 자치구·카테고리 둘 다 매칭된 commerce 이벤트만 카운트
        from plan_writer import _policy_match
        home5 = (ctx.persona.get("home_dong_code") or "")[:5]
        work5 = (ctx.persona.get("work_dong_code") or "")[:5]
        policy_hits = sum(
            1 for e in events
            if e.get("poi_id") and any(_policy_match(e, p, home5, work5) for p in (ctx.policy or []))
        )

        return {
            "aid": aid, "status": "ok",
            "elapsed": round(time.time() - t0, 2),
            # 단계별 timing (병목 분석용)
            **{f"timing_{k}": v for k, v in timing.items()},
            "n_events": len(events), "n_includes": n_inc,
            "n_visited_memories": n_mem,
            "avg_sat": round(avg_sat, 3) if avg_sat is not None else None,
            "balance": state.get("balance"),
            "mood": round(state.get("mood", 0), 3) if state else None,
            "fatigue": round(state.get("fatigue", 0), 3) if state else None,
            "tokens_in": tokens_in, "tokens_out": tokens_out,
            "policy_hits": policy_hits,
            # 정책 사용 트래킹 (옵션 A)
            "grant_applied_today": sum(grants_applied_today.values()),
            "grant_expired_today": sum(inactive_grant_remaining.values()),
            "policy_spend_today": sum(today_policy_spend.values()),
            "grant_remaining_total": sum(merged_grant_remaining.values()),
            "policy_spend_corrected": policy_spend_corrected,
            "cm_propensity": cm_meta.get("propensity"),
            "s1_daily_propensity": getattr(s1, "daily_propensity", None),
            "s1_grant_use": getattr(s1, "grant_use", None),
            "s1_grant_style": getattr(s1, "grant_style", None),
            "s1_grant_spread_days": getattr(s1, "grant_spread_days", None),
            "s1_grant_plan_reason": getattr(s1, "grant_plan_reason", None),
            "s1_grant_extra_spend": getattr(s1, "grant_extra_spend", None),
            "s1_grant_kept_share": getattr(s1, "grant_kept_share", None),
            "cm_substituted": cm_meta.get("substituted"),
            "cm_intended_grant_today": cm_meta.get("intended_grant_today"),
            "cm_grant_carry_in": cm_meta.get("grant_carry_in"),
            "cm_grant_carry_out": cm_meta.get("grant_carry_out"),
            "cm_grant_plan_days": cm_meta.get("grant_plan_days_effective"),
            "cm_eligible_base": cm_meta.get("eligible_base"),
            "cm_additional_from_grant": cm_meta.get("additional_from_grant"),
            "cm_personal_total": cm_meta.get("personal_total"),
            "cm_anchor_total": cm_meta.get("anchor_total"),
            "cm_plan_over_anchor": cm_meta.get("plan_over_anchor"),
            "cm_propensity_center": cm_meta.get("propensity_center"),
            "cm_day_multiplier": cm_meta.get("day_multiplier"),
            "cm_planned_total": cm_meta.get("planned_total"),
            "cm_today_total": cm_meta.get("today_total"),
            "cm_grant_choice_mode": cm_meta.get("grant_choice_mode"),
            "cm_grant_choice_share_mean": cm_meta.get("grant_choice_share_mean"),
            "cm_grant_posture": cm_meta.get("grant_posture"),
            "cm_mpc_new_share": cm_meta.get("mpc_new_share"),
            "spend_decile": ctx.persona.get("spend_decile"),
            "cm_mpc_new_share_effective": cm_meta.get("mpc_new_share_effective"),
            "cm_grant_extra_rate": cm_meta.get("grant_extra_rate"),
            "s1_grant_use": getattr(s1, "grant_use", None),
            "s1_grant_style": getattr(s1, "grant_style", None),
            "cm_online_share_source": cm_meta.get("online_share_source"),
            "cm_online_total": cm_meta.get("online_total"),
            "cm_online_share": cm_meta.get("online_share_effective"),
            "cm_today_total_incl_online": cm_meta.get("today_total_incl_online"),
            "s1_online_share": getattr(s1, "online_share", None),
            "cm_selected_policy_liquidity": cm_meta.get("selected_policy_liquidity", 0),
            "cm_policy_requested_total": sum(
                (cm_meta.get("policy_spend_requested") or {}).values()
            ),
            "cm_policy_allocated_total": cm_meta.get("policy_spend_allocated_total", 0),
            "cm_policy_eligible_spend_total": cm_meta.get("policy_eligible_spend_total", 0),
            "cm_policy_eligible_event_count": cm_meta.get("policy_eligible_event_count", 0),
            "cm_policy_payment_coverage": cm_meta.get("policy_payment_coverage", 0),
            "cm_policy_liquidity_relief": cm_meta.get("policy_liquidity_relief", 0),
            "cm_mechanical_policy_uplift": cm_meta.get("mechanical_policy_uplift", 0),
            "s1_attempts": m1["attempt"] + 1,
            "s1_timing": m1.get("s1_timing"),
            "prompt_timing": m1.get("prompt_timing"),
            "dawn_timing": dict(ctx.dawn_timing),
            "s2_attempts": (m2.get("attempt", 0) or 0) + 1 if not m2.get("skipped") else 0,
            "s2_timing": m2.get("s2_timing"),
            # Stage 2 fallback 카운트 (사후 분석용)
            "review_lookup_count": m2.get("review_lookup_count", 0),
            "fb_resolve_dong": m2.get("resolve_dong_placeholder_fallback", 0),
            "fb_cand_sub_match": m2.get("cand_sub_match", 0),
            "fb_cand_l1_dong": m2.get("cand_fallback_l1_dong", 0),
            "fb_cand_l1_district": m2.get("cand_fallback_l1_district", 0),
            "fb_cand_all_empty": m2.get("cand_all_empty", 0),
            "fb_hallucinations_corrected": m2.get("hallucinations_corrected", 0),
            "fb_hallucinations_dropped": m2.get("hallucinations_dropped", 0),
            "fb_order_mismatch": m2.get("order_mismatch", 0),
            "fb_missing_picks_filled": m2.get("missing_picks_filled", 0),
            # 같은 (dong, sub_cat) 이벤트 후보 풀 분할 (같은 날 반복 방문 차단)
            "fb_pool_split_groups": m2.get("pool_split_groups", 0),
            "fb_pool_split_events": m2.get("pool_split_events", 0),
        }
    except Exception as e:
        return {
            "aid": aid, "status": "error",
            "elapsed": round(time.time() - t0, 2),
            "error": str(e)[:200],
            "trace": traceback.format_exc(limit=3)[-500:],
        }


# =========================================================
# 일별 병목 리포트·백업
# =========================================================
def _write_timing_diagnostics(day_str: str, metrics_path: Path) -> dict:
    timing_dir = OUT_DIR / "timing"
    timing_path = timing_dir / f"day_{day_str}.json"
    report = write_day_timing_report(metrics_path, timing_path)

    rows = load_jsonl(metrics_path)
    slow = slow_cases(
        rows,
        dawn_sec=float(os.environ.get("SLOW_DAWN_SEC", "2")),
        stage1_sec=float(os.environ.get("SLOW_STAGE1_SEC", "60")),
        stage2_sec=float(os.environ.get("SLOW_STAGE2_SEC", "60")),
    )
    write_json_atomic(timing_dir / f"slow_{day_str}.json", slow)

    top = report.get("bottleneck_rank") or []
    if top:
        print("  [병목] 누적시간 상위:")
        for item in top[:8]:
            print(
                f"    {item['path']}: total={item['total_sec']:.1f}s "
                f"avg={item['avg_sec']:.3f}s p95={item['p95_sec']:.3f}s"
            )
    cache = report.get("cache") or {}
    print(
        f"  [캐시] persona={100 * cache.get('persona_hit_rate', 0):.1f}% "
        f"policy={100 * cache.get('policy_hit_rate', 0):.1f}% | "
        f"slow={len(slow)}명 → {timing_path.parent}"
    )
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _daily_backup(day_str: str, day_summary: dict, agent_ids: list[str]) -> None:
    """BACKUP_DIR 지정 시 그날 표본 State와 산출물을 원자적으로 백업한다.

    에이전트별 추가 작업은 없고 하루 종료 후 Neo4j 쿼리 1회만 수행한다.
    """
    raw_dir = os.environ.get("BACKUP_DIR")
    if not raw_dir:
        return
    backup_dir = Path(raw_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, dict] = {}

    try:
        state_path = backup_dir / f"state_{day_str}.jsonl"
        tmp_state = state_path.with_name(state_path.name + f".tmp.{os.getpid()}")
        n_state = 0
        with driver_session() as session, tmp_state.open("w", encoding="utf-8") as fp:
            rows = session.run(
                """
                UNWIND $aids AS aid
                MATCH (a:Agent {id: aid})-[:HAS_STATE {day: date($day)}]->(st:State)
                RETURN aid, st.balance AS balance, st.month_spent AS month_spent,
                       st.grant_received AS grant_received,
                       st.grant_remaining AS grant_remaining,
                       st.policy_used AS policy_used,
                       st.mood AS mood, st.fatigue AS fatigue
                ORDER BY aid
                """,
                aids=agent_ids,
                day=day_str,
            )
            for row in rows:
                fp.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
                n_state += 1
        tmp_state.replace(state_path)
        copied[state_path.name] = {
            "bytes": state_path.stat().st_size,
            "sha256": _sha256(state_path),
        }

        candidates = [
            METRICS_DIR / f"day_{day_str}.jsonl",
            CHECK_DIR / f"done_{day_str}.json",
            CHECK_DIR / f"failed_{day_str}.json",
            OUT_DIR / "timing" / f"day_{day_str}.json",
            OUT_DIR / "timing" / f"slow_{day_str}.json",
            OUT_DIR / "summary.json",
        ]
        for src in candidates:
            if not src.exists():
                continue
            relative = src.relative_to(OUT_DIR)
            dst = backup_dir / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp_dst = dst.with_name(dst.name + f".tmp.{os.getpid()}")
            shutil.copy2(src, tmp_dst)
            tmp_dst.replace(dst)
            copied[str(relative)] = {
                "bytes": dst.stat().st_size,
                "sha256": _sha256(dst),
            }

        manifest_path = backup_dir / "backup_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            manifest = {"days": {}}
        manifest.setdefault("days", {})[day_str] = {
            "backed_up_at": datetime.now().isoformat(timespec="seconds"),
            "state_rows": n_state,
            "expected_agents": len(agent_ids),
            "day_summary": day_summary,
            "files": copied,
        }
        write_json_atomic(manifest_path, manifest)
        print(f"  [backup] {day_str}: State {n_state}/{len(agent_ids)}행, 파일 {len(copied)}개 → {backup_dir}")
    except Exception as exc:
        print(f"  [backup] {day_str} 실패: {exc}")


# =========================================================
# Day 루프
# =========================================================
def run_day(agents: list[str], today: date, day_idx: int, workers: int = 64) -> dict:
    day_str = today.isoformat()
    done_path = CHECK_DIR / f"done_{day_str}.json"
    failed_path = CHECK_DIR / f"failed_{day_str}.json"
    metrics_path = METRICS_DIR / f"day_{day_str}.jsonl"

    done_aids: set[str] = set()
    if done_path.exists():
        done_aids = set(json.loads(done_path.read_text(encoding="utf-8")))
        print(f"[resume] {len(done_aids)} agents already done for {day_str}, skipping")

        # ★ resume 시 jsonl dedup — status=ok 줄을 aid당 1개만 남김 (이전 error 줄과 중복 제거)
        if metrics_path.exists() and done_aids:
            seen_ok: dict[str, str] = {}
            other_lines: list[str] = []
            with metrics_path.open(encoding="utf-8") as fp_r:
                for line in fp_r:
                    try:
                        j = json.loads(line)
                        if j.get("status") == "ok":
                            seen_ok[j["aid"]] = line   # 마지막 ok가 이김
                        else:
                            other_lines.append(line)
                    except json.JSONDecodeError:
                        continue
            # ok 줄만 dedup해서 다시 씀 (error 줄은 폐기 — 어차피 retry로 채움)
            with metrics_path.open("w", encoding="utf-8") as fp_w:
                for aid in done_aids:
                    if aid in seen_ok:
                        fp_w.write(seen_ok[aid])
            print(f"[resume] jsonl dedup: kept {len(seen_ok)} ok rows, dropped {len(other_lines)} error rows")

    remaining = [a for a in agents if a not in done_aids]
    print(f"[Day {day_idx} {day_str}] processing {len(remaining)} agents with {workers} workers")

    # 메트릭 jsonl append 모드
    lock = Lock()
    fail_list: list[dict] = []
    ok_count = 0
    err_count = 0
    t_start = time.time()
    last_progress = 0

    # write 실패 retry + 매 500 agent마다 checkpoint snapshot
    def _safe_write(fp, line):
        for retry in range(3):
            try:
                fp.write(line)
                fp.flush()
                return True
            except OSError as e:
                if retry == 2:
                    print(f"  [warn] file write failed 3x, dropping line: {e}")
                    return False
                time.sleep(0.5)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_one, aid, today, day_idx): aid for aid in remaining}
        # 각 결과마다 파일 open/close — Google Drive 등 sync 환경에서 핸들 깨짐 회피
        for fut in as_completed(futures):
            res = fut.result()
            with lock:
                try:
                    with metrics_path.open("a", encoding="utf-8") as fp_m:
                        _safe_write(fp_m, json.dumps(res, ensure_ascii=False) + "\n")
                except OSError as e:
                    print(f"  [warn] metrics write OSError: {e}")
                if res["status"] == "ok":
                    ok_count += 1
                    done_aids.add(res["aid"])
                else:
                    err_count += 1
                    fail_list.append(res)
            total_done = ok_count + err_count
            # 진행률
            if total_done - last_progress >= max(20, len(remaining)//20):
                elapsed = time.time() - t_start
                rate = total_done / elapsed if elapsed > 0 else 0
                eta = (len(remaining) - total_done) / rate if rate > 0 else 0
                print(f"  {total_done}/{len(remaining)} (ok={ok_count}, err={err_count}) "
                      f"@ {rate:.1f}/s, ETA {eta:.0f}s")
                last_progress = total_done
            # 500 agent마다 checkpoint snapshot (resume 안전)
            if total_done % 500 == 0:
                try:
                    done_path.write_text(json.dumps(sorted(done_aids), ensure_ascii=False), encoding="utf-8")
                except OSError as e:
                    print(f"  [warn] checkpoint snapshot failed: {e}")

    try:
        done_path.write_text(json.dumps(sorted(done_aids), ensure_ascii=False), encoding="utf-8")
        failed_path.write_text(json.dumps(fail_list, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"  [warn] final checkpoint write failed: {e}")

    agent_elapsed = time.time() - t_start
    print(
        f"[Day {day_idx} {day_str}] agent phase done in {agent_elapsed:.0f}s "
        f"— ok={ok_count}, err={err_count}"
    )
    timing_report = _write_timing_diagnostics(day_str, metrics_path)
    day_result = {
        "day": day_str,
        "ok": int(timing_report.get("agents_ok") or 0),
        "err": int(timing_report.get("agents_error") or 0),
        "agent_elapsed_sec": agent_elapsed,
        "night2_elapsed_sec": 0.0,
        "elapsed_sec": agent_elapsed,
        "timing_top": (timing_report.get("bottleneck_rank") or [])[:10],
    }

    # ═══════════════════════════════════════════
    # Night Phase 2 — 상호작용 대상 선정 + 의도 분류 LLM
    # (노션 다이어그램 Phase 2: 매일 자정 자동 동작 → Phase 3로 D+1 Plan에 영향)
    # ═══════════════════════════════════════════
    try:
        from night_interaction import select_interaction_pairs
        from night_intent_llm import run_intent_classification
        t_n2 = time.time()
        # 멱등성: 같은 day Conversation 이미 14,000건 이상 적재됐으면 Night2 전체 skip
        # (select_interaction_pairs까지 다시 도는 비용 회피)
        from neo4j_load._common import driver_session as _n2_session
        try:
            with _n2_session() as _s:
                _n2_existing = _s.run(
                    "MATCH (c:Conversation) WHERE c.day = date($d) RETURN count(c) AS n",
                    d=day_str
                ).single()["n"]
        except Exception:
            _n2_existing = 0
        if _n2_existing >= 50:
            print(f"  [Night2] {day_str}: 이미 {_n2_existing} Conversation 적재됨 — 전체 skip")
        else:
            pairs = select_interaction_pairs(today, verbose=False)
            if pairs:
                print(f"  [Night2] {len(pairs)} pairs, classifying intents ...")
                n2_stats = run_intent_classification(today, pairs, workers=workers, verbose=False)
                wstats = n2_stats.get("write", {})
                by_intent = wstats.get("by_intent", {})
                print(f"  [Night2] Conversation +{wstats.get('created',0)} "
                      f"(약속={by_intent.get('약속',0)}, 이슈={by_intent.get('이슈',0)}, "
                      f"추천={by_intent.get('추천',0)}, 기타={by_intent.get('기타',0)}) "
                      f"in {time.time()-t_n2:.0f}s")
            else:
                print(f"  [Night2] no candidate pairs for {day_str}")
        day_result["night2_elapsed_sec"] = time.time() - t_n2
    except Exception as e:
        print(f"  [Night2] failed: {e}")

    day_result["elapsed_sec"] = time.time() - t_start
    print(
        f"[Day {day_idx} {day_str}] done in {day_result['elapsed_sec']:.0f}s "
        f"(agent={day_result['agent_elapsed_sec']:.0f}s, "
        f"Night2={day_result['night2_elapsed_sec']:.0f}s)"
    )
    return day_result


# =========================================================
# 메인 — 3일 순차
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-01", help="시뮬 시작일")
    ap.add_argument("--days", type=int, default=3, help="시뮬 일수")
    ap.add_argument("--limit", type=int, default=None, help="agent 수 제한 (dry-run용)")
    ap.add_argument("--gu", default=None, help="자치구 코드 필터 (예: 11680 강남)")
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    agents = fetch_agents(limit=args.limit, gu_only=args.gu)
    print(f"=== 시뮬 시작 ===")
    print(f"  agents: {len(agents)}, days: {args.days}, start: {start}, workers: {args.workers}")
    print(f"  output: {OUT_DIR}")
    print()

    summary = []
    for i in range(args.days):
        today = start + timedelta(days=i)
        s = run_day(agents, today, day_idx=i, workers=args.workers)
        summary.append(s)
        # 매일 최신 summary를 먼저 원자적으로 저장한 뒤 선택적 외부 백업.
        write_json_atomic(
            OUT_DIR / "summary.json",
            {"summary": summary, "args": vars(args), "updated_at": datetime.now().isoformat()},
        )
        _daily_backup(today.isoformat(), s, agents)
        print()

    print("=== 시뮬 종료 ===")
    for s in summary:
        print(f"  Day {s['day']}: ok={s['ok']}, err={s['err']}, {s['elapsed_sec']:.0f}s")
    total_elapsed = sum(s["elapsed_sec"] for s in summary)
    total_ok = sum(s["ok"] for s in summary)
    total_err = sum(s["err"] for s in summary)
    print(f"\n  TOTAL: {total_ok} ok / {total_err} err, {total_elapsed:.0f}s")

    write_json_atomic(
        OUT_DIR / "summary.json",
        {"summary": summary, "args": vars(args), "completed_at": datetime.now().isoformat()},
    )


if __name__ == "__main__":
    main()

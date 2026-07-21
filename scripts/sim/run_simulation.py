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
import json
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neo4j_load"))
from _common import driver_session  # noqa: E402
from dawn_context import build_dawn_context  # noqa: E402
from stage1_intent import call_stage1  # noqa: E402
from stage2_poi import call_stage2, merge_to_final_events  # noqa: E402
from plan_writer import (  # noqa: E402
    write_plan, track_policy_usage,
    night_finalize_yesterday, night_create_state,
    apply_grant_to_prev_state, get_grant_amount,
    _grant_for_single_policy,
    aggregate_policy_spend, validate_policy_spend,
)


# Google Drive 동기화 폴더(G:\)는 file write 충돌 위험 → 로컬 디스크 사용
# 기본: ~/sim_output (Windows: C:\Users\<user>\sim_output, Linux: $HOME/sim_output)
import os
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
    q = "MATCH (a:Agent) WHERE (a)-[:LIVES_AT]->() "
    if gu_only:
        q += "AND (a)-[:LIVES_AT]->(:POI)-[:IN_DONG]->(:Dong)<-[:HAS_DONG]-(:District {code:$gu}) "
    q += "RETURN a.id AS id ORDER BY a.id"
    if limit:
        q += f" LIMIT {limit}"
    with driver_session() as s:
        if gu_only:
            return [r["id"] for r in s.run(q, gu=gu_only)]
        return [r["id"] for r in s.run(q)]


# =========================================================
# 한 agent의 1일 처리 (스레드 워커)
# =========================================================
def _build_policy_budget_summary(policies: list[dict] | None, prev_policy_used: dict[str, int] | None,
                                  grants_applied_today: dict[str, int] | None = None,
                                  grant_remaining: dict[str, int] | None = None) -> str:
    """Stage2 LLM에 노출할 정책 예산 요약 텍스트.

    - subsidy/voucher: 잔여 cap = cap - used
    - grant: 오늘 받은 지원금 + 누적 잔여액 명시 (LLM이 거래별 policy_spend 분리할 때 참조)
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
            lines.append(
                f"{pid} {name} [정부 지원금] — {' / '.join(parts)}. 본인 부담 0. "
                f"하는 소비는 지원금으로 결제(policy_spend 기입, 개인 자산 보존). "
                f"또 돈 때문에 미뤄온 소비가 있었다면 부담 없어진 지금 해소 가능(지급액 클수록 여지 큼). 강요 아닌 기회."
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
        spending_decile = ctx.persona.get("spending_decile")   # 소비 10분위 (1~10, decile_grants 정책용)
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
            amt = _grant_for_single_policy(income, pol, decile=spending_decile)
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
        # 어제 grant_remaining 파싱 → 1번만, 아래에서 재사용
        prev_grant_remaining = _read_state_json(ctx.state, "grant_remaining")
        # Stage2 LLM 노출용 잔여 가용액 (어제까지 잔여 + 오늘 받음)
        grant_avail_today: dict[str, int] = dict(prev_grant_remaining)
        for pid, amt in grants_applied_today.items():
            grant_avail_today[pid] = grant_avail_today.get(pid, 0) + int(amt)
        ctx.persona["policy_budget_summary"] = _build_policy_budget_summary(
            ctx.policy, prev_used_for_budget, grants_applied_today, grant_avail_today
        )

        # 사용처 제한 정책(민생회복 소비쿠폰류, poi_restricted=true) 감지
        # → 쿠폰 잔액이 있으면 Stage2에 [쿠폰] 마커·정렬 가점 활성 + 정책사용 하드검증
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
        # Stage2 actual_spent를 상대 가중치로만 쓰고, 오늘 총지출을 p×가용자산(지원금 포함)으로
        # 재정규화 → 소득탄력성·MPC 정상화. CONSUMPTION_MODEL=legacy 로 기존 동작 복귀.
        cm_meta = {"applied": False}
        if os.environ.get("CONSUMPTION_MODEL", "propensity") != "legacy":
            from consumption import apply_consumption_model
            _is_weekend = today.weekday() >= 5
            # 제한 grant(사용처 제한 poi_restricted / 업종 스코프 target_l1s)는
            # '제한 예산 봉투'로 분리 — 필터 통과 거래에만 흐름 (업종 DiD 전제).
            # 무제한 grant만 가용자산에 합산 (기존 P009 동작 그대로).
            # 정책 유형 일반화: 어떤 grant든 속성(poi_restricted·target_l1s)이
            # 곧 봉투 필터가 된다 — 쿠폰 특화 하드코딩 없음.
            _pol_by_id = {p["id"]: p for p in (ctx.policy or [])}
            _envelopes = []
            _grant_unrestricted: dict[str, int] = {}
            for pid, amt in grant_avail_today.items():
                pol = _pol_by_id.get(pid) or {}
                scoped = bool(pol.get("poi_restricted")) or bool(pol.get("target_l1s"))
                if scoped and int(amt) > 0:
                    _envelopes.append({
                        "pid": pid, "amount": int(amt),
                        "require_poi_eligible": bool(pol.get("poi_restricted")),
                        "categories": (pol.get("target_l1s") or None),
                    })
                else:
                    _grant_unrestricted[pid] = int(amt)
            cm_meta = apply_consumption_model(
                events,
                daily=ctx.persona.get("daily_we") if _is_weekend else ctx.persona.get("daily_wd"),
                income_tier=income,
                tendency=ctx.persona.get("tendency"),
                balance=(ctx.state or {}).get("balance"),
                grant_avail=_grant_unrestricted,
                llm_propensity=getattr(s1, "daily_propensity", None),
                restricted_envelopes=_envelopes,
            )

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
            "policy_spend_today": sum(today_policy_spend.values()),
            "grant_remaining_total": sum(merged_grant_remaining.values()),
            "policy_spend_corrected": policy_spend_corrected,
            "s1_attempts": m1["attempt"] + 1,
            "s1_t_llm": m1.get("t_llm"),   # t_s1 중 순수 LLM 왕복(재시도 포함) — 나머지=파싱·검증·프롬프트빌드
            # ── Stage2 병목 분해 (t_s2 = 후보조회 + 최근쿼리 + 프롬프트 + LLM + 리뷰조회 + 파싱/기타) ──
            "s2_timing": m2.get("s2_timing"),
            # ── Dawn(Neo4j 조회) 병목 분해 — 메모리 누적 시 t_memory 증가 감지 ──
            "dawn_timing": getattr(ctx, "dawn_timing", None),
            "s2_attempts": (m2.get("attempt", 0) or 0) + 1 if not m2.get("skipped") else 0,
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
# 매일 데이터 저장 — 마운트 스토리지 백업 (워크로드 비정상 종료 대비)
# =========================================================
def _daily_backup(day_str: str, day_summary: dict) -> None:
    """매일 하루 끝에 그날 산출물을 별도 스토리지(BACKUP_DIR)에 저장.

    GPU LIVE 워크로드는 메모리 과다 시 비정상 종료될 수 있고, 재생성 시 마운트
    스토리지만 유지된다(메가존 안내). 그래서 매일:
      1) 그날 State snapshot을 JSONL로 export (Neo4j data가 날아가도 산출물 복구 가능,
         online read라 DB 정지 불필요)
      2) metrics/checkpoints/summary 를 BACKUP_DIR로 복사 (resume 가능 상태 보존)
      3) backup_manifest.jsonl 에 진척 기록
    BACKUP_DIR 미지정이면 아무것도 안 함(무해). 실패해도 런은 계속(치명 아님).

    ※ 근본 대비: SIM_OUTPUT_DIR·Neo4j data 디렉토리 자체를 마운트 스토리지에 두는 것.
       이 함수는 그 위에 '그날 스냅샷'을 얹는 추가 안전장치.
    """
    bdir = os.environ.get("BACKUP_DIR")
    if not bdir:
        return
    import shutil
    bpath = Path(bdir)
    try:
        bpath.mkdir(parents=True, exist_ok=True)
        # 1) 그날 State snapshot (정책 효과 분석 핵심 필드) — online read
        state_f = bpath / f"state_{day_str}.jsonl"
        n_state = 0
        try:
            with driver_session() as s, state_f.open("w", encoding="utf-8") as fp:
                for r in s.run(
                    "MATCH (st:State {day: date($d)}) "
                    "RETURN st.agent_id AS aid, st.balance AS balance, "
                    "st.month_spent AS month_spent, st.grant_received AS grant_received, "
                    "st.grant_remaining AS grant_remaining, st.policy_used AS policy_used, "
                    "st.mood AS mood, st.fatigue AS fatigue", d=day_str,
                ):
                    fp.write(json.dumps(dict(r), ensure_ascii=False) + "\n")
                    n_state += 1
        except Exception as e:
            print(f"  [backup] State export 실패({day_str}): {e}")
        # 2) 그날 산출물 파일 복사 (metrics/checkpoints — resume 재개용)
        for sub in ("metrics", "checkpoints"):
            src = OUT_DIR / sub
            if not src.exists():
                continue
            dst = bpath / sub
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob(f"*{day_str}*"):
                try:
                    shutil.copy2(f, dst / f.name)
                except OSError:
                    pass
        # stage2_slow / summary 도 함께
        for name in ("summary.json", "stage2_slow.jsonl"):
            src = OUT_DIR / name
            if src.exists():
                try:
                    shutil.copy2(src, bpath / name)
                except OSError:
                    pass
        # 3) manifest — 어디까지 백업됐는지 (재개 판단용)
        with (bpath / "backup_manifest.jsonl").open("a", encoding="utf-8") as fp:
            fp.write(json.dumps({
                "day": day_str, "backed_up_at": datetime.now().isoformat(timespec="seconds"),
                "state_rows": n_state, **day_summary,
            }, ensure_ascii=False) + "\n")
        print(f"  [backup] {day_str} → {bdir} (State {n_state}행 + metrics/checkpoints 복사 완료)")
    except Exception as e:
        print(f"  [backup] {day_str} 백업 실패(치명 아님, 런 계속): {e}")


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

    # ── Stage2 병목 집계(러닝 합) + 느린 케이스 상세 로그 ──
    slow_sec = float(os.environ.get("STAGE2_SLOW_SEC", "45"))   # t_s2 이 값 초과 → 상세 로그
    slow_path = METRICS_DIR.parent / "stage2_slow.jsonl"
    s2agg = {"n": 0, "t_cand": 0.0, "t_recent": 0.0, "t_prompt": 0.0,
             "t_llm": 0.0, "t_review": 0.0, "t_parse": 0.0, "t_s2": 0.0,
             "n_llm": 0, "n_slow": 0}
    # Dawn(Neo4j 조회) 병목 집계 — 메모리 누적 영향(t_memory) 추적
    dawnagg = {"n": 0, "t_dawn": 0.0, "t_memory": 0.0, "t_knows_poi": 0.0,
               "t_social": 0.0, "t_persona": 0.0, "t_state": 0.0,
               "n_mem_ret": 0, "n_mem_total": 0, "n_mem_total_cnt": 0}

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
                    # Stage2 병목 집계 + 느린 케이스 상세 로그
                    # Dawn 병목 집계
                    dt = res.get("dawn_timing") or {}
                    t_dawn = res.get("timing_t_dawn")
                    if dt and t_dawn is not None:
                        dawnagg["n"] += 1
                        dawnagg["t_dawn"] += float(t_dawn)
                        for k in ("t_memory", "t_knows_poi", "t_social", "t_persona", "t_state"):
                            dawnagg[k] += float(dt.get(k) or 0.0)
                        dawnagg["n_mem_ret"] += int(dt.get("n_memory_ret") or 0)
                        if dt.get("n_memory_total") is not None:
                            dawnagg["n_mem_total"] += int(dt["n_memory_total"])
                            dawnagg["n_mem_total_cnt"] += 1
                    st = res.get("s2_timing") or {}
                    t_s2 = res.get("timing_t_s2")
                    if st and t_s2 is not None:
                        s2agg["n"] += 1
                        for k in ("t_cand", "t_recent", "t_prompt", "t_llm", "t_review"):
                            s2agg[k] += float(st.get(k) or 0.0)
                        s2agg["n_llm"] += int(st.get("n_llm_calls") or 0)
                        s2agg["t_s2"] += float(t_s2)
                        # 파싱/기타 = t_s2 - 측정된 구간 합 (Neo4j 세션·검증·fallback 등)
                        measured = sum(float(st.get(k) or 0.0)
                                       for k in ("t_cand", "t_recent", "t_prompt", "t_llm", "t_review"))
                        s2agg["t_parse"] += max(0.0, float(t_s2) - measured)
                        if float(t_s2) >= slow_sec:
                            s2agg["n_slow"] += 1
                            try:
                                with slow_path.open("a", encoding="utf-8") as fp_s:
                                    fp_s.write(json.dumps({
                                        "aid": res["aid"], "day": day_str, "t_s2": round(float(t_s2), 2),
                                        "t_s1": res.get("timing_t_s1"),
                                        "s2": st, "n_events": res.get("n_events"),
                                        "s2_attempts": res.get("s2_attempts"),
                                        "review_lookup_count": res.get("review_lookup_count"),
                                        "tokens_in": res.get("tokens_in"), "tokens_out": res.get("tokens_out"),
                                    }, ensure_ascii=False) + "\n")
                            except OSError:
                                pass
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

    elapsed = time.time() - t_start
    print(f"[Day {day_idx} {day_str}] done in {elapsed:.0f}s — ok={ok_count}, err={err_count}")

    # ── Stage2 병목 요약 (어느 단계가 t_s2를 지배하는지 한눈에) ──
    if s2agg["n"] > 0:
        n = s2agg["n"]
        avg = {k: s2agg[k] / n for k in ("t_cand", "t_recent", "t_prompt", "t_llm", "t_review", "t_parse", "t_s2")}
        share = {k: (100.0 * s2agg[k] / s2agg["t_s2"] if s2agg["t_s2"] > 0 else 0.0)
                 for k in ("t_cand", "t_recent", "t_prompt", "t_llm", "t_review", "t_parse")}
        print(f"  [Stage2 병목] 평균 t_s2={avg['t_s2']:.1f}s "
              f"= 후보조회 {avg['t_cand']:.1f}s({share['t_cand']:.0f}%) "
              f"+ 최근쿼리 {avg['t_recent']:.2f}s({share['t_recent']:.0f}%) "
              f"+ 프롬프트 {avg['t_prompt']:.2f}s({share['t_prompt']:.0f}%) "
              f"+ LLM {avg['t_llm']:.1f}s({share['t_llm']:.0f}%) "
              f"+ 리뷰조회 {avg['t_review']:.1f}s({share['t_review']:.0f}%) "
              f"+ 파싱/기타 {avg['t_parse']:.1f}s({share['t_parse']:.0f}%)")
        print(f"  [Stage2 병목] 평균 LLM 호출수 {s2agg['n_llm']/n:.2f}회/agent, "
              f"느린케이스(t_s2≥{slow_sec:.0f}s) {s2agg['n_slow']}건 → {slow_path.name}")

    # ── Dawn(Neo4j 조회) 병목 요약 — 메모리 누적 영향 추적 ──
    if dawnagg["n"] > 0:
        n = dawnagg["n"]
        da = {k: dawnagg[k] / n for k in ("t_dawn", "t_memory", "t_knows_poi", "t_social", "t_persona", "t_state")}
        mem_share = 100.0 * dawnagg["t_memory"] / dawnagg["t_dawn"] if dawnagg["t_dawn"] > 0 else 0.0
        memtot = (f", 메모리노드 평균 {dawnagg['n_mem_total']/dawnagg['n_mem_total_cnt']:.0f}개/agent"
                  if dawnagg["n_mem_total_cnt"] > 0 else " (총 메모리수는 DAWN_MEM_COUNT=1 로 측정)")
        print(f"  [Dawn 병목] 평균 t_dawn={da['t_dawn']:.2f}s "
              f"= 메모리조회 {da['t_memory']:.3f}s({mem_share:.0f}%) "
              f"+ knows_poi {da['t_knows_poi']:.3f}s + social {da['t_social']:.3f}s "
              f"+ persona {da['t_persona']:.3f}s + state {da['t_state']:.3f}s "
              f"| 반환메모리 {dawnagg['n_mem_ret']/n:.1f}개/agent{memtot}")

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
        from _common import driver_session as _n2_session
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
            return {"day": day_str, "ok": ok_count, "err": err_count, "elapsed_sec": elapsed}
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
    except Exception as e:
        print(f"  [Night2] failed: {e}")

    return {"day": day_str, "ok": ok_count, "err": err_count, "elapsed_sec": elapsed}


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
        # ★ 매일 하루 끝에 별도 스토리지 백업 (BACKUP_DIR 지정 시) — 워크로드 종료 대비
        _daily_backup(today.isoformat(), s)
        print()

    print("=== 시뮬 종료 ===")
    for s in summary:
        print(f"  Day {s['day']}: ok={s['ok']}, err={s['err']}, {s['elapsed_sec']:.0f}s")
    total_elapsed = sum(s["elapsed_sec"] for s in summary)
    total_ok = sum(s["ok"] for s in summary)
    total_err = sum(s["err"] for s in summary)
    print(f"\n  TOTAL: {total_ok} ok / {total_err} err, {total_elapsed:.0f}s")

    (OUT_DIR / "summary.json").write_text(
        json.dumps({"summary": summary, "args": vars(args)}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()

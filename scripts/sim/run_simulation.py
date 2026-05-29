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
from datetime import date, timedelta
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
def process_one(aid: str, today: date, day_idx: int) -> dict:
    """1 agent 1일. 결과 메타 dict 반환 (실패 시 status='error')."""
    t0 = time.time()
    try:
        ctx = build_dawn_context(aid, today)
        if not ctx.persona:
            return {"aid": aid, "status": "no_persona", "elapsed": time.time() - t0}

        # grant 정책 — effective_from 당일 새벽에 전날 State 잔액에 지원금 추가
        # Dawn 컨텍스트가 업데이트된 잔액을 보여줘 LLM이 추가 예산 인식
        grant_applied = 0
        for pol in (ctx.policy or []):
            if pol.get("type") == "grant":
                eff = pol.get("effective_from", "")
                if str(today) == eff:
                    income = ctx.persona.get("income") or ctx.persona.get("p_income_level") or ""
                    grant_applied = get_grant_amount(income, ctx.policy)
                    if grant_applied > 0:
                        apply_grant_to_prev_state(aid, today, grant_applied)
                    break

        # grant 적용 후 컨텍스트 재빌드 (잔액 반영)
        if grant_applied > 0:
            ctx = build_dawn_context(aid, today)

        s1, m1 = call_stage1(aid, today, ctx=ctx)
        s2, _cands, m2 = call_stage2(aid, s1, ctx.persona, today)
        events = merge_to_final_events(s1, s2, ctx.persona)

        # 정책 cap 잔액 추적 (만족도·소비액은 Stage2 LLM이 설정)
        prev_policy_used = ctx.get_policy_used()
        updated_policy_used = track_policy_usage(
            events, ctx.persona,
            active_policies=ctx.policy,
            policy_used=prev_policy_used,
        )

        # 활성 정책 카테고리 셋 (오늘 Dawn 컨텍스트에서 추출) — 사후 측정용 라벨
        active_policy_cats: set[str] = set()
        for pol in ctx.policy:
            for l1 in (pol.get("target_l1s") or []):
                if l1:
                    active_policy_cats.add(l1)

        day_type = "weekend" if today.weekday() >= 5 else "weekday"
        tokens_in = m1["tokens_in"] + (m2.get("tokens_in") or 0)
        tokens_out = m1["tokens_out"] + (m2.get("tokens_out") or 0)
        _, n_inc = write_plan(aid, today, events, day_type, tokens_in, tokens_out)

        # Night Phase — Day 1 새벽엔 어제(Day 0) Plan 없으므로 finalize는 Day 2 이상에서만
        n_mem = 0
        if day_idx >= 1:
            n_mem = night_finalize_yesterday(aid, today)
        state = night_create_state(aid, today, policy_used=updated_policy_used)

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
            "n_events": len(events), "n_includes": n_inc,
            "n_visited_memories": n_mem,
            "avg_sat": round(avg_sat, 3) if avg_sat is not None else None,
            "balance": state.get("balance"),
            "mood": round(state.get("mood", 0), 3) if state else None,
            "fatigue": round(state.get("fatigue", 0), 3) if state else None,
            "tokens_in": tokens_in, "tokens_out": tokens_out,
            "policy_hits": policy_hits,
            "s1_attempts": m1["attempt"] + 1,
            "s2_attempts": (m2.get("attempt", 0) or 0) + 1 if not m2.get("skipped") else 0,
            # Stage 2 fallback 카운트 (사후 분석용)
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

    elapsed = time.time() - t_start
    print(f"[Day {day_idx} {day_str}] done in {elapsed:.0f}s — ok={ok_count}, err={err_count}")

    # ═══════════════════════════════════════════
    # Night Phase 2 — 상호작용 대상 선정 + 의도 분류 LLM
    # (노션 다이어그램 Phase 2: 매일 자정 자동 동작 → Phase 3로 D+1 Plan에 영향)
    # ═══════════════════════════════════════════
    try:
        from night_interaction import select_interaction_pairs
        from night_intent_llm import run_intent_classification
        t_n2 = time.time()
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

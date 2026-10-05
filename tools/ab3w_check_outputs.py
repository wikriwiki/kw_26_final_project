"""3주 A/B 시험 출력 점검 — 실측 비교에 쓸 출력이 빠짐없이·제대로 나왔는지 (2026-10-06).

    python tools/ab3w_check_outputs.py <BASE 폴더>     (실행기 9단계 채점 뒤에 tools/ab3w_score.sh 가 부른다)
모든 항목을 출력하고, 하나라도 실패면 종료 코드 1. 숫자의 크기가 아니라 '비교에 쓸 수 있는가'를 본다.
"""
import glob
import json
import os
import sys

B = sys.argv[1].rstrip("/")
bad = []


def ok(cond, msg):
    print(("  [통과] " if cond else "  [실패] ") + msg)
    if not cond:
        bad.append(msg)


def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if os.path.exists(path) else []


man = json.load(open(f"{B}/run_manifest.json", encoding="utf-8"))
case, tag = man["case"], man["tag"]
roster = json.load(open(f"{B}/roster.json", encoding="utf-8"))
N, pre_days, post_days = len(roster), man["pre_days"], man["post_days"]
print(f"== {case} {tag}: {N}명, 정책 전 {pre_days}일({man['pre_start']}~), 두 갈래 {post_days}일({man['start']}~{man['post_end']}), 정책 {man['policy_id'] or '(환경형)'}")
log = open(f"{B}/orchestrate.log", encoding="utf-8").read()
ok("=== 9 채점" in log or "=== 끝" in log, "실행기가 채점 단계까지 갔다")
ok(os.path.exists(f"{B}/fork.marker"), "그래프 복제(두 그래프 같음 확인)")

# 하루 지표: 사람·날 전부 ok, 건너뜀 0, 정책 노출은 정책 있음 쪽에만, 계획 기준선이 쓰였다
for side, n in (("pre", pre_days), ("on", post_days), ("off", post_days)):
    files = sorted(glob.glob(f"{B}/{side}/metrics/day_*.jsonl"))
    rs = [r for f in files for r in rows(f)]
    ok(len(files) == n and len(rs) == N * n and all(r.get("status") == "ok" for r in rs),
       f"{side}: 하루 지표 {len(files)}일 x {N}명, 모두 ok ({len(rs)}행)")
    pol = [r for r in rs if r.get("experience_policy_ids")]
    if side == "on" and man["policy_id"]:
        ok(len(pol) > 0, f"on: 정책이 보인 사람-날 {len(pol)}/{len(rs)}")
    else:
        ok(len(pol) == 0, f"{side}: 정책이 보인 사람-날 0 ({len(pol)})")
    if side != "pre":
        spend = [r for r in rs if (r.get("cm_planned_total") or 0) > 0]
        based = sum(1 for r in spend if r.get("cm_plan_baseline"))
        ok(not spend or len(spend) - based <= max(1, len(spend) // 10),
           f"{side}: 계획 통로 기준선 사용 {based}/{len(spend)}")
        envs = {r.get("experience_environment_id") for r in rs}
        want = man["env_on"] if side == "on" else man["env_off"]
        ok(envs == {want or None}, f"{side}: 사회 배경 {envs} (기대 {want or '없음'})")

# 원장
sec_on, sec_off = rows(f"{B}/on/sector.ledger.jsonl"), rows(f"{B}/off/sector.ledger.jsonl")
ok(len(sec_on) == len(sec_off) == N * post_days, f"업종 원장 행 수 on {len(sec_on)} / off {len(sec_off)} (기대 {N * post_days})")
need = ("total_spent", "offline_spent", "online_spent", "by_sub", "by_l1", "instant_discount_won", "policy_rebate_won")
ok(all(all(k in r for k in need) for r in sec_on + sec_off), "업종 원장 칸: " + ", ".join(need))
ok(all(os.path.exists(f"{B}/{s}/sector.ledger.jsonl.manifest.json") for s in ("on", "off")), "업종 원장 manifest")
ok(sum(r["offline_spent"] for r in sec_on) > 0 and sum(r["offline_spent"] for r in sec_off) > 0, "가게 지출이 0 이 아니다")
disc = sum(r["instant_discount_won"] for r in sec_on); reb = sum(r["policy_rebate_won"] for r in sec_on)
ok(sum(r["instant_discount_won"] + r["policy_rebate_won"] for r in sec_off) == 0, "정책 없음 쪽 할인·환급 0")
pid = man["policy_id"]
if pid in ("P014", "P016"):
    ok(disc > 0, f"정책 있음 쪽 결제 할인 합 {disc:,}원 (0 이면 할인이 결제에 안 닿았다)")
if pid == "P015":
    print(f"  [참고] 정책 있음 쪽 할인 {disc:,}원 · 환급 {reb:,}원 (하루 시험은 외식 환급 조건(4번째)에 못 닿을 수 있다)")
if pid in ("P010", "P013"):
    pol_on, pol_off = rows(f"{B}/on/policy.ledger.jsonl"), rows(f"{B}/off/policy.ledger.jsonl")
    ok(len(pol_on) == len(pol_off) == N * post_days, f"정책 원장 행 수 on {len(pol_on)} / off {len(pol_off)}")
    rec = sum(r["grant_received_cumulative"] for r in pol_on if r["day"] == man["post_end"])
    used = sum(r["grant_spent_today"] for r in pol_on)
    ok(rec > 0, f"지원금 받은 합 {rec:,}원 (받는 사람이 있어야 한다)")
    print(f"  [참고] 지원금으로 낸 돈 {used:,}원, 사용처 지출(정책 규칙) on {sum(r['eligible_offline_spent'] for r in pol_on):,} / off {sum(r['eligible_offline_spent'] for r in pol_off):,}")
    ok(all(r["grant_received_cumulative"] == 0 for r in pol_off), "정책 없음 쪽 지원금 0")
if pid == "P012":
    cb = rows(f"{B}/on/cashback.ledger.jsonl")
    ok(len(cb) > 0, f"캐시백 원장 {len(cb)}행")
    ok(sum(r.get("sangsaeng_eligible_offline_spent", 0) for r in sec_on) > 0, "적립 대상 가게 지출이 기록됐다")
if case == "distancing":
    d_on, d_off = rows(f"{B}/on/distancing.ledger.jsonl"), rows(f"{B}/off/distancing.ledger.jsonl")
    ok(len(d_on) == len(d_off) == N * post_days, f"거리두기 원장 행 수 {len(d_on)}/{len(d_off)}")
    ok(all(all(k in r for k in ("restaurant_won", "cafe_won", "retail_won", "korean_restaurant_won")) for r in d_on + d_off),
       "거리두기 원장 칸(식사·카페·소매·한식)")

# 기억 모음·덤프
for side in ("on", "off"):
    m = json.load(open(f"{B}/{side}/dossier.jsonl.manifest.json", encoding="utf-8"))
    ok(m["expected_days"] == pre_days + post_days and not m["state_day_gaps"]
       and m["totals"]["states"] == N * (pre_days + post_days) and m["totals"]["memories"] > 0,
       f"{side}: 기억 모음 상태 {m['totals']['states']} · 기억 {m['totals']['memories']} · 계획항목 {m['totals']['plan_items']}")
    ok(os.path.exists(f"{B}/{side}/graph_backup/SHA256SUMS") and os.path.exists(f"{B}/{side}/external_copy_verified.txt"),
       f"{side}: 그래프 덤프·체크섬")
ok(os.path.exists(f"{B}/score/score_ab3w.json"), "채점 결과 파일(score/score_ab3w.json)")
print(f"== {case}: {'모두 통과' if not bad else str(len(bad)) + '개 실패'}")
sys.exit(1 if bad else 0)

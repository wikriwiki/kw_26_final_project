#!/usr/bin/env bash
# P012 28일 창 — 문턱이 월 단위라 짧은 창에서는 유인이 작동하지 않는다.
# 근거: experiments/error_budget/diagnosis_06.md
#
#   bash tools/run_p012_28d.sh
#
# **P012.json 은 손대지 않는다.** effective_from 만 되돌린 사본(P012_28d.json)을
# 쓴다. 원본은 값(10-25)과 notes(10-15)가 어긋나 있었고, 사본은 notes 가 적어
# 둔 설계를 따른다.
#
#   무정책 2021-10-01 ~ 10-14   (14일)
#   정책   2021-10-15 ~ 10-28   (14일)
#   합계 28일 · 월 경계 없음 · 같은 에이전트
set -uo pipefail
cd /data/repo
source /data/venv/bin/activate
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass
export PYTHONIOENCODING=utf-8 PYTHONPATH=/data/repo LLM_BASE_URL=http://localhost:8000/v1
# **EXP_SEED_SANGSAENG=0 이어야 한다.** 08_initial_state 는 적립 누적을
# `앵커 x 비율 x DAY_ZERO.day` 로 시드한다 — 월중에 시작하는 런이 "이미 그 달에
# 얼마를 썼다" 를 반영하려는 장치다. 그런데 이 런은 **10-01 에 시작**하므로
# 10월 누적은 0 이어야 한다. DAY_ZERO=2021-09-30 이면 .day=30 이라 **한 달치를
# 통째로 시드**하고, plan_writer 는 월 경계에서 리셋하지 않는다
# (`s.sangsaeng_month_spent = prev + today`, 누적만 한다).
# 그대로 두면 전원이 시작부터 문턱 언저리/초과가 되어 9일 창과 **반대 방향으로**
# 설계가 깨진다.
export EXP_SANGSAENG_BASE_RATIO=0.268 EXP_SEED_SANGSAENG=0 EXP_BALANCE_DAYS=39
export EXP_DURABLES=1 EXP_CATLINE=fold EXP_POLICY_ANONYMOUS=1 POLICY_POI_SORT_BOOST=0
# 창이 28일이면 지갑이 마른다 — 소득 주입이 필요하다(기억: 관측 창 한계 약 26일).
export EXP_DAILY_INCOME=anchor
# [회계 고침은 이 런에서 **끈다**]
# EXP_PLAN_DRIVES_TOTAL 은 개인별 기준 계획액을 쓰는데, 그 기준선은 **이 런의
# 워밍업 날**에서만 만들 수 있다(다른 런에서 가져오면 표본이 어긋난다 — 9일
# 창에서 limit 100 표본과 기준선이 2명만 겹친 적이 있다). 그런데 런이 시작돼야
# 워밍업 날이 생기므로 닭과 달걀이다. 게다가 consumption 은 기준선을 **한 번만
# 읽고 캐시**하므로 도중에 파일이 생겨도 안 잡힌다.
#
# 그래서 이 런은 **창 하나만 바꾼다.**
#   이 런        28일 창 · v5 · 회계 고침 **꺼짐**  <- 창의 효과를 홀로 잰다
#   다음 런      같은 창 · 이 런의 워밍업으로 만든 기준선 · 고침 켜짐
# 이렇게 해야 창과 회계를 **가려서** 읽을 수 있다. 한 번에 둘을 바꾸면
# 무엇이 움직였는지 못 가른다.
unset EXP_PLAN_BASELINE_FILE
export EXP_PLAN_DRIVES_TOTAL=0

OUT=/data/p012_28d; mkdir -p $OUT
LOG=$OUT/run.log
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $LOG; }
D0=2021-09-30; ST=2021-10-01; DY=28; N=500
OFF=2021-10-13:2021-10-14      # 정책 직전 이틀
ON=2021-10-27:2021-10-28       # 정책 13~14일째 (문턱이 쌓인 뒤)
VAR=${SIM_VARIANT:-v5}

# !! 순서가 생명이다 (2026-09-25) !!
#
# `97_reset_run_artifacts.py` 는 Plan·Memory·Conversation·State **그리고 Policy** 를
# 지운다(그 파일 47행). 정책을 먼저 적재하고 그 뒤에 reset 을 돌리면 **적재한 정책이
# 지워진 채로 런이 돈다.** 오류도 안 나고 로그에는 "적재 완료" 가 찍힌다.
#
# 그렇게 두 런을 잃었다:
#   · 이 런의 첫 시도  18일 × 500명을 정책 없이 돌았다 (성향에 10-15 단계가 없고
#                      policy_hits 가 모든 날 0 이었다)
#   · p013_ruler       12일 × 700명. 지원금이 한 푼도 안 나갔고 "정책 반응 +6.36%"
#                      가 사실은 예열 드리프트였다
#                      experiments/plan_channel/P013_evidence_is_weaker.md
#
# **reset → 정책 적재 → initial_state → 런** 이 맞는 순서다.
say "런 산출물 초기화 (Policy 도 함께 지워진다 — 적재는 이 다음이다)"
export SIM_PROMPT_VARIANT=$VAR SIM_OUTPUT_DIR=$OUT/28d_$VAR
rm -rf "$OUT/28d_$VAR"; mkdir -p "$OUT/28d_$VAR"
python scripts/neo4j_load/97_reset_run_artifacts.py > /dev/null 2>&1

say "정책 적재 — P012_28d (P012.json 은 손대지 않는다)"
python scripts/neo4j_load/10_load_grant_policy.py data/neo4j_load/policies/P012_28d.json 2>&1 | tail -3 | tee -a $LOG

# **관문**: 정책이 정말 그래프에 있고 지역에 걸렸는지 본다. 없으면 여기서 멈춘다 —
# 런을 걸어 놓고 18시간 뒤에 아는 일이 다시 없게 한다.
python - <<'PYEOF' 2>&1 | tee -a $LOG
import sys
sys.path.insert(0, "/data/repo/scripts")
from neo4j_load._common import driver_session
with driver_session() as s:
    rows = list(s.run(
        "MATCH (p:Policy) RETURN p.id AS id, p.type AS t, "
        "toString(p.effective_from) AS f, count{(p)-[:applied_to]->()} AS na"))
for r in rows:
    print("  정책 %s type=%s from=%s applied_to=%d개 지역"
          % (r["id"], r["t"], r["f"], r["na"]))
if not rows:
    sys.exit("  ** 거부: Policy 노드가 0개다. 런을 걸지 않는다")
if any(r["na"] == 0 for r in rows):
    sys.exit("  ** 거부: 어느 지역에도 안 걸린 정책이 있다. 런을 걸지 않는다")
PYEOF
[ ${PIPESTATUS[0]:-1} -ne 0 ] && { say "정책 관문 실패 — 그만둔다"; exit 1; }

say "=========== [28d_$VAR] $VAR · N=$N · $ST 부터 $DY 일 ==========="
DAY_ZERO=$D0 python scripts/neo4j_load/08_initial_state.py > /dev/null 2>&1
python -u scripts/sim/run_simulation.py --start $ST --days $DY --limit $N \
    --workers 48 --environment covid_2021 2>&1 | stdbuf -oL grep -E "^  Day|TOTAL" | tee -a $LOG

say "[28d_$VAR] 채점"
python scripts/sim/score_policy.py --policy P012 --off $OFF --on $ON \
    --label "28d_$VAR" --json-out "$OUT/score_28d_$VAR.json" --per-agent 2>&1 | tail -24 | tee -a $LOG

say "--- 문턱 모양: 반응이 언저리에 몰렸는가"
python scripts/report/threshold_response_shape.py "$OUT/28d_$VAR/metrics" \
    --off $OFF --on $ON 2>&1 | tee -a $LOG
say "--- 다음 런을 위한 개인 기준선 만들기 (이 런의 워밍업 3일)"
python - "$OUT/28d_$VAR/metrics" /data/plan_baseline_28d.json <<'PYEOF' 2>&1 | tee -a $LOG
import json, glob, os, sys, statistics as st
M, OUTF = sys.argv[1], sys.argv[2]
BASE = ("2021-10-01", "2021-10-02", "2021-10-03")
acc = {}
for f in sorted(glob.glob(os.path.join(M, "*.jsonl"))):
    day = os.path.basename(f)[4:-6]
    if day not in BASE:
        continue
    for l in open(f, encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        try:
            d = json.loads(l)
        except ValueError:
            continue
        if d.get("status") != "ok":
            continue
        v = d.get("cm_planned_total")
        if isinstance(v, (int, float)) and v > 0:
            acc.setdefault(d["aid"], []).append(float(v))
bl = {k: round(st.mean(v), 1) for k, v in acc.items() if v}
json.dump(bl, open(OUTF, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print("  기준선 %d명 -> %s (기준일 %s · 비교창 밖)" % (len(bl), OUTF, list(BASE)))
PYEOF
say "=== P012_28D_DONE ==="

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
export EXP_PLAN_BASELINE_FILE=/data/plan_baseline_28d.json

OUT=/data/p012_28d; mkdir -p $OUT
LOG=$OUT/run.log
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $LOG; }
D0=2021-09-30; ST=2021-10-01; DY=28; N=500
OFF=2021-10-13:2021-10-14      # 정책 직전 이틀
ON=2021-10-27:2021-10-28       # 정책 13~14일째 (문턱이 쌓인 뒤)
VAR=${SIM_VARIANT:-v5}

say "정책 적재 — P012_28d (P012.json 은 손대지 않는다)"
python - <<'PYEOF' 2>&1 | tee -a $LOG
import sys
sys.path.insert(0, "/data/repo/scripts")
from neo4j_load._common import driver_session
with driver_session() as s:
    s.run("MATCH (p:Policy) DETACH DELETE p")
    print("  비운 뒤 :Policy %d개" % s.run("MATCH (p:Policy) RETURN count(p) AS n").single()["n"])
PYEOF
python scripts/neo4j_load/10_load_grant_policy.py data/neo4j_load/policies/P012_28d.json 2>&1 | tail -3 | tee -a $LOG

say "=========== [28d_$VAR] $VAR · N=$N · $ST 부터 $DY 일 ==========="
export SIM_PROMPT_VARIANT=$VAR SIM_OUTPUT_DIR=$OUT/28d_$VAR
rm -rf "$OUT/28d_$VAR"; mkdir -p "$OUT/28d_$VAR"
python scripts/neo4j_load/97_reset_run_artifacts.py > /dev/null 2>&1
DAY_ZERO=$D0 python scripts/neo4j_load/08_initial_state.py > /dev/null 2>&1
python -u scripts/sim/run_simulation.py --start $ST --days $DY --limit $N \
    --workers 48 --environment covid_2021 2>&1 | stdbuf -oL grep -E "^  Day|TOTAL" | tee -a $LOG

say "[28d_$VAR] 채점"
python scripts/sim/score_policy.py --policy P012 --off $OFF --on $ON \
    --label "28d_$VAR" --json-out "$OUT/score_28d_$VAR.json" --per-agent 2>&1 | tail -24 | tee -a $LOG

say "--- 문턱 모양: 반응이 언저리에 몰렸는가"
python scripts/report/threshold_response_shape.py "$OUT/28d_$VAR/metrics" \
    --off $OFF --on $ON 2>&1 | tee -a $LOG
say "=== P012_28D_DONE ==="

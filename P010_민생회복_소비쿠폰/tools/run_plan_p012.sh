#!/usr/bin/env bash
# 계획의 정책 반응이 총액에 통과하는가 — 본런. 사전등록 experiments/plan_channel/prereg.md
#
#   bash tools/run_plan_p012.sh
#
# **거시 저장소(/data/repo)에서 돈다.** 기존 P012 읽기(result_r2_v5)와 창·N·환경이
# 같아야 비교가 이 고침의 효과만 말한다. 프롬프트는 **v5 그대로** — 바꾸는 것은
# 회계 한 줄이다(max(앵커,계획) -> 앵커 x clamp(계획/개인기준선) x SCALE).
set -uo pipefail
cd /data/repo
source /data/venv/bin/activate
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass
export PYTHONIOENCODING=utf-8 PYTHONPATH=/data/repo LLM_BASE_URL=http://localhost:8000/v1
export EXP_SANGSAENG_BASE_RATIO=0.268 EXP_SEED_SANGSAENG=1 EXP_BALANCE_DAYS=39
export EXP_DURABLES=1 EXP_CATLINE=fold EXP_POLICY_ANONYMOUS=1 POLICY_POI_SORT_BOOST=0
unset EXP_DAILY_INCOME
export EXP_PLAN_BASELINE_FILE=/data/plan_baseline_p012.json

OUT=/data/plan_p012; mkdir -p $OUT
LOG=$OUT/run.log
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $LOG; }
D0=2021-10-17; ST=2021-10-18; DY=9; N=500
OFF=2021-10-21:2021-10-22
ON=2021-10-25:2021-10-26

[ -f "$EXP_PLAN_BASELINE_FILE" ] || { say "기준선 파일이 없다 — 멈춘다"; exit 3; }
say "기준선 $(python -c "import json;print(len(json.load(open('$EXP_PLAN_BASELINE_FILE'))))")명"

# ------------------------------------------------------------- 관문
# **LLM 관문을 버렸다.** 처음 설계는 하루 x 100명을 두 번 돌려 총액이 달라지는지
# 봤는데, 에이전트 표본이 `--limit` 에 따라 달라진다는 것을 놓쳤다
# (fetch_agents 는 소비 10분위 비례 층화표본을 limit 별로 뽑는다).
# limit=100 표본은 기준선 500명과 **2명**만 겹쳤다. 그런데 관문은
# "79% 의 총액이 달라졌다 — 통과" 라고 답했다. 그 79% 는 회계가 아니라
# **런 간 LLM 잡음**이었다. 관문이 재려던 것을 못 재고 통과시킨 것이다.
#
# 그래서 관문을 **공짜로 확실한 것**으로 바꾼다.
#   1 본런이 쓸 표본과 기준선이 겹치는가 (LLM 0회)
#   2 회계 분기가 실제로 다른 수를 내는가 (LLM 0회, 단위시험과 같은 경로)
# 산술은 단위시험 15개가 이미 지킨다. 관문이 볼 것은 **닿는가** 뿐이다.
say "--- 관문 1: 본런 표본과 기준선이 겹치는가"
python - "$N" "$EXP_PLAN_BASELINE_FILE" <<'PYEOF' 2>&1 | tee -a $LOG
import json, sys
sys.path.insert(0, 'scripts/sim')
import run_simulation as R
n = int(sys.argv[1])
bl = set(json.load(open(sys.argv[2], encoding='utf-8')))
ids = set(R.fetch_agents(limit=n))
hit = len(ids & bl)
pct = 100.0 * hit / max(1, len(ids))
print('  본런 표본 %d명 · 기준선 %d명 · 겹침 %d (%.1f%%)' % (len(ids), len(bl), hit, pct))
if pct < 95.0:
    print('  **관문 실패** — 기준선이 표본을 못 덮는다. 고침이 거의 안 걸린다')
    sys.exit(3)
print('  관문 1 통과')
PYEOF
[ ${PIPESTATUS[0]:-1} -ne 0 ] && { say "관문 1 실패 — 본런을 돌리지 않는다"; exit 3; }

say "--- 관문 2: 회계 분기가 다른 수를 내는가 (LLM 0회)"
python - <<'PYEOF' 2>&1 | tee -a $LOG
import importlib, os, sys
sys.path.insert(0, 'scripts/sim')
ev = lambda e: [dict(category='식사', poi_id='A', actual_spent=e, policy_spend={},
                     coupon_eligible=True, actual_satisfaction=.9, price_factor=1)]
def run(flag, each, aid):
    os.environ['EXP_PLAN_DRIVES_TOTAL'] = flag
    import consumption as C
    importlib.reload(C)
    return C.apply_consumption_model(ev(each), daily=40000, income_tier='중',
                                     tendency='보통', balance=5_000_000, aid=aid)
import json
aid = sorted(json.load(open(os.environ['EXP_PLAN_BASELINE_FILE'], encoding='utf-8')))[0]
off = run('0', 30000, aid)
on = run('1', 30000, aid)
os.environ['EXP_PLAN_DRIVES_TOTAL'] = '1'
print('  기준선 보유 에이전트 %s' % aid)
print('  끄고 %d · 켜고 %d' % (off['personal_total'], on['personal_total']))
if off['personal_total'] == on['personal_total']:
    print('  **관문 실패** — 분기가 같은 수를 낸다. 기준선 파일이나 플래그가 안 닿았다')
    sys.exit(3)
print('  관문 2 통과 — 경로가 닿는다')
PYEOF
[ ${PIPESTATUS[0]:-1} -ne 0 ] && { say "관문 2 실패 — 본런을 돌리지 않는다"; exit 3; }


# ------------------------------------------------------------- 본런
TAG=plan_v5
if [ -f "$OUT/score_$TAG.json" ]; then say "[$TAG] 끝남 — 건너뜀"; exit 0; fi
say "=========== [$TAG] v5 · PLAN_DRIVES_TOTAL=1 · N=$N · $ST 부터 $DY 일 ==========="
export SIM_PROMPT_VARIANT=v5 SIM_OUTPUT_DIR=$OUT/$TAG EXP_PLAN_DRIVES_TOTAL=1
rm -rf "$OUT/$TAG"; mkdir -p "$OUT/$TAG"
python scripts/neo4j_load/97_reset_run_artifacts.py > /dev/null 2>&1
DAY_ZERO=$D0 python scripts/neo4j_load/08_initial_state.py > /dev/null 2>&1
python -u scripts/sim/run_simulation.py --start $ST --days $DY --limit $N \
    --workers 48 --environment covid_2021 2>&1 | stdbuf -oL grep -E "^  Day|TOTAL" | tee -a $LOG
say "[$TAG] 채점 — 그래프가 비워지기 전에 지금 한다"
python scripts/sim/score_policy.py --policy P012 --off $OFF --on $ON \
    --label "$TAG" --json-out "$OUT/score_$TAG.json" --per-agent 2>&1 | tail -24 | tee -a $LOG
say "=== PLAN_P012_DONE ==="

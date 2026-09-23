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

# ------------------------------------------------------------- 배선 관문
gate () {
  local TAG=$1 FLAG=$2
  say "--- 관문 [$TAG] PLAN_DRIVES_TOTAL=$FLAG · v5 · 하루 · 100명"
  export SIM_PROMPT_VARIANT=v5 SIM_OUTPUT_DIR=$OUT/$TAG EXP_PLAN_DRIVES_TOTAL=$FLAG
  rm -rf "$OUT/$TAG"; mkdir -p "$OUT/$TAG"
  python scripts/neo4j_load/97_reset_run_artifacts.py > /dev/null 2>&1
  DAY_ZERO=$D0 python scripts/neo4j_load/08_initial_state.py > /dev/null 2>&1
  python -u scripts/sim/run_simulation.py --start $ST --days 1 --limit 100 \
      --workers 48 --environment covid_2021 2>&1 | stdbuf -oL grep -E "^  Day|TOTAL" | tee -a $LOG
}

if [ ! -f "$OUT/gate_done" ]; then
  gate gate_off 0
  gate gate_on  1
  say "--- 관문 판정"
  python - "$OUT" <<'PYEOF' 2>&1 | tee -a $LOG
import json, glob, sys, statistics as st
out = sys.argv[1]
def rows(tag):
    r = []
    for f in glob.glob('%s/%s/metrics/*.jsonl' % (out, tag)):
        for l in open(f, encoding='utf-8'):
            l = l.strip()
            if not l:
                continue
            try:
                d = json.loads(l)
            except Exception:
                continue
            if d.get('status') == 'ok':
                r.append(d)
    return r
A, B = rows('gate_off'), rows('gate_on')
print('  계획  끄고 %d · 켜고 %d' % (len(A), len(B)))
if not A or not B:
    print('  **관문 실패** — 계획이 안 나왔다'); sys.exit(3)
ka = {d['aid']: d for d in A}
kb = {d['aid']: d for d in B}
both = [k for k in ka if k in kb]
diff = sum(1 for k in both
           if ka[k].get('cm_today_total_incl_online') != kb[k].get('cm_today_total_incl_online'))
print('  둘 다 있는 사람 %d · 총액이 달라진 사람 %d (%.1f%%)'
      % (len(both), diff, 100 * diff / max(1, len(both))))
if diff == 0:
    print('  **관문 실패** — 경로가 안 걸렸다(플래그나 기준선 파일이 안 닿았다)'); sys.exit(3)
def med(rs, k):
    v = [d[k] for d in rs if isinstance(d.get(k), (int, float))]
    return st.median(v) if v else None
a, b = med(A, 'cm_today_total_incl_online'), med(B, 'cm_today_total_incl_online')
d = (b - a) / a * 100
print('  총액 중앙  끄고 %s · 켜고 %s · 차이 %+.2f%%' % ('{:,}'.format(int(a)), '{:,}'.format(int(b)), d))
if abs(d) > 25:
    print('  **관문 실패** — 수준이 25%% 넘게 흔들렸다. 폭주다'); sys.exit(3)
print('  관문 통과 — 경로가 살아 있고 수준이 폭주하지 않는다')
PYEOF
  [ ${PIPESTATUS[0]:-1} -ne 0 ] && { say "관문에서 멈춘다 — 본런을 돌리지 않는다"; exit 3; }
  touch "$OUT/gate_done"
fi

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

#!/usr/bin/env bash
# 2층 가르기 본런 — 사전등록 experiments/split_anchor/prereg.md (+ amend_01_scope.md)
#
#   bash tools/run_split_p012.sh <KEEP_MEAN>
#
# **거시 저장소(/data/repo)에서 돈다.** 기존 P012 읽기(result_r2_v5)와 창·N·코드가
# 같아야 비교가 2층 가르기의 효과만 말한다. 그래서 파일을 덮지 않고 심는다
# (tools/patch_split_anchor.py — AST 동일 확인함).
#
# 순서는 사전등록 그대로다. 관문이 먼저고, 관문이 깨지면 본런을 안 돌린다.
#
#   나-1  배선 관문   v5(=online_share 없음) + SPLIT=1 로 하루.
#                    분기가 살아 있고 기준값이 현행과 같은지 본다.
#   나-2  수준 관문   같은 하루를 SPLIT 끄고 한 번 더. 중앙이 ±1% 안이어야 한다.
#   다    본런       v5offsite + SPLIT=1 + 얼린 KEEP_MEAN · N=500 · 9일
set -uo pipefail
cd /data/repo
source /data/venv/bin/activate
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass
export PYTHONIOENCODING=utf-8 PYTHONPATH=/data/repo LLM_BASE_URL=http://localhost:8000/v1
# 라운드2 와 같은 환경이어야 한다 — 하나라도 다르면 비교가 섞인다.
export EXP_SANGSAENG_BASE_RATIO=0.268 EXP_SEED_SANGSAENG=1 EXP_BALANCE_DAYS=39
export EXP_DURABLES=1 EXP_CATLINE=fold EXP_POLICY_ANONYMOUS=1 POLICY_POI_SORT_BOOST=0
unset EXP_DAILY_INCOME

KEEP_MEAN="${1:?KEEP_MEAN 을 인자로 줘야 한다 — 교정 탐침에서 잰 값}"
OUT=/data/split_p012; mkdir -p $OUT
LOG=$OUT/run.log
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $LOG; }

D0=2021-10-17; ST=2021-10-18; DY=9; N=500
OFF=2021-10-21:2021-10-22
ON=2021-10-25:2021-10-26

say "KEEP_MEAN=$KEEP_MEAN (교정 탐침에서 재서 얼린 값)"

# --------------------------------------------------------------- 관문
gate () {
  local TAG=$1 SPLIT=$2
  say "--- 관문 [$TAG] SPLIT=$SPLIT · v5 · 하루 · 100명"
  export SIM_PROMPT_VARIANT=v5 SIM_OUTPUT_DIR=$OUT/$TAG EXP_SPLIT_ANCHOR=$SPLIT
  export EXP_KEEP_MEAN=$KEEP_MEAN
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

srcs = {d.get('cm_online_share_source') for d in B}
print('  켠 쪽 online_share_source:', srcs)
if srcs != {'split_base'}:
    print('  **관문 실패** — 분기가 안 걸렸거나 online_share 가 섞여 들어왔다'); sys.exit(3)

def med(rs, k):
    v = [d[k] for d in rs if isinstance(d.get(k), (int, float))]
    return st.median(v) if v else None

for k in ('cm_today_total', 'cm_today_total_incl_online'):
    a, b = med(A, k), med(B, k)
    if a is None or b is None:
        print('  **관문 실패** — %s 가 비었다' % k); sys.exit(3)
    d = (b - a) / a * 100
    ok = abs(d) <= 1.0
    print('  %-28s 끄고 %9s · 켜고 %9s · 차이 %+.2f%%  %s'
          % (k, '{:,}'.format(int(a)), '{:,}'.format(int(b)), d, '통과' if ok else '**실패**'))
    if not ok:
        sys.exit(3)
print('  관문 통과 — 수준을 안 건드렸다')
PYEOF
  [ ${PIPESTATUS[0]:-1} -ne 0 ] && { say "관문에서 멈춘다 — 본런을 돌리지 않는다"; exit 3; }
  touch "$OUT/gate_done"
fi

# --------------------------------------------------------------- 본런
TAG=split_v5offsite
if [ -f "$OUT/score_$TAG.json" ]; then say "[$TAG] 끝남 — 건너뜀"; exit 0; fi
say "=========== [$TAG] v5offsite · SPLIT=1 · KEEP_MEAN=$KEEP_MEAN · N=$N · $ST 부터 $DY 일 ==========="
export SIM_PROMPT_VARIANT=v5offsite SIM_OUTPUT_DIR=$OUT/$TAG
export EXP_SPLIT_ANCHOR=1 EXP_KEEP_MEAN=$KEEP_MEAN
rm -rf "$OUT/$TAG"; mkdir -p "$OUT/$TAG"
python -c "
import sys; sys.path.insert(0,'scripts/sim')
import prompts, hashlib
p = prompts.get('v5offsite').SYSTEM_PROMPT
print('  프롬프트 v5offsite · %d자 · sha256 %s' % (len(p), hashlib.sha256(p.encode()).hexdigest()[:16]))
for w in ('캐시백','문턱','쿠폰','바우처','적립'):
    if w in p: print('    주의: %s 가 본문에 있다 (%d회)' % (w, p.count(w)))
" | tee -a $LOG
python scripts/neo4j_load/97_reset_run_artifacts.py > /dev/null 2>&1
DAY_ZERO=$D0 python scripts/neo4j_load/08_initial_state.py > /dev/null 2>&1
python -u scripts/sim/run_simulation.py --start $ST --days $DY --limit $N \
    --workers 48 --environment covid_2021 2>&1 | stdbuf -oL grep -E "^  Day|TOTAL" | tee -a $LOG

say "[$TAG] 채점 — 그래프가 비워지기 전에 지금 한다"
python scripts/sim/score_policy.py --policy P012 --off $OFF --on $ON \
    --label "$TAG" --json-out "$OUT/score_$TAG.json" --per-agent 2>&1 | tail -24 | tee -a $LOG
say "=== SPLIT_P012_DONE ==="

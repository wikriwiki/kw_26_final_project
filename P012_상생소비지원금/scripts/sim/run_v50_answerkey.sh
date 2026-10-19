#!/usr/bin/env bash
# v50 — v45 를 정답지에 대고 처음 잰다. 거리두기 12일 창, 200명, v5 대 v45.
#
# 관문 라운드 다섯을 돌려 v45 를 만들었는데 정답지에 대고 잰 적이 없다. 관문은
# 전제이고 정답지 수렴이 목표다. stage3 의 -7.0% 와 비교하지 않는다 — 서버·모델이
# 그때와 같다는 보장이 없으므로 v5 를 같은 조건에서 다시 잰다.
set -uo pipefail
cd /data/repo
source /data/venv/bin/activate
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass
export PYTHONIOENCODING=utf-8 PYTHONPATH=/data/repo LLM_BASE_URL=http://localhost:8000/v1
export EXP_SANGSAENG_BASE_RATIO=0.268 EXP_SEED_SANGSAENG=1 EXP_BALANCE_DAYS=39
export EXP_DURABLES=1 EXP_CATLINE=fold EXP_POLICY_ANONYMOUS=1 POLICY_POI_SORT_BOOST=0
# 12일은 지갑 한계(중앙 25.9일) 안이다. v30 과 섞지 않기 위해 소득은 켜지 않는다.
unset EXP_DAILY_INCOME

OUT=/data/v50_answerkey; mkdir -p $OUT
LOG=$OUT/v50.log
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $LOG; }

D0=2020-11-13; ST=2020-11-14; DY=12; N=200
OFF=2020-11-17:2020-11-18
ON=2020-11-24:2020-11-25

run_one () {
  local TAG=$1 VAR=$2
  if [ -f "$OUT/score_$TAG.json" ]; then say "[$TAG] 끝남 — 건너뜀"; return 0; fi
  say "=========== [$TAG] 프롬프트 $VAR · N=$N · $ST 부터 $DY 일 ==========="
  export SIM_PROMPT_VARIANT=$VAR SIM_OUTPUT_DIR=$OUT/$TAG
  rm -rf "$OUT/$TAG"; mkdir -p "$OUT/$TAG"
  python -c "
import sys; sys.path.insert(0,'scripts/sim')
import prompts, hashlib
p = prompts.get('$VAR').SYSTEM_PROMPT
print('  프롬프트 %s · %d자 · sha256 %s' % ('$VAR', len(p), hashlib.sha256(p.encode()).hexdigest()[:16]))
for w in ('캐시백','문턱','쿠폰','바우처'):
    if w in p: print('    주의: %s 가 본문에 있다 (%d회)' % (w, p.count(w)))
" | tee -a $LOG
  python scripts/neo4j_load/97_reset_run_artifacts.py > /dev/null 2>&1
  DAY_ZERO=$D0 python scripts/neo4j_load/08_initial_state.py > /dev/null 2>&1
  python -u scripts/sim/run_simulation.py --start $ST --days $DY --limit $N \
      --workers 48 --environment covid_2021 2>&1 | stdbuf -oL grep -E "^  Day|TOTAL" | tee -a $LOG
  say "[$TAG] 채점 — 그래프가 비워지기 전에 지금 한다"
  python scripts/sim/score_policy.py --policy DISTANCING_2020 --off $OFF --on $ON \
      --label "$TAG" --json-out "$OUT/score_$TAG.json" 2>&1 | tail -18 | tee -a $LOG
}

run_one v50_v5  v5  || say "v5 런 경고"
run_one v50_v45 v45 || say "v45 런 경고"

say "=========== 정답지와의 거리 ==========="
python - <<'PYEOF' | tee -a $LOG
import json, os
out = '/data/v50_answerkey'
MEASURED = {'DS-1': -14.1, 'DS-2': 4.2}
def rows(tag):
    p = os.path.join(out, 'score_%s.json' % tag)
    if not os.path.exists(p): return {}
    d = json.load(open(p, encoding='utf-8'))
    return {r['id']: r for r in (d.get('results') or [])}
A, B = rows('v50_v5'), rows('v50_v45')
if not A or not B:
    print('채점 결과가 아직 없다'); raise SystemExit
def pct(r):
    return None if not r or not r.get('base') else r['mean']/r['base']*100.0
print('%-6s %12s %12s %10s %10s' % ('지표','v5','v45','실측','더 가까운 쪽'))
for k in ('DS-1','DS-2','DS-4'):
    x, y = pct(A.get(k)), pct(B.get(k))
    m = MEASURED.get(k)
    if x is None or y is None:
        print('%-6s %12s %12s' % (k, x, y)); continue
    if m is None:
        print('%-6s %+11.1f%% %+11.1f%% %10s %10s' % (k, x, y, '-', '-'))
        continue
    da, db = abs(x-m), abs(y-m)
    print('%-6s %+11.1f%% %+11.1f%% %+9.1f%% %10s  (|차| %.1f 대 %.1f)'
          % (k, x, y, m, 'v45' if db < da else ('v5' if da < db else '같다'), da, db))
print()
for k in ('DS-3',):
    ra, rb = A.get(k), B.get(k)
    print('%-6s v5 hit=%s · v45 hit=%s' % (k, ra and ra.get('hit'), rb and rb.get('hit')))
print()
print('사전등록 합격선: v45 의 DS-1·DS-2 가 둘 다 실측에 더 가깝다')
print('예측했던 것: 조금 가깝거나 비슷하다. 구간을 좁게 적지 않았다.')
print('규칙: 이 결과를 보고 프롬프트를 고치지 않는다. 일회 평가다.')
PYEOF
say "=== V50_DONE ==="

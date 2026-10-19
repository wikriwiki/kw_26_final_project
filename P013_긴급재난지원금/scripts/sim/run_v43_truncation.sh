#!/usr/bin/env bash
# v43 — 잘린 78자를 돌려주면 P012 가 달라지는가. 프롬프트는 v5 그대로, 한 변수.
#
# P012 본문 358자가 런타임 정책 블록에서 280자에 잘린다. 끊기는 자리가 문장
# 한가운데이고, 사라지는 것이 "3% 문턱을 못 넘기면 이번 달 혜택은 사라집니다" 다.
# 기전은 앞 280자에 남으므로 사라지는 것은 비대칭 유인뿐이다.
#
# result_FINAL 의 +11.5%p 와 비교하지 않는다. 서버·모델이 그때와 같다는 보장이
# 없으므로 절단 있는 판(A)을 같은 런에서 다시 잰다.
set -uo pipefail
cd /data/repo
source /data/venv/bin/activate
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass
export PYTHONIOENCODING=utf-8 PYTHONPATH=/data/repo LLM_BASE_URL=http://localhost:8000/v1
export EXP_SANGSAENG_BASE_RATIO=0.268 EXP_SEED_SANGSAENG=1 EXP_BALANCE_DAYS=39
export EXP_DURABLES=1 EXP_CATLINE=fold EXP_POLICY_ANONYMOUS=1 POLICY_POI_SORT_BOOST=0
# 9일은 지갑 한계(중앙 25.9일) 안이다. 소득은 켜지 않는다 — v30 과 섞지 않기 위해서다.
unset EXP_DAILY_INCOME

OUT=/data/v43_truncation; mkdir -p $OUT
LOG=$OUT/v43.log
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $LOG; }

D0=2021-10-18; ST=2021-10-19; DY=9; N=200
OFF=2021-10-21:2021-10-22
ON=2021-10-25:2021-10-26
SRC=scripts/sim/dawn_context.py

# 절단 스위치. 원본을 건드리기 전에 반드시 백업하고, 어떻게 끝나든 되돌린다.
# 소스를 고쳐 가며 도는 러너이므로 중간에 죽으면 저장소가 수정된 채 남는다.
[ -f $SRC.v43_orig ] || cp $SRC $SRC.v43_orig
restore_source () { cp $SRC.v43_orig $SRC; echo "[$(date +%H:%M:%S)] 원본 복구됨" >> $LOG; }
trap restore_source EXIT INT TERM
set_truncation () {   # on = 지금 그대로, off = 고침
  if [ "$1" = "on" ]; then
    cp $SRC.v43_orig $SRC
  else
    python - <<'PY'
import io
p = 'scripts/sim/dawn_context.py'
s = io.open(p, encoding='utf-8').read()
h = s.index('def _format_policy_facts'); t = s.index('\ndef ', h + 10)
body = s[h:t]
fixed = body.replace('.split())[:280]', '.split())')
assert fixed != body, '절단을 찾지 못했다'
io.open(p, 'w', encoding='utf-8', newline='\n').write(s[:h] + fixed + s[t:])
print('절단 제거')
PY
  fi
  python - <<'PY'
import io
s = io.open('scripts/sim/dawn_context.py', encoding='utf-8').read()
h = s.index('def _format_policy_facts'); t = s.index('\ndef ', h + 10)
print('  지금 상태: 고정길이 절단', '있음' if '[:280]' in s[h:t] else '없음')
PY
}

run_one () {
  local TAG=$1 TRUNC=$2
  if [ -f "$OUT/score_$TAG.json" ]; then say "[$TAG] 끝남 — 건너뜀"; return 0; fi
  say "=========== [$TAG] 절단 $TRUNC · N=$N · $ST 부터 $DY 일 ==========="
  set_truncation "$TRUNC" | tee -a $LOG
  export SIM_PROMPT_VARIANT=v5 SIM_OUTPUT_DIR=$OUT/$TAG
  rm -rf "$OUT/$TAG"; mkdir -p "$OUT/$TAG"
  python scripts/neo4j_load/97_reset_run_artifacts.py > /dev/null 2>&1
  python scripts/neo4j_load/10_load_grant_policy.py \
    data/neo4j_load/policies/P012.json > /dev/null 2>&1 || { say "[$TAG] 정책 적재 실패"; return 1; }
  DAY_ZERO=$D0 python scripts/neo4j_load/08_initial_state.py > /dev/null 2>&1
  python -u scripts/sim/run_simulation.py --start $ST --days $DY --limit $N \
      --workers 48 --environment covid_2021 2>&1 | stdbuf -oL grep -E "^  Day|TOTAL" | tee -a $LOG
  say "[$TAG] 채점"
  python scripts/sim/score_policy.py --policy P012 --off $OFF --on $ON \
      --label "$TAG" --json-out "$OUT/score_$TAG.json" 2>&1 | tail -16 | tee -a $LOG
}

run_one v43_A_truncated on  || say "A 경고"
run_one v43_B_full      off || say "B 경고"

say "원본 복구 (trap 이 한 번 더 한다)"
restore_source

say "=========== 두 판의 차이 ==========="
python - <<'PY' | tee -a $LOG
import json, os
out = '/data/v43_truncation'
def row(tag, key='P012-1'):
    p = os.path.join(out, 'score_%s.json' % tag)
    if not os.path.exists(p): return None
    d = json.load(open(p, encoding='utf-8'))
    for r in d.get('results') or []:
        if r.get('id') == key: return r
    return None
def pct(r):
    return None if not r or not r.get('base') else r['mean']/r['base']*100.0
for key in ('P012-1', 'P012-2'):
    a, b = row('v43_A_truncated', key), row('v43_B_full', key)
    x, y = pct(a), pct(b)
    if x is None or y is None:
        print('%-8s 읽지 못함' % key); continue
    print('%-8s A(절단) %+.1f%%  ·  B(전체) %+.1f%%  ·  차이 %+.1f%%p' % (key, x, y, y - x))
    print('         A CI %s (n=%d) · B CI %s (n=%d)'
          % (a.get('ci'), a.get('n') or 0, b.get('ci'), b.get('n') or 0))
print()
print('합격선: B 의 P012-1 이 A 보다 높고 B 의 CI 가 0 을 제외한다')
print('        P012-2 는 양쪽 모두 움직이지 않아야 한다')
print('예측했던 것: B > A. 틀렸으면 그대로 적는다.')
PY
say "=== V43_DONE ==="

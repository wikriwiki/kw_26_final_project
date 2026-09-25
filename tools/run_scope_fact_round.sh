#!/usr/bin/env bash
# 범위의 산술 한 줄 — 전체 라운드. 사전등록 experiments/scope_fact/prereg.md
#
# **탐침이 changed>0 을 내야만 돌린다.** changed=0 이면 그 줄은 무력하고,
# 무력한 줄을 들고 12일 × 200명 × 두 팔을 태우지 않는다. 이 스크립트가
# 탐침 판정을 직접 읽어 확인한다 — 사람이 기억하는 것에 의존하지 않는다.
#
# ## 창을 원래 설계로 되돌린다
#
# 지금 채점표의 위약 창(10-21:22 → 10-25:26)은 **양쪽이 모두 P090 발효 기간
# (10-11~10-31) 안**이라 대조가 성립하지 않는다. 즉시 할인이라 기간 안에서
# 쌓이는 것이 없으므로 구조적으로 0 이 나온다. 정책 파일 notes 의 원래 설계로
# 되돌리되 **요일을 맞춘다** — 무정책 10-07:08(목금) → 정책 10-14:15(목금).
# 원 설계(정책 10-11:12)는 월·화라 요일이 어긋났다. 위약의 주 지표가 업종
# 구성이라 요일 어긋남이 그대로 지표에 얹힌다(experiments/scope_fact/amend_window.md).
#
# ## 팔 둘은 그 한 줄만 다르다
#
#   scope_off  EXP_SCOPE_FACT=0
#   scope_on   EXP_SCOPE_FACT=1
#
# 프롬프트도 에이전트도 창도 같다. 그래서 차이가 생기면 그 줄 때문이다.
set -uo pipefail
cd /data/repo
source /data/venv/bin/activate
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass
export PYTHONIOENCODING=utf-8 PYTHONPATH=/data/repo LLM_BASE_URL=http://localhost:8000/v1
export EXP_SANGSAENG_BASE_RATIO=0.268 EXP_SEED_SANGSAENG=1 EXP_BALANCE_DAYS=39
export EXP_DURABLES=1 EXP_CATLINE=fold EXP_POLICY_ANONYMOUS=1 POLICY_POI_SORT_BOOST=0
unset EXP_DAILY_INCOME

OUT=/data/scope_round; mkdir -p $OUT
LOG=$OUT/round.log
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $LOG; }

VAR=${SIM_PROMPT_VARIANT:-v5}
D0=2021-10-03; ST=2021-10-04; DY=12; N=200
OFF=2021-10-07:2021-10-08
ON=2021-10-14:2021-10-15

# ------------------------------------------------- ① 탐침이 통과했는가 (관문)
say "탐침 판정을 확인한다 — changed=0 이면 돌리지 않는다"
python - <<'PY_GATE'
import json, os, sys
p = '/data/scope_probe/responses.jsonl'
if not os.path.exists(p):
    sys.exit('탐침 응답이 없다 — 먼저 run_scope_fact_probe.sh 를 돌릴 것')
sys.path.insert(0, '/data/validation_v3/repo/scripts/sim')
import importlib.util
spec = importlib.util.spec_from_file_location(
    'probe', '/data/validation_v3/repo/scripts/sim/scope_fact_probe.py')
pr = importlib.util.module_from_spec(spec); spec.loader.exec_module(pr)
rows = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
tot = 0
for seed in sorted({r.get('seed') for r in rows if r.get('seed') is not None}):
    s = pr.compare([r for r in rows if r.get('seed') == seed])
    print('  시드 %-5s 쌍 %d · 달라진 시민 %d' % (seed, s['paired'], s['changed']))
    tot += s['changed']
if tot == 0:
    sys.exit('**탐침에서 계획이 하나도 안 달라졌다 — 등록된 중단 규칙대로 돌리지 않는다.**')
print('  통과 — 달라진 시민 합계 %d' % tot)
PY_GATE
if [ $? -ne 0 ]; then say "관문 불통과 — 여기서 멈춘다"; exit 1; fi

# ------------------------------------------------------------ ② 그래프 준비
say "그래프를 비우고 P090(위약) 하나만 넣는다"
cat > $OUT/_clear.py <<'PY_CLEAR'
import sys
sys.path.insert(0, "/data/repo/scripts")
from neo4j_load._common import driver_session
with driver_session() as s:
    s.run("MATCH (p:Policy) DETACH DELETE p")
    print("  비운 뒤 :Policy %d개" % s.run("MATCH (p:Policy) RETURN count(p) AS n").single()["n"])
PY_CLEAR
python $OUT/_clear.py 2>&1 | tee -a $LOG
python scripts/neo4j_load/10_load_grant_policy.py data/neo4j_load/policies/P090.json 2>&1 | tee -a $LOG

run_one () {
  local TAG=$1 FLAG=$2
  if [ -f "$OUT/score_$TAG.json" ]; then say "[$TAG] 끝남 — 건너뜀"; return 0; fi
  say "=========== [$TAG] EXP_SCOPE_FACT=$FLAG · $VAR · N=$N · $ST 부터 $DY 일 ==========="
  export SIM_PROMPT_VARIANT=$VAR SIM_OUTPUT_DIR=$OUT/$TAG EXP_SCOPE_FACT=$FLAG
  rm -rf "$OUT/$TAG"; mkdir -p "$OUT/$TAG"
  # 그 줄이 실제로 붙는지(또는 안 붙는지) 런 전에 눈으로 확인한다
  python -c "
import os, sys, json; sys.path.insert(0,'scripts/sim')
from mechanisms import sector_voucher
row = json.load(open('data/neo4j_load/policies/P090.json', encoding='utf-8'))
line = sector_voucher.status('P090', row, {}, {})
on = '혜택은 위 업종에서만' in line
print('  EXP_SCOPE_FACT=%s · 그 줄 %s' % (os.environ.get('EXP_SCOPE_FACT'), '붙음' if on else '없음'))
assert on == (os.environ.get('EXP_SCOPE_FACT') == '1'), '플래그와 렌더가 어긋난다'
" | tee -a $LOG
  # !! 2026-09-25 고침 — 97_reset 이 **Policy 도 지운다**(그 파일 47행).
  # 위에서 적재한 P090 이 여기서 지워진 채 런이 돌았다. 오류도 안 난다.
  # 같은 버그로 p013_ruler 와 p012_28d 첫 시도를 잃었다.
  # experiments/plan_channel/P013_evidence_is_weaker.md
  python scripts/neo4j_load/97_reset_run_artifacts.py > /dev/null 2>&1
  python scripts/neo4j_load/10_load_grant_policy.py \
      data/neo4j_load/policies/P090.json 2>&1 | tail -2 | tee -a $LOG
  python - <<'PYEOF' 2>&1 | tee -a $LOG
import sys
sys.path.insert(0, "/data/repo/scripts")
from neo4j_load._common import driver_session
with driver_session() as s:
    rows = list(s.run("MATCH (p:Policy) RETURN p.id AS id, "
                      "count{(p)-[:applied_to]->()} AS na"))
for r in rows:
    print("  정책 %s applied_to=%d개 지역" % (r["id"], r["na"]))
if not rows or any(r["na"] == 0 for r in rows):
    sys.exit("  ** 거부: 정책이 없거나 어느 지역에도 안 걸렸다")
PYEOF
  [ ${PIPESTATUS[0]:-1} -ne 0 ] && { say "[$TAG] 정책 관문 실패 — 그만둔다"; return 1; }
  DAY_ZERO=$D0 python scripts/neo4j_load/08_initial_state.py > /dev/null 2>&1
  python -u scripts/sim/run_simulation.py --start $ST --days $DY --limit $N \
      --workers 48 --environment covid_2021 2>&1 | stdbuf -oL grep -E "^  Day|TOTAL" | tee -a $LOG
  say "[$TAG] 채점 — 그래프가 비워지기 전에 지금 한다"
  python scripts/sim/score_policy.py --policy PLACEBO_FAKE --off $OFF --on $ON \
      --elig-policy data/neo4j_load/policies/P090.json --per-agent \
      --label "$TAG" --json-out "$OUT/score_$TAG.json" 2>&1 | tail -18 | tee -a $LOG
}

run_one scope_off 0 || say "off 팔 경고"
run_one scope_on  1 || say "on 팔 경고"

# ------------------------------------------------------------ ③ 등록된 판정
say "=========== 등록된 판정 ==========="
cat > $OUT/_verdict.py <<'PY_VERDICT'
import json, os
OUT = '/data/scope_round'


def rows(tag):
    p = os.path.join(OUT, 'score_%s.json' % tag)
    if not os.path.exists(p):
        return {}
    return {r['id']: r for r in (json.load(open(p, encoding='utf-8')).get('results') or [])}


A, B = rows('scope_off'), rows('scope_on')
if not (A and B):
    print('채점 결과가 아직 없다'); raise SystemExit


def pct(r):
    if not r or not r.get('base'):
        return None
    return r['mean'] / r['base'] * 100.0


print('주 지표  PL-2 — 대상 아닌 업종. |평균| 이 0 에 가까워지는가')
x, y = pct(A.get('PL-2')), pct(B.get('PL-2'))
if x is None or y is None:
    print('   값 없음  off=%s on=%s' % (x, y))
else:
    print('   off %+.1f%%  →  on %+.1f%%   |차| %.1f → %.1f  %s'
          % (x, y, abs(x), abs(y), '가까워짐' if abs(y) < abs(x) else '멀어짐'))
print()
print('부 지표  PL-1 — 대상 업종. **잃으면 안 된다**')
for tag, R in (('off', A), ('on', B)):
    r = R.get('PL-1')
    print('   %-4s %s · hit=%s' % (tag, ('%+.1f%%' % pct(r)) if pct(r) is not None else '—',
                                   r and r.get('hit')))
print()
print('런 간 이동 — **이 창에서는 아직 재 본 적이 없다.**')
print('result_2026_09_15 는 정책 창이 10-11:12 였고 여기는 10-14:15 다(요일을 맞췄다).')
print('창이 다르므로 그 런을 런 간 이동의 짝으로 쓰지 않는다. 그래서 이 라운드는')
print('**팔 사이 차이의 부호**까지만 읽고, 크기를 프롬프트의 성질로 적지 않는다.')
print()
print('규칙: 이 결과를 보고 그 줄의 문구를 고치지 않는다.')
print('      탐침은 Stage1 계획만 봤다. 금액 제로섬은 이 PL-2 로만 알 수 있다.')
PY_VERDICT
python $OUT/_verdict.py | tee -a $LOG
say "=== SCOPE_ROUND_DONE ==="

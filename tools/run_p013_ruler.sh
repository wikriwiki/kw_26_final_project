#!/usr/bin/env bash
# P013 자 만들기 — 같은 프롬프트(v5)를 두 팔로 돌려 **런 간 이동**을 처음 잰다.
#
# 사전등록: experiments/p013_ruler/prereg.md
#   후보를 비교하지 않는다. 팔 둘이 같은 프롬프트다.
#   두 팔의 차이는 다음 라운드의 판정 문턱으로만 쓴다.
#
# 왜 n=700 인가
#   EM-2 필요 280 · EM-3 필요 620 (지금 200). 둘 다 넘긴다.
#
# 창은 기존 stage3 런과 같다 — 그 n=200 런이 세 번째 관측점이 된다.
set -uo pipefail
cd /data/repo
source /data/venv/bin/activate
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass
export PYTHONIOENCODING=utf-8 PYTHONPATH=/data/repo LLM_BASE_URL=http://localhost:8000/v1
export EXP_SANGSAENG_BASE_RATIO=0.268 EXP_SEED_SANGSAENG=1 EXP_BALANCE_DAYS=39
export EXP_DURABLES=1 EXP_CATLINE=fold EXP_POLICY_ANONYMOUS=1 POLICY_POI_SORT_BOOST=0
unset EXP_DAILY_INCOME      # 12일은 지갑 한계 안이고, P013 이 05-13 에 지갑을 채운다

OUT=/data/p013_ruler; mkdir -p $OUT
LOG=$OUT/p013.log
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $LOG; }

D0=2020-05-03; ST=2020-05-04; DY=12; N=700
OFF=2020-05-07:2020-05-08
ON=2020-05-14:2020-05-15

# ------------------------------------------------------------ 그래프에 P013 하나만
say "그래프를 비우고 P013 하나만 넣는다"
cat > $OUT/_clear.py <<'PY_CLEAR'
import sys
sys.path.insert(0, "/data/repo/scripts")
from neo4j_load._common import driver_session
with driver_session() as s:
    s.run("MATCH (p:Policy) DETACH DELETE p")
    print("  비운 뒤 :Policy %d개" % s.run("MATCH (p:Policy) RETURN count(p) AS n").single()["n"])
PY_CLEAR
python $OUT/_clear.py 2>&1 | tee -a $LOG
python scripts/neo4j_load/10_load_grant_policy.py data/neo4j_load/policies/P013.json 2>&1 | tee -a $LOG

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
" | tee -a $LOG
  # !! 2026-09-25 고침 — 순서가 뒤집혀 있었다 !!
  #
  # `97_reset_run_artifacts.py` 는 State·Plan 만 지우는 게 아니라 **Policy 도 지운다**
  # (그 파일 47행). 위에서 적재한 P013 이 여기서 지워진 채 런이 돌았다. 오류도 안 나고
  # 로그에는 "적재 완료" 가 찍힌다. 그래서 **12일 × 700명이 정책 없이 돌았고**,
  # 지원금이 한 푼도 안 나갔으며 "정책 반응 +6.36%" 는 예열 드리프트였다.
  # experiments/plan_channel/P013_evidence_is_weaker.md
  #
  # reset 은 팔마다 돌아야 하므로(State 를 비워야 한다) **정책을 그 뒤에 다시 넣는다.**
  python scripts/neo4j_load/97_reset_run_artifacts.py > /dev/null 2>&1
  python scripts/neo4j_load/10_load_grant_policy.py \
      data/neo4j_load/policies/P013.json 2>&1 | tail -2 | tee -a $LOG
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
  python scripts/sim/score_policy.py --policy EMERGENCY_2020 --off $OFF --on $ON \
      --elig-policy data/neo4j_load/policies/P013.json --per-agent \
      --label "$TAG" --json-out "$OUT/score_$TAG.json" 2>&1 | tail -18 | tee -a $LOG
}

# **두 팔 모두 v5 다.** 바꾸는 것이 없다 — 그것이 이 라운드의 전부다.
run_one ruler_a v5 || say "A 팔 경고"
run_one ruler_b v5 || say "B 팔 경고"

# ------------------------------------------------------------------ 두 팔의 차이
say "=========== 같은 프롬프트 두 팔이 얼마나 움직였나 ==========="
cat > $OUT/_shift.py <<'PY_SHIFT'
import json, os
OUT = '/data/p013_ruler'


def rows(tag):
    p = os.path.join(OUT, 'score_%s.json' % tag)
    if not os.path.exists(p):
        return {}
    return {r['id']: r for r in (json.load(open(p, encoding='utf-8')).get('results') or [])}


A, B = rows('ruler_a'), rows('ruler_b')
if not (A and B):
    print('채점 결과가 아직 없다')
    raise SystemExit


def pct(r):
    if not r or not r.get('base'):
        return None
    return r['mean'] / r['base'] * 100.0


print('%-7s %10s %10s %9s  %s' % ('지표', 'A', 'B', '이동', '구간이 0 을 지나는가'))
for k in ('EM-2', 'EM-3'):
    x, y = pct(A.get(k)), pct(B.get(k))
    if x is None or y is None:
        print('%-7s %10s %10s' % (k, x, y))
        continue
    def crosses(r):
        ci = (r or {}).get('ci')
        return '지난다' if (ci and ci[0] <= 0 <= ci[1]) else '안 지난다'
    print('%-7s %+9.1f%% %+9.1f%% %8.1f%%p  A %s · B %s'
          % (k, x, y, abs(y - x), crosses(A.get(k)), crosses(B.get(k))))

for k in ('EM-4',):
    print('%-7s 순위 — A %s · B %s'
          % (k, (A.get(k) or {}).get('hit'), (B.get(k) or {}).get('hit')))

# 에이전트 단위 — 총합만 보면 어디서 왔는지 모른다
for tag in ('ruler_a', 'ruler_b'):
    p = os.path.join(OUT, 'score_%s.agents.jsonl' % tag)
    print('%s 에이전트 기록 %s' % (tag, ('%d행' % sum(1 for _ in open(p, encoding='utf-8')))
                                  if os.path.exists(p) else '없음'))
print()
print('이 수는 **런 간 이동의 아래끝**이다 — 한 스크립트 안에서 연달아 돌렸으므로')
print('실행 사이에 끼어드는 환경 차이가 빠져 있다. 거리두기의 0.5%p·2.7%p 보다')
print('작다면 그 이동은 런의 성질이 아니라 실행 사이의 무엇이고, 고칠 수 있다.')
print()
print('규칙: 두 팔 중 어느 쪽이 정답지에 가까운지 적지 않는다. 같은 프롬프트다.')
PY_SHIFT
python $OUT/_shift.py | tee -a $LOG
say "=== P013_RULER_DONE ==="

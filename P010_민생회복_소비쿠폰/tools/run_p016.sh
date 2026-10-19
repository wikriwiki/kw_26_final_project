#!/usr/bin/env bash
# P016 농축산물 할인쿠폰 — 새 기전이 코드 수정 없이 붙는가, 모델이 새 규칙을 읽는가.
#
# 사전등록  experiments/p016/prereg.md (커밋 69d8af7)
#           experiments/p016/amend_window.md — 창을 고쳤다. 고칠 때 P016 런 0건
#
#   주 지표  C2  대상 POI 증가율 > 전체 마트 증가율 (순위)
#   부 지표  C3  비중 증가 — 방향만 적고 판정에 쓰지 않는다 (n 이 2.5배 모자라다)
#   검출불가  C1  금액 증가 — 필요 n 14,597. 부호가 맞아도 '맞혔다'고 하지 않는다
#   참고     E1  할인비용 1원당 추가 소비 — 2021년 별도 추정이라 채점 안 함
#
# **이 라운드는 프롬프트를 고르지 않는다.** 팔이 하나다. 기전이 붙는지를 본다.
#
# 창이 왜 이 날짜인가
#   사업기간 2020-07-30~11-30 (정답지 p20). 앞 창은 시작 이틀 전, 뒤 창은 닷새 뒤.
#   둘 다 '생활 속 거리두기' 구간이다 — 수도권 2단계는 08-16 부터라 레짐이 안 낀다.
#   요일을 맞춘다: 07-28 화 · 07-29 수 · 08-04 화 · 08-05 수.
#
# 그래프에 P016 하나만 넣는다
#   2020-07-24~08-05 에 겹치는 다른 정책은 P013(긴급재난지원금)뿐인데 지급일이
#   05-13 이라 이 런 안에서는 아무도 받지 못한다. 잔액 0 인 지갑 카드만 붙어
#   잡음이 된다. 그래서 넣지 않는다.
set -uo pipefail
cd /data/repo
source /data/venv/bin/activate
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass
export PYTHONIOENCODING=utf-8 PYTHONPATH=/data/repo LLM_BASE_URL=http://localhost:8000/v1
# 다른 정답지 라운드와 같은 환경으로 둔다 — 한 번에 두 가지를 바꾸지 않는다.
export EXP_SANGSAENG_BASE_RATIO=0.268 EXP_SEED_SANGSAENG=1 EXP_BALANCE_DAYS=39
export EXP_DURABLES=1 EXP_CATLINE=fold EXP_POLICY_ANONYMOUS=1 POLICY_POI_SORT_BOOST=0
unset EXP_DAILY_INCOME     # 13일은 지갑 한계(중앙 25.9일) 안이다

OUT=/data/p016; mkdir -p $OUT
LOG=$OUT/p016.log
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $LOG; }

VAR=${SIM_PROMPT_VARIANT:-v5}      # 거시 라인의 현행 프롬프트
D0=2020-07-23; ST=2020-07-24; DY=13; N=500
OFF=2020-07-28:2020-07-29
ON=2020-08-04:2020-08-05
TAG=p016_$VAR

if [ -f "$OUT/score_$TAG.json" ]; then say "[$TAG] 이미 끝남"; exit 0; fi

say "=========== [$TAG] $VAR · N=$N · $ST 부터 $DY 일 ==========="
export SIM_PROMPT_VARIANT=$VAR SIM_OUTPUT_DIR=$OUT/$TAG
rm -rf "$OUT/$TAG"; mkdir -p "$OUT/$TAG"

python -c "
import sys; sys.path.insert(0,'scripts/sim')
import prompts, hashlib
p = prompts.get('$VAR').SYSTEM_PROMPT
print('  프롬프트 %s · %d자 · sha256 %s' % ('$VAR', len(p), hashlib.sha256(p.encode()).hexdigest()[:16]))
for w in ('6.957','11.6','1.9%p','농축산물','할인쿠폰','늘어난다'):
    if w in p: print('    **주의: %s 가 프롬프트 본문에 있다**' % w)
" | tee -a $LOG

# ---------------------------------------------------------------- 그래프 준비
say "그래프를 비우고 P016 하나만 넣는다"
cat > $OUT/_clear.py <<'PY_CLEAR'
import sys
sys.path.insert(0, "/data/repo/scripts")
from neo4j_load._common import driver_session
with driver_session() as s:
    s.run("MATCH (p:Policy) DETACH DELETE p")
    print("  비운 뒤 :Policy %d개" % s.run("MATCH (p:Policy) RETURN count(p) AS n").single()["n"])
PY_CLEAR
python $OUT/_clear.py 2>&1 | tee -a $LOG
python scripts/neo4j_load/10_load_grant_policy.py data/neo4j_load/policies/P016.json 2>&1 | tee -a $LOG

say "런 준비 — 이전 런 흔적 제거 + 초기 상태"
python scripts/neo4j_load/97_reset_run_artifacts.py > /dev/null 2>&1
DAY_ZERO=$D0 python scripts/neo4j_load/08_initial_state.py > /dev/null 2>&1

# ------------------------------------------------------------------ 런 전 점검
# 이 점검이 없어서 같은 자리가 여드레 동안 틀려 있었다 — 사용처 표시가 안 붙으면
# 에이전트에게는 자격 있는 가게가 하나도 없는 셈이고, 그 런은 프롬프트와 무관하게
# 값을 못 낸다. 돌리기 전에 멈춘다(experiments/MERGE_REVERTED_A_FIX.md).
say "정책이 에이전트에게 실제로 도달하는지 먼저 본다"
cat > $OUT/_precheck.py <<'PY_PRECHECK'
import sys, json
sys.path.insert(0, "/data/repo/scripts")
sys.path.insert(0, "/data/repo/scripts/sim")
from mechanisms import poi_restriction
from neo4j_load._common import driver_session
with driver_session() as s:
    rows = [dict(r["p"]) for r in s.run(
        "MATCH (p:Policy) WHERE date('2020-08-04') >= p.effective_from "
        "AND date('2020-08-04') <= p.effective_until RETURN p")]
ids, spec, mark = poi_restriction(rows, {})
print("  발효 중 %s" % [r.get("id") for r in rows])
print("  사용처 표시 %s · 표시문구 %s" % (sorted(ids), mark))
print("  판정 룰 %s" % (json.dumps(spec, ensure_ascii=False) if spec else "없다"))
if not ids:
    raise SystemExit("표시가 켜지지 않는다 — 모델이 대상 업종을 못 본다")
if not spec:
    raise SystemExit("판정 룰을 못 읽었다 — mech_params 를 확인할 것")
if mark != "[농할]":
    raise SystemExit("표시문구가 %r 이다 — P016 이 선언한 것은 [농할]" % mark)
print("  OK")
PY_PRECHECK
python $OUT/_precheck.py > $OUT/precheck.txt 2>&1
RC=$?
cat $OUT/precheck.txt | tee -a $LOG
if [ $RC -ne 0 ]; then say "런 전 점검 실패 — 돌리지 않는다"; exit 1; fi

# ----------------------------------------------------------------------- 본런
python -u scripts/sim/run_simulation.py --start $ST --days $DY --limit $N \
    --workers 48 --environment covid_2021 2>&1 | stdbuf -oL grep -E "^  Day|TOTAL" | tee -a $LOG

say "[$TAG] 채점 — 그래프가 비워지기 전에 지금 한다"
python scripts/sim/score_policy.py --policy P016 --off $OFF --on $ON \
    --elig-policy data/neo4j_load/policies/P016.json \
    --label "$TAG" --json-out "$OUT/score_$TAG.json" 2>&1 | tail -20 | tee -a $LOG

# ------------------------------------------------------------------ 등록된 판정
say "=========== 등록된 판정 ==========="
cat > $OUT/_verdict.py <<'PY_VERDICT'
import json, glob
f = sorted(glob.glob('/data/p016/score_p016_*.json'))
if not f:
    print('채점 결과가 없다'); raise SystemExit
R = {r['id']: r for r in (json.load(open(f[-1], encoding='utf-8')).get('results') or [])}
MARK = {True: '적중', False: '빗나감', None: '관측부족'}
c2 = R.get('C2')
print('주 지표  C2 (순위) — **이것만으로 판정한다**')
print('   %s' % (c2.get('got') if c2 else '없음'))
print('   판정: %s' % MARK.get(c2 and c2.get('hit')))
print()
print('부 지표  C3 (비중) — 방향만 적는다. 필요 n 이 2.5배 모자라다')
print('   %s' % ((R.get('C3') or {}).get('got') or '없음'))
print()
print('검출불가 C1 (금액) — 부호가 맞아도 맞혔다고 하지 않는다. 필요 n 14,597')
print('   %s' % ((R.get('C1') or {}).get('got') or '없음'))
print()
print('규칙: 이 결과를 보고 P016 전용 프롬프트나 행동 로직을 고치지 않는다.')
PY_VERDICT
python $OUT/_verdict.py | tee -a $LOG
say "=== P016_DONE ==="

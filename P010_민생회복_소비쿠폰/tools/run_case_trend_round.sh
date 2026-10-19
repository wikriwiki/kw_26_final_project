#!/usr/bin/env bash
# 후보 2(확진 수준에 기준 배수) 전체 라운드. 설계 experiments/case_trend/design_note.md
#
# **탐침이 갈려야만 돌린다.** 이 스크립트가 탐침 판정을 직접 읽어 확인한다 —
# 사람이 기억하는 것에 의존하지 않는다.
#
# ## 팔이 셋인 이유 — 반증 조건이 설계에 등록돼 있다
#
#   trend_off   거리두기 · EXP_CASE_TREND=0
#   trend_on    거리두기 · EXP_CASE_TREND=1      ← 그 한 줄만 다르다
#   null_on     **시점 위약** · EXP_CASE_TREND=1
#
# 셋째 팔이 반증이다. 시점 위약은 정책이 없는 자리이고 배수만 붙는다.
# **거기서 소비가 움직이면 그 줄은 상태가 아니라 방향을 주입한 것**이므로,
# 거리두기에서 좋아졌더라도 채택하지 않는다. 설계에 먼저 등록한 조건이다.
#
# ## 비용
#
#   trend_off/on  12일 × 500명 × 2  ≈ 16시간
#   null_on        9일 × 200명 × 1  ≈  3시간
#
# n=500 을 쓰는 이유: 같은 창의 n=500 런에서 DS-1·DS-4 의 구간이 0 을 벗어났다.
# n=200 에서는 네 지표 중 하나만 적중이라 후보 간 차이를 읽을 수 없다.
set -uo pipefail
cd /data/repo
source /data/venv/bin/activate
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass
export PYTHONIOENCODING=utf-8 PYTHONPATH=/data/repo LLM_BASE_URL=http://localhost:8000/v1
export EXP_SANGSAENG_BASE_RATIO=0.268 EXP_SEED_SANGSAENG=1 EXP_BALANCE_DAYS=39
export EXP_DURABLES=1 EXP_CATLINE=fold EXP_POLICY_ANONYMOUS=1 POLICY_POI_SORT_BOOST=0
unset EXP_DAILY_INCOME

OUT=/data/ct_round; mkdir -p $OUT
LOG=$OUT/round.log
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $LOG; }
VAR=${SIM_PROMPT_VARIANT:-v5}

# ------------------------------------------------- ① 탐침이 갈렸는가 (관문)
say "탐침 판정을 확인한다 — 부호 검정이 갈리지 않았으면 돌리지 않는다"
python - <<'PY_GATE'
import json, os, sys
p = '/data/ct_probe/responses.jsonl'
if not os.path.exists(p):
    sys.exit('탐침 응답이 없다 — 먼저 run_case_trend_probe.sh 를 돌릴 것')
sys.path.insert(0, '/data/validation_v3/repo/scripts/sim')
import importlib.util
spec = importlib.util.spec_from_file_location(
    'probe', '/data/validation_v3/repo/scripts/sim/scope_fact_probe.py')
pr = importlib.util.module_from_spec(spec); spec.loader.exec_module(pr)
rows = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
s = pr.compare(rows)
best = 1.0
for k, lbl in (('sign_propensity', '소비성향'), ('sign_events', '전체 이벤트'),
               ('sign_excluded', '제외업종')):
    t = s[k]
    print('  %-10s 기대방향 %d · 반대 %d · 동점 %d · p=%.3f'
          % (lbl, t['hit'], t['miss'], t['tie'], t['p']))
    best = min(best, t['p'])
if best >= 0.05:
    sys.exit('**어느 지표도 동전 던지기와 갈리지 않았다(최소 p=%.3f) — 돌리지 않는다.**' % best)
print('  갈렸다 — 최소 p=%.3f' % best)
PY_GATE
if [ $? -ne 0 ]; then say "관문 불통과 — 여기서 멈춘다"; exit 1; fi

run_one () {
  local TAG=$1 FLAG=$2 POL=$3 D0=$4 ST=$5 DY=$6 N=$7 OFF=$8 ON=$9
  if [ -f "$OUT/score_$TAG.json" ]; then say "[$TAG] 끝남 — 건너뜀"; return 0; fi
  say "=========== [$TAG] EXP_CASE_TREND=$FLAG · $POL · N=$N · $ST 부터 $DY 일 ==========="
  export SIM_PROMPT_VARIANT=$VAR SIM_OUTPUT_DIR=$OUT/$TAG EXP_CASE_TREND=$FLAG
  rm -rf "$OUT/$TAG"; mkdir -p "$OUT/$TAG"
  # 그 줄이 실제로 붙는지(또는 안 붙는지) 런 전에 확인한다
  python -c "
import os, sys; sys.path.insert(0,'scripts/sim')
from datetime import date
from environments import build_environment
d = date.fromisoformat('${ON%%:*}')
f = build_environment('covid_2021', d).get('facts') or []
on = any('2주 전' in x for x in f)
print('  EXP_CASE_TREND=%s · 그 줄 %s' % (os.environ.get('EXP_CASE_TREND'), '붙음' if on else '없음'))
assert on == (os.environ.get('EXP_CASE_TREND') == '1'), '플래그와 렌더가 어긋난다'
" | tee -a $LOG
  python scripts/neo4j_load/97_reset_run_artifacts.py > /dev/null 2>&1
  DAY_ZERO=$D0 python scripts/neo4j_load/08_initial_state.py > /dev/null 2>&1
  python -u scripts/sim/run_simulation.py --start $ST --days $DY --limit $N \
      --workers 48 --environment covid_2021 2>&1 | stdbuf -oL grep -E "^  Day|TOTAL" | tee -a $LOG
  say "[$TAG] 채점"
  python scripts/sim/score_policy.py --policy $POL --off $OFF --on $ON \
      --per-agent --label "$TAG" --json-out "$OUT/score_$TAG.json" 2>&1 | tail -18 | tee -a $LOG
}

# 거리두기 — stage5 와 같은 창·같은 n
run_one trend_off 0 DISTANCING_2020 2020-11-13 2020-11-14 12 500 \
        2020-11-17:2020-11-18 2020-11-24:2020-11-25 || say "off 경고"
run_one trend_on  1 DISTANCING_2020 2020-11-13 2020-11-14 12 500 \
        2020-11-17:2020-11-18 2020-11-24:2020-11-25 || say "on 경고"
# 반증 — 시점 위약에 배수만 붙인다
run_one null_on   1 PLACEBO_TIMING  2021-10-17 2021-10-18  9 200 \
        2021-10-21:2021-10-22 2021-10-25:2021-10-26 || say "위약 경고"

say "=========== 등록된 판정 ==========="
cat > $OUT/_verdict.py <<'PY_VERDICT'
import json, os
OUT = '/data/ct_round'
MEASURED = {'DS-1': -14.1, 'DS-2': 4.2}


def rows(tag):
    p = os.path.join(OUT, 'score_%s.json' % tag)
    if not os.path.exists(p):
        return {}
    return {r['id']: r for r in (json.load(open(p, encoding='utf-8')).get('results') or [])}


def pct(r):
    return None if not r or not r.get('base') else r['mean'] / r['base'] * 100.0


A, B, C = rows('trend_off'), rows('trend_on'), rows('null_on')
if not (A and B):
    print('채점 결과가 아직 없다'); raise SystemExit

print('**반증부터 본다** — 시점 위약에서 움직이면 그 줄은 상태가 아니라 방향이다')
bad = []
for k in ('PT-1', 'PT-2', 'PT-3'):
    r = C.get(k)
    print('   %-6s hit=%s  %s' % (k, r and r.get('hit'),
                                  ('%+.1f%%' % pct(r)) if pct(r) is not None else '—'))
    if r and r.get('hit') is False:
        bad.append(k)
if bad:
    print('   **위약이 %s 에서 깨졌다 — 거리두기에서 좋아졌더라도 채택하지 않는다.**' % ', '.join(bad))
print()
print('주 지표  DS-1 — 실측 -14.1% 에 가까워지는가')
x, y = pct(A.get('DS-1')), pct(B.get('DS-1'))
if x is None or y is None:
    print('   값 없음')
else:
    print('   off %+.1f%% → on %+.1f%%   |차| %.1f → %.1f   이동 %.1f%%p'
          % (x, y, abs(x + 14.1), abs(y + 14.1), abs(y - x)))
    print('   런 간 이동 0.5%p 를 넘어야 읽는다 → %s'
          % ('읽는다' if abs(y - x) > 0.5 else '**런으로도 설명된다**'))
print()
print('부 지표  DS-3 순위 — **잃으면 안 된다**')
for tag, R in (('off', A), ('on', B)):
    print('   %-4s hit=%s' % (tag, (R.get('DS-3') or {}).get('hit')))
print()
print('DS-2 는 읽지 않는다 — 런 간 이동 13.1·2.7%p, 검출에 약 10,000명 필요')
print('규칙: 이 결과를 보고 그 줄의 문구를 고치지 않는다.')
PY_VERDICT
python $OUT/_verdict.py | tee -a $LOG
say "=== CT_ROUND_DONE ==="

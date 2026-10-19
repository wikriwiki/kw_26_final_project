#!/usr/bin/env bash
# p012m 을 **있는 그래프 위에서** 이어 돌린다. 복원도 초기화도 하지 않는다.
#
# 왜 전용 파일인가: 공용 resume 은 다른 케이스와 함께 동결돼 있고, 무엇보다
# `EXP_ELIGIBLE_CHANNEL` 과 앵커 비례 소득이 빠져 있다. 그대로 이어 돌리면 회계가
# 런 중간에 바뀐 채 진행된다 — 조용히 다른 실험이 된다.
#
# 이어하기는 원래 런과 **같은 코드·같은 설정**이어야 한다. run_simulation 이 날마다
# cohort 서명(run_id·환경·프롬프트·실행 지문)을 적어 두고, 다르면 거부한다. 그 관문을
# 우회하지 않는다 — 코드가 바뀌었으면 이어 돌리지 말고 새 런으로 가야 한다.
#
#   bash tools/resume_p012m_20260929.sh <attempt> [on|off]
set -Eeuo pipefail
umask 077
attempt=${1:?attempt id (1,2,...)}
arm=${2:-on}
[[ $arm =~ ^(on|off)$ ]]
REPO=/data/pilot_repo_20260927
BASE=/data/multipolicy_v53_20260928/p012m
ARM=$BASE/$arm
cd "$REPO"
source /data/venv/bin/activate
source <(grep '^export NEO4J_URI=' "$REPO/tools/run_p013_ruler.sh" | head -1)
export PYTHONPATH="$REPO" PYTHONIOENCODING=utf-8 LLM_BASE_URL=http://localhost:8000/v1
export SIM_PROMPT_VARIANT=v53 LLM_MODE=exaone_4_5
export EXP_SANGSAENG_BASE_RATIO=0.268 EXP_SEED_SANGSAENG=1 EXP_BALANCE_DAYS=39
export EXP_DURABLES=1 EXP_CATLINE=fold EXP_POLICY_ANONYMOUS=1 POLICY_POI_SORT_BOOST=0
# 본런과 **같은** 회계·소득으로 이어야 한다. 하나라도 빠지면 다른 실험이 된다.
export EXP_ELIGIBLE_CHANNEL=1
export EXP_DAILY_INCOME=anchor:0.41667
unset EXP_DAILY_INCOME_MAP
unset SIM_ALLOW_STAGE2_FALLBACK
export SIM_OUTPUT_DIR="$ARM"

log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$ARM/resume_${attempt}.log"; }
trap 'log "FAILED at line $LINENO; 그래프/출력은 그대로 둔다"' ERR

START=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["start"])' "$ARM/run_manifest.json")
DAYS=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["days"])' "$ARM/run_manifest.json")
N=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["citizens"])' "$ARM/run_manifest.json")
ENV_ID=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment"])' "$ARM/run_manifest.json")
export SIM_ENVIRONMENT="$ENV_ID"
WORKERS=${P012M_WORKERS:-64}

# run_id 는 cohort 서명에 들어간다 — 원래 런이 쓴 것과 **글자까지** 같아야 한다
# (원래 런너는 RUN_REVISION 을 끼워 넣는다). 그래서 기록된 cohort 에서 읽는다.
COHORT="$ARM/cohort_${START}.json"
test -s "$COHORT"
SIM_RUN_ID=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$COHORT")
test -n "$SIM_RUN_ID"
export SIM_RUN_ID

test -s "$ARM/graph_restored.marker"
test -s "$BASE/roster.json"
test ! -e "$ARM/resume_${attempt}.exitcode"
test ! -e "$ARM/graph_backup/SHA256SUMS"
test "$(curl -fsS -m 8 localhost:8000/v1/models | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')" = 'LGAI-EXAONE/EXAONE-4.5-33B-AWQ'
if pgrep -af '[r]un_simulation.py' | grep -q .; then
  log '다른 시뮬이 돌고 있다 — 동시 이어하기를 거부한다'; exit 1
fi
# 코드가 런 때와 같아야 한다 — 다르면 이어 돌리지 않는다(관문을 우회하지 않는다).
if ! sha256sum -c "$ARM/frozen_inputs.sha256" >/dev/null 2>&1; then
  log '런 때와 코드가 다르다 — 이어 돌리지 않는다. frozen_inputs.sha256 을 확인하라'
  sha256sum -c "$ARM/frozen_inputs.sha256" 2>&1 | grep -v ': OK$' | tee -a "$ARM/resume_${attempt}.log" || true
  exit 1
fi

log "이어하기 attempt=$attempt · $arm · $N명 x ${DAYS}일 · workers=$WORKERS · run_id=$SIM_RUN_ID"
for offset in $(seq 0 $((DAYS - 1))); do
  day=$(date -d "$START + $offset days" +%F)
  if [[ -s "$ARM/day_${day}.json" ]]; then
    log "  $day 이미 끝났다 — 건너뛴다"
    continue
  fi
  success=0
  for try in 1 2 3; do
    log "  Day $day try $try"
    if python -u scripts/sim/run_simulation.py --start "$day" --days 1 \
         --roster "$BASE/roster.json" --workers "$WORKERS" --environment "$ENV_ID" \
         > "$ARM/day_${day}_resume${attempt}_try${try}.run.log" 2>&1 && \
       python - "$ARM/summary.json" "$ARM/day_${day}.json" "$day" "$N" <<'PY'
import json, sys
from pathlib import Path
d = json.load(open(sys.argv[1])); day = sys.argv[3]
assert d.get('completed_at') and len(d['summary']) == 1
r = d['summary'][0]
assert r['day'] == day and r['ok'] == int(sys.argv[4]) and r['err'] == 0
Path(sys.argv[2]).write_text(json.dumps(r, ensure_ascii=False, indent=2) + '\n')
PY
    then success=1; break; fi
    log "    실패 — 마지막 줄: $(tail -1 "$ARM/day_${day}_resume${attempt}_try${try}.run.log")"
  done
  if [[ $success != 1 ]]; then
    log "  $day 세 번 실패 — 멈춘다"
    echo 1 > "$ARM/resume_${attempt}.exitcode"
    exit 1
  fi
  log "  $day 완료"
done
echo 0 > "$ARM/resume_${attempt}.exitcode"
log "이어하기 완료 — ${DAYS}일 전부"

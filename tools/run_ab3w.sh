#!/usr/bin/env bash
# 3주 A/B — 정책 전 1주(공통) → 그래프를 두 Neo4j 에 그대로 복제 → 같은 사람을 정책 있음/없음으로 1주씩 (2026-10-05)
#
#   AB_CASE=p013 AB_TAG=small AB_ROSTER=/data/ab3w/rosters/p013_small_150.json \
#   AB_PRE_DAYS=3 AB_POST_DAYS=4 bash tools/run_ab3w.sh
#
# 사용자 결정(2026-10-05): "정책 주입 전 1주일하고 주입 시점 이후로 정책 적용, 미적용 1주일씩 돌려서 AB테스트".
# 두 갈래는 정책 시작일 아침에 같은 그래프(기억·상태·약속까지)에서 출발한다 — 정책 전 격차가 구조적으로 0 이다.
#
# 단계(끝난 단계는 표지 파일을 남기고, 다시 부르면 건너뛴다 — 이어 돌리기):
#   1 준비     명부·정책 파일·엔진 지문·모델 증거를 고정한다
#   2 복원     정책 있음 쪽 Neo4j 에 깨끗한 그래프를 넣고 Day0 를 놓는다(정책 없음)
#   3 정책 전  같은 사람을 정책 없이 돌린다(계획 통로 기준선 파일 없음 = 배수 1)
#   4 기준선   정책 전 날들의 계획액으로 사람별 평일/주말 기준선을 만든다(두 갈래가 같은 파일을 쓴다)
#   5 복제     그래프를 덤프해 정책 없음 쪽 Neo4j 에 그대로 넣고, 두 그래프가 같은지 센다
#   6 정책     정책 있음 쪽에만 정책을 넣는다(환경형은 사회 배경 ID 만 다르다)
#   7 두 갈래  정책 있음/없음을 동시에 돌린다
#   8 보존     원장·기억 모음(정책 전 주 포함 전 기간)·그래프 덤프·체크섬
#
# 지킬 것: 건너뛴 사람-날이 있으면 멈춘다(EXP_NO_SKIP=1 — 그날 행동·기억이 비면 안 된다). 조용한 대체 금지.
# 서버 전용 비밀번호는 저장소 밖 파일($AB_CRED, 기본 /data/ab3w/neo4j_credentials.sh)에서 읽는다.
set -Eeuo pipefail
umask 077

: "${AB_CASE:?AB_CASE required: p012|p013|p010 ...}" "${AB_TAG:?AB_TAG required}" "${AB_ROSTER:?AB_ROSTER required}"
AB_PRE_DAYS=${AB_PRE_DAYS:-7}
AB_POST_DAYS=${AB_POST_DAYS:-7}
AB_WORKERS=${AB_WORKERS:-32}                       # 갈래마다. A100 하나는 합 64 근처에서 포화한다
AB_REPO=${AB_REPO:-$(cd "$(dirname "$0")/.." && pwd)}
AB_ROOT=${AB_ROOT:-/data/ab3w}
AB_PRE_GRAPH=${AB_PRE_GRAPH:-/data/backup_20260927_pre_pilot}
AB_LLM_BASE_URL=${AB_LLM_BASE_URL:-http://localhost:8000/v1}   # 소규모=A100 직접, 본런=GPU 풀 중계
AB_MODEL_ID=${AB_MODEL_ID:-LGAI-EXAONE/EXAONE-4.5-33B-AWQ}
AB_NEO_ON=${AB_NEO_ON:-/data/neo4j-community-5.26.0};   AB_BOLT_ON=${AB_BOLT_ON:-7687};  AB_HTTP_ON=${AB_HTTP_ON:-7474}
AB_NEO_OFF=${AB_NEO_OFF:-/data/neo4j2-community-5.26.0}; AB_BOLT_OFF=${AB_BOLT_OFF:-7688}; AB_HTTP_OFF=${AB_HTTP_OFF:-7475}
AB_CRED=${AB_CRED:-$AB_ROOT/neo4j_credentials.sh}  # NEO4J_PASSWORD_ON / NEO4J_PASSWORD_OFF 를 export 하는 파일
AB_MAX_ATTEMPTS=${AB_MAX_ATTEMPTS:-12}             # 사람-날 하나의 시도 예산. 실행 지문에 들어가므로 런 도중 바꾸지 않는다

source "$AB_REPO/tools/ab3w_cases.sh"
BASE=$AB_ROOT/${AB_CASE}_${AB_TAG}
PRE_START=$(date -d "$START - $AB_PRE_DAYS days" +%F)
DAY0=$(date -d "$PRE_START - 1 day" +%F)
POST_END=$(date -d "$START + $((AB_POST_DAYS-1)) days" +%F)
mkdir -p "$BASE" "$AB_ROOT/locks"
LOG=$BASE/orchestrate.log
log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
trap 'log "FAILED at line $LINENO — 그래프·출력은 그대로 둔다"' ERR

cd "$AB_REPO"
source /data/venv/bin/activate
test -s "$AB_CRED" || { log "비밀번호 파일이 없다: $AB_CRED"; exit 1; }
source "$AB_CRED"
: "${NEO4J_PASSWORD_ON:?}" "${NEO4J_PASSWORD_OFF:?}"
export PYTHONPATH="$AB_REPO" PYTHONIOENCODING=utf-8 NEO4J_USER=neo4j
# 정책 기간은 정책 파일 그대로(원장 내보내기가 파일과 다르면 멈춘다)
FROM=''; UNTIL=''
if [[ -n $POLICY ]]; then
  read -r FROM UNTIL < <(python -c 'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); print(p["effective_from"], p["effective_until"])' "$POLICY")
fi

# 두 인스턴스를 이 런이 쥔다. 다른 런이 같은 인스턴스를 쓰면 여기서 멈춘다.
exec 8>"$AB_ROOT/locks/neo4j_${AB_BOLT_ON}.lock";  flock -n 8 || { log "Neo4j $AB_BOLT_ON 을 다른 런이 쓰고 있다"; exit 1; }
exec 9>"$AB_ROOT/locks/neo4j_${AB_BOLT_OFF}.lock"; flock -n 9 || { log "Neo4j $AB_BOLT_OFF 을 다른 런이 쓰고 있다"; exit 1; }

# ---------------------------------------------------------------- 공통 실행 설정
# 두 갈래와 정책 전 주가 모두 같은 값으로 돈다(실행 지문이 같아야 짝이다). 정책은 그래프에만 있다.
common_env() {
  export LLM_BASE_URL=$AB_LLM_BASE_URL LLM_MODE=exaone_4_5 SIM_PROMPT_VARIANT=v53
  export EXP_SANGSAENG_BASE_RATIO=0.268 EXP_SEED_SANGSAENG=1 EXP_BALANCE_DAYS=39
  export EXP_DURABLES=1 EXP_CATLINE=fold EXP_POLICY_ANONYMOUS=1 POLICY_POI_SORT_BOOST=0
  # 하루 소득: 평소 소비 수준(앵커)만큼 매일 채운다(사용자 선택 '1번', 잔액 유지). 계수 1/2.40 은 적립 채널의 눈금과 같다.
  export EXP_DAILY_INCOME=anchor:0.41667; unset EXP_DAILY_INCOME_MAP
  export EXP_ELIGIBLE_CHANNEL=1
  # 계획 통로: 정책 전 주에는 기준선 파일이 없어 배수 1, 정책 시작일부터 같은 경로에 기준선이 놓인다.
  export EXP_PLAN_DRIVES_TOTAL=1 EXP_PLAN_BASELINE_FILE=$BASE/plan_baseline_live.json
  export EXP_NO_SKIP=1 EXP_AGENT_DAY_MAX_ATTEMPTS=$AB_MAX_ATTEMPTS
  unset SIM_ALLOW_STAGE2_FALLBACK EXP_PAYMENT_CHOICE EXP_GRANT_USE EXP_SPREAD_DAYS
  # 운영자 셸에 남은 값이 런에 새지 않게 한다(사회 배경은 --environment 로만, 백업 경로는 갈래끼리 겹치면 덮어쓴다).
  unset SIM_ENVIRONMENT BACKUP_DIR SIM_POST_DAY_BACKUP_HOOK
  local kv
  for kv in "${CASE_EXPORTS[@]}"; do export "$kv"; done
}
on_env()  { export NEO4J_URI=bolt://localhost:$AB_BOLT_ON  NEO4J_PASSWORD=$NEO4J_PASSWORD_ON; }
off_env() { export NEO4J_URI=bolt://localhost:$AB_BOLT_OFF NEO4J_PASSWORD=$NEO4J_PASSWORD_OFF; }

neo_up() {   # $1=설치 폴더 $2=http 포트. tmux 밖에서 올린다(세션이 닫히면 Neo4j 가 함께 내려갔다).
  local neo=$1 http=$2 i
  curl -fsS -m 5 -o /dev/null "http://localhost:$http" && return 0
  for i in $(seq 1 90); do
    if (( i % 10 == 1 )); then setsid nohup "$neo/bin/neo4j" start >/dev/null 2>&1 < /dev/null || true; fi
    curl -fsS -m 5 -o /dev/null "http://localhost:$http" && return 0
    sleep 2
  done
  log "Neo4j($neo) 를 올리지 못했다"; return 1
}
neo_down() {  # $1=설치 폴더
  "$1/bin/neo4j" stop >/dev/null 2>&1 || true
  local i
  for i in $(seq 1 60); do
    "$1/bin/neo4j" status >/dev/null 2>&1 || return 0
    sleep 2
  done
  log "Neo4j($1) 가 내려가지 않는다"; return 1
}
query_ready() {
  local i
  for i in $(seq 1 45); do
    python -c "
from scripts.neo4j_load._common import driver_session
with driver_session() as s: assert s.run('RETURN 1 AS ready').single()['ready'] == 1
" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}
graph_fingerprint() {   # 현재 NEO4J_URI 그래프의 라벨·관계 수와 명부 사람들의 상태 요약
  python tools/ab3w_graph_fingerprint.py "$BASE/roster.json"
}

# 실행 ID 는 정책 전 주와 두 갈래가 같다(ab3w-<정책>-<태그>). 엔진은 전날 State·기억이 다른 실행 ID 에서 오면
# 멈춘다(run_simulation 'previous State belongs to a different experience run', experience 'cannot mix runs') —
# 두 갈래는 같은 정책 전 주의 기억을 이어받는 같은 경험이고, 서로 다른 그래프·출력 폴더에서 갈래 이름(on/off)으로 갈린다.
llm_wait() {   # 모델 서버가 답할 때까지 기다린다(최대 30분). 끊긴 사이 시도 예산이 닳지 않게 한다.
  local i
  for i in $(seq 1 180); do
    curl -fsS -m 8 "$AB_LLM_BASE_URL/models" >/dev/null 2>&1 && return 0
    (( i % 6 == 1 )) && log "  모델 서버가 답하지 않는다 — 기다린다 ($AB_LLM_BASE_URL)"
    sleep 10
  done
  log "모델 서버가 30분 동안 답하지 않는다 — 멈춘다"; return 1
}

# 하루씩 돈다. 한 날을 세 번까지 다시 부른다. 시도 예산을 다 쓴 사람이 있으면(AgentDayExhausted):
#  - 그날 로그에 모델 서버 연결 오류가 있으면 그 사람들의 실패 기록(checkpoints/attempts_<날>.jsonl)을 지우지 않고
#    옆으로 옮긴 뒤 다시 부른다(하루 최대 2번). 서버가 잠깐 끊겨 예산이 닳은 것은 그 사람의 실패가 아니다.
#  - 연결 오류가 없으면 진짜 실패라 건너뛰지 않고 멈춘다.
# $1=출력 폴더 $2=첫날 $3=날 수 $4=사회 배경 ID $5=on|off(그래프) $6=실행 ID $7=1 이면 첫날 뒤 계획 통로 확인
run_days() {
  local dir=$1 first=$2 n=$3 env_id=$4 side=$5 run_id=$6 check_plan=${7:-0}
  local N
  N=$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$BASE/roster.json")
  mkdir -p "$dir"
  (
    common_env
    if [[ $side == on ]]; then on_env; else off_env; fi
    export SIM_OUTPUT_DIR=$dir SIM_RUN_ID=$run_id
    for ((i=0; i<n; i++)); do
      day=$(date -d "$first + $i days" +%F)
      [[ -s $dir/day_$day.json ]] && continue
      ok=0; moved=0
      for attempt in 1 2 3 4 5; do
        llm_wait || exit 1
        args=(--start "$day" --days 1 --roster "$BASE/roster.json" --workers "$AB_WORKERS")
        if [[ -n $env_id ]]; then args+=(--environment "$env_id"); fi
        log "[$(basename "$dir")] $day 시도 $attempt"
        runlog="$dir/day_${day}_attempt${attempt}.run.log"
        if python -u scripts/sim/run_simulation.py "${args[@]}" > "$runlog" 2>&1 && \
           python tools/ab3w_check_day.py "$dir/summary.json" "$dir/day_$day.json" "$day" "$N"; then
          ok=1; break
        fi
        if grep -q "AgentDayExhausted" "$runlog"; then
          if (( moved < 2 )) && grep -qiE "APIConnectionError|APITimeoutError|timed out|Connection refused|ConnectError|ConnectTimeout|ReadTimeout|RemoteDisconnected|Bad Gateway|Service Unavailable|ServerDisconnected" "$runlog"; then
            moved=$((moved + 1))
            mv "$dir/checkpoints/attempts_${day}.jsonl" "$dir/checkpoints/attempts_${day}.outage$(date +%s).jsonl"
            log "[$(basename "$dir")] $day: 모델 서버 연결 오류로 시도 예산이 닳았다 — 실패 기록을 옮기고 다시 부른다 ($moved/2)"
            continue
          fi
          log "[$(basename "$dir")] $day: 시도 예산($AB_MAX_ATTEMPTS)을 다 쓴 사람이 있다 — 건너뛰지 않고 멈춘다"
          exit 1
        fi
        (( attempt >= 3 && moved == 0 )) && break
      done
      [[ $ok == 1 ]] || { log "[$(basename "$dir")] $day 실패 — 멈춘다"; exit 1; }
      if [[ $check_plan == 1 && $i == 0 ]]; then
        # 계획 통로가 첫날부터 켜졌는지 — 1주가 다 끝난 뒤가 아니라 첫날 뒤에 잡는다.
        python tools/ab3w_plan_channel_check.py "$dir/metrics" | tee -a "$LOG" || exit 1
      fi
    done
    python scripts/report/audit_stage2_generation.py --metrics-dir "$dir/metrics" \
      --expected-per-day "$N" --json-out "$dir/stage2.json" --strict >> "$dir/audit.log" 2>&1
  )
}

# ---------------------------------------------------------------- 1 준비
if [[ ! -s $BASE/prepared.marker ]]; then
  log "=== 1 준비: $AB_CASE · 정책 전 $PRE_START~$(date -d "$START - 1 day" +%F) · 두 갈래 $START~$POST_END · 명부 $(basename "$AB_ROSTER")"
  test -s "$AB_ROSTER" && test -s "$AB_ROSTER.manifest.json"
  cp -n "$AB_ROSTER" "$BASE/roster.json"; cmp "$AB_ROSTER" "$BASE/roster.json"
  cp -n "$AB_ROSTER.manifest.json" "$BASE/roster.manifest.json"
  served=$(curl -fsS -m 10 "$AB_LLM_BASE_URL/models" | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
  [[ $served == "$AB_MODEL_ID" ]] || { log "모델 응답이 $AB_MODEL_ID 가 아니다: $served ($AB_LLM_BASE_URL)"; exit 1; }
  if [[ -n $POLICY ]]; then
    test -s "$POLICY"
    ( common_env; SIM_OUTPUT_DIR="$BASE/preflight" NEO4J_URI='' python scripts/sim/policy_preflight.py "$POLICY" )
  fi
  ( common_env; python tools/ab3w_engine_record.py "$BASE/engine.json" )
  # 실제 서빙 모델 증거(A100 의 SGLang 명령줄·모델 ID). 원장 내보내기가 각 출력 폴더에서 읽는다 — 없으면 멈춘다.
  # 본런의 GPU 풀 중계는 A100 과 같은 모델인지 확인한 일꾼만 받는다(deploy/gpu_pool_kw26).
  python tools/capture_served_model_evidence_20260928.py --out "$BASE/served_model_evidence.json" \
    || { log '모델 증거 수집 실패 — 멈춘다'; exit 1; }
  for sub in pre on off; do mkdir -p "$BASE/$sub"; cp "$BASE/served_model_evidence.json" "$BASE/$sub/"; done
  printf '{"case":"%s","tag":"%s","design":"pre_week_common_then_paired_on_off","pre_start":"%s","start":"%s","post_end":"%s","pre_days":%s,"post_days":%s,"day0":"%s","policy_file":"%s","policy_id":"%s","env_pre":"%s","env_on":"%s","env_off":"%s","served_model":"%s","llm_base_url":"%s","workers_per_arm":%s,"max_attempts":%s,"pre_graph":"%s","prepared_at":"%s"}\n' \
    "$AB_CASE" "$AB_TAG" "$PRE_START" "$START" "$POST_END" "$AB_PRE_DAYS" "$AB_POST_DAYS" "$DAY0" "$POLICY" "$PID" \
    "$ENV_PRE" "$ENV_ON" "$ENV_OFF" "$AB_MODEL_ID" "$AB_LLM_BASE_URL" "$AB_WORKERS" "$AB_MAX_ATTEMPTS" "$AB_PRE_GRAPH" "$(date -Is)" \
    > "$BASE/run_manifest.json"
  (cd "$BASE" && sha256sum roster.json roster.manifest.json run_manifest.json engine.json > frozen_inputs.sha256)
  if [[ -n $POLICY ]]; then sha256sum "$POLICY" > "$BASE/frozen_policy.sha256"; fi
  date -Is > "$BASE/prepared.marker"
fi
(cd "$BASE" && sha256sum -c frozen_inputs.sha256 >/dev/null) || { log '고정한 입력이 바뀌었다'; exit 1; }
if [[ -n $POLICY ]]; then sha256sum -c "$BASE/frozen_policy.sha256" >/dev/null || { log '정책 파일이 바뀌었다'; exit 1; }; fi
# 엔진 소스·실행 설정이 준비 때와 같아야 이어 돌린다(정책 전 주와 두 갈래가 같은 엔진이어야 한다).
( common_env; python tools/ab3w_engine_record.py --check "$BASE/engine.json" ) || { log '엔진 소스나 실행 설정이 준비 때와 다르다'; exit 1; }

# ---------------------------------------------------------------- 2 복원
if [[ ! -s $BASE/pre/graph_restored.marker ]]; then
  log "=== 2 복원: $AB_PRE_GRAPH → Neo4j $AB_BOLT_ON (Day0 $DAY0)"
  (cd "$AB_PRE_GRAPH" && sha256sum -c SHA256SUMS >/dev/null)
  mkdir -p "$BASE/pre"
  neo_down "$AB_NEO_ON"
  "$AB_NEO_ON/bin/neo4j-admin" database load neo4j --from-path="$AB_PRE_GRAPH" --overwrite-destination=true
  neo_up "$AB_NEO_ON" "$AB_HTTP_ON"
  (
    common_env; on_env
    query_ready
    # reset 은 정책 노드도 지운다 — 정책은 복제 뒤 정책 있음 쪽에만 넣으므로 여기서 지워져도 된다.
    python scripts/neo4j_load/97_reset_run_artifacts.py
    DAY_ZERO="$DAY0" python scripts/neo4j_load/08_initial_state.py
    SIM_OUTPUT_DIR="$BASE/pre/preflight" python scripts/sim/policy_preflight.py --expect-no-policy
  )
  printf 'restored_from=%s\nrestored_at=%s\n' "$AB_PRE_GRAPH" "$(date -Is)" > "$BASE/pre/graph_restored.marker"
fi

# ---------------------------------------------------------------- 3 정책 전 주
if [[ ! -s $BASE/pre/complete.marker ]]; then
  rm -f "$BASE/plan_baseline_live.json"     # 정책 전 주: 기준선 없음 = 계획 배수 1
  neo_up "$AB_NEO_ON" "$AB_HTTP_ON"
  log "=== 3 정책 전 주: $PRE_START 부터 ${AB_PRE_DAYS}일 (사회 배경 '${ENV_PRE:-없음}')"
  run_days "$BASE/pre" "$PRE_START" "$AB_PRE_DAYS" "$ENV_PRE" on "ab3w-$AB_CASE-$AB_TAG"
  date -Is > "$BASE/pre/complete.marker"
fi

# ---------------------------------------------------------------- 4 기준선
if [[ ! -s $BASE/plan_baseline_frozen.sha256 ]]; then
  pre_days=$(python -c 'import sys; from datetime import date, timedelta as t; s=date.fromisoformat(sys.argv[1]); print(",".join(str(s+t(i)) for i in range(int(sys.argv[2]))))' "$PRE_START" "$AB_PRE_DAYS")
  python tools/build_plan_baseline_20261003.py --metrics "$BASE/pre/metrics" --days "$pre_days" \
    --out "$BASE/plan_baseline_frozen.json" | tee -a "$LOG"
  python tools/ab3w_baseline_coverage.py "$BASE/plan_baseline_frozen.json" "$BASE/roster.json" | tee -a "$LOG"
  (cd "$BASE" && sha256sum plan_baseline_frozen.json > plan_baseline_frozen.sha256)
fi
(cd "$BASE" && sha256sum -c plan_baseline_frozen.sha256 >/dev/null)

# ---------------------------------------------------------------- 5 복제
if [[ ! -s $BASE/fork.marker ]]; then
  log "=== 5 복제: Neo4j $AB_BOLT_ON 의 그래프를 $AB_BOLT_OFF 에 그대로 넣는다"
  mkdir -p "$BASE/pre/graph_backup"
  if [[ ! -s $BASE/pre/graph_backup/SHA256SUMS ]]; then
    neo_down "$AB_NEO_ON"
    "$AB_NEO_ON/bin/neo4j-admin" database dump neo4j --to-path="$BASE/pre/graph_backup" --overwrite-destination=true
    "$AB_NEO_ON/bin/neo4j" version > "$BASE/pre/graph_backup/neo4j_version.txt"
    (cd "$BASE/pre/graph_backup" && sha256sum neo4j.dump neo4j_version.txt > SHA256SUMS)
  fi
  (cd "$BASE/pre/graph_backup" && sha256sum -c SHA256SUMS >/dev/null)
  neo_down "$AB_NEO_OFF"
  "$AB_NEO_OFF/bin/neo4j-admin" database load neo4j --from-path="$BASE/pre/graph_backup" --overwrite-destination=true
  neo_up "$AB_NEO_ON" "$AB_HTTP_ON"
  neo_up "$AB_NEO_OFF" "$AB_HTTP_OFF"
  fp_on=$(on_env; query_ready && graph_fingerprint)
  fp_off=$(off_env; query_ready && graph_fingerprint)
  printf '%s\n' "$fp_on" > "$BASE/pre/graph_fingerprint_on.json"
  printf '%s\n' "$fp_off" > "$BASE/pre/graph_fingerprint_off.json"
  [[ -n $fp_on && $fp_on == "$fp_off" ]] || { log '복제한 두 그래프가 다르다 — 멈춘다'; exit 1; }
  log "  두 그래프가 같다 ($(printf '%s' "$fp_on" | sha256sum | cut -c1-12))"
  date -Is > "$BASE/fork.marker"
fi

# ---------------------------------------------------------------- 6 정책
if [[ ! -s $BASE/policy.marker ]]; then
  neo_up "$AB_NEO_ON" "$AB_HTTP_ON"
  neo_up "$AB_NEO_OFF" "$AB_HTTP_OFF"
  if [[ -n $POLICY ]]; then
    log "=== 6 정책: $PID 를 정책 있음 쪽(Neo4j $AB_BOLT_ON)에만 넣는다"
    (
      common_env; on_env
      python scripts/neo4j_load/10_load_grant_policy.py "$POLICY"
      SIM_OUTPUT_DIR="$BASE/on/preflight" python scripts/sim/policy_preflight.py --require-db "$POLICY"
    )
  else
    log "=== 6 정책: 환경형 — 정책 있음 '$ENV_ON' / 정책 없음 '$ENV_OFF'"
    ( common_env; on_env; SIM_OUTPUT_DIR="$BASE/on/preflight" python scripts/sim/policy_preflight.py --expect-no-policy )
  fi
  ( common_env; off_env; SIM_OUTPUT_DIR="$BASE/off/preflight" python scripts/sim/policy_preflight.py --expect-no-policy )
  date -Is > "$BASE/policy.marker"
fi

# ---------------------------------------------------------------- 7 두 갈래
cp -n "$BASE/plan_baseline_frozen.json" "$BASE/plan_baseline_live.json"
cmp "$BASE/plan_baseline_frozen.json" "$BASE/plan_baseline_live.json"
if [[ ! -s $BASE/arms_complete.marker ]]; then
  neo_up "$AB_NEO_ON" "$AB_HTTP_ON"
  neo_up "$AB_NEO_OFF" "$AB_HTTP_OFF"
  log "=== 7 두 갈래: $START 부터 ${AB_POST_DAYS}일, 갈래마다 동시 $AB_WORKERS"
  run_days "$BASE/on"  "$START" "$AB_POST_DAYS" "$ENV_ON"  on  "ab3w-$AB_CASE-$AB_TAG" 1 & pid_on=$!
  run_days "$BASE/off" "$START" "$AB_POST_DAYS" "$ENV_OFF" off "ab3w-$AB_CASE-$AB_TAG" 1 & pid_off=$!
  rc_on=0; rc_off=0
  wait "$pid_on" || rc_on=$?
  wait "$pid_off" || rc_off=$?
  [[ $rc_on == 0 && $rc_off == 0 ]] || { log "두 갈래 종료 코드: 있음 $rc_on / 없음 $rc_off — 멈춘다(이어 돌리면 끝난 날은 건너뛴다)"; exit 1; }
  # 계획 통로가 실제로 켜졌는지 — 소비가 있는 행의 대부분에 기준선이 쓰였어야 한다.
  python tools/ab3w_plan_channel_check.py "$BASE/on/metrics" "$BASE/off/metrics" | tee -a "$LOG"
  date -Is > "$BASE/arms_complete.marker"
fi

# ---------------------------------------------------------------- 8 보존
preserve_arm() {   # $1=on|off
  local arm=$1 dir=$BASE/$1 neo
  [[ -s $dir/external_copy_verified.txt ]] && return 0
  (
    common_env
    if [[ $arm == on ]]; then on_env; else off_env; fi
    extra=()
    if [[ $arm == on && -n $POLICY ]]; then
      extra=(--policy-id "$PID" --policy-file "$POLICY" --effective-from "$FROM" --effective-until "$UNTIL")
    fi
    for ledger in $LEDGERS; do
      case $ledger in
        sector)   python scripts/report/export_multi_policy_sector_ledger.py --roster "$BASE/roster.json" \
                    --start "$START" --end "$POST_END" --arm "$arm" "${extra[@]}" \
                    --metrics-dir "$dir/metrics" --out "$dir/sector.ledger.jsonl";;
        policy)   python scripts/report/export_policy_daily_ledger.py --roster "$BASE/roster.json" \
                    --start "$START" --end "$POST_END" --arm "$arm" --policy-id "$PID" --policy-file "$POLICY" \
                    --metrics-dir "$dir/metrics" --out "$dir/policy.ledger.jsonl";;
        distancing)
                  if [[ $arm == on ]]; then darm=restricted; else darm=control_hold; fi
                  python scripts/report/export_distancing_daily_ledger.py --arm "$darm" --roster "$BASE/roster.json" \
                    --start "$START" --end "$POST_END" --metrics-dir "$dir/metrics" --out "$dir/distancing.ledger.jsonl";;
        cashback) python scripts/report/export_cashback_month.py --allow-partial-month \
                    --month "${START:0:7}" --arm "$arm" --policy-id "$PID" --policy-file "$POLICY" \
                    --base-ratio 0.268 --roster "$BASE/roster.json" \
                    --metrics-dir "$dir/metrics" --out "$dir/cashback.ledger.jsonl";;
        *) echo "모르는 원장 $ledger" >&2; exit 2;;
      esac
    done
    # 기억·계획·지출·지갑 — 정책 전 주부터 끝까지, 사람 단위로. 날이 하나라도 빠지면 멈춘다.
    python scripts/report/export_agent_dossier.py --roster "$BASE/roster.json" \
      --start "$PRE_START" --end "$POST_END" --arm "$arm" --out "$dir/dossier.jsonl"
    python tools/ab3w_dossier_check.py "$dir/dossier.jsonl.manifest.json" "$((AB_PRE_DAYS + AB_POST_DAYS))"
  ) 2>&1 | tee -a "$LOG"
  if [[ $arm == on ]]; then neo=$AB_NEO_ON; else neo=$AB_NEO_OFF; fi
  mkdir -p "$dir/graph_backup"
  if [[ ! -s $dir/graph_backup/SHA256SUMS ]]; then
    neo_down "$neo"
    "$neo/bin/neo4j-admin" database dump neo4j --to-path="$dir/graph_backup" --overwrite-destination=true
    "$neo/bin/neo4j" version > "$dir/graph_backup/neo4j_version.txt"
    (cd "$dir/graph_backup" && sha256sum neo4j.dump neo4j_version.txt > SHA256SUMS)
    if [[ $arm == on ]]; then neo_up "$AB_NEO_ON" "$AB_HTTP_ON"; else neo_up "$AB_NEO_OFF" "$AB_HTTP_OFF"; fi
  fi
  (cd "$dir/graph_backup" && sha256sum -c SHA256SUMS >/dev/null)
  (cd "$dir" && find . -path ./graph_backup -prune -o -type f -print0 | sort -z | xargs -0 sha256sum > "../${arm}_outputs.sha256")
  printf 'verified_on=%s\n' "$(date -Is)" > "$dir/external_copy_verified.txt"
  log "=== 8 보존 $arm 완료"
}
# 보존 전에 두 인스턴스를 띄운다 — 재부팅·덤프 실패 뒤 다시 불러도 내보내기가 연결되게.
neo_up "$AB_NEO_ON" "$AB_HTTP_ON"
neo_up "$AB_NEO_OFF" "$AB_HTTP_OFF"
preserve_arm on
preserve_arm off
(cd "$BASE/pre" && find . -path ./graph_backup -prune -o -type f -print0 | sort -z | xargs -0 sha256sum > ../pre_outputs.sha256)
# ---------------------------------------------------------------- 9 채점
# 공통 채점(같은 사람·같은 날 정책 있음 - 없음) + 정책 전용 채점(P012·P013·거리두기). 결과는 $BASE/score/.
log "=== 9 채점"
( common_env; on_env; bash tools/ab3w_score.sh "$BASE" "$AB_CASE" ) >> "$LOG" 2>&1 \
  || { log '채점 실패 — 원장·덤프는 보존돼 있다. tools/ab3w_score.sh 로 다시 채점한다'; exit 1; }
log "=== 끝: $BASE (채점 $BASE/score)"

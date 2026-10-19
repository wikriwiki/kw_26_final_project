#!/usr/bin/env bash
# 채점기 고침을 서버로 올린다. **돌고 있는 라운드가 없을 때만.**
#
#   bash tools/deploy_scoring_fixes.sh            # 무엇이 바뀌는지만 본다
#   bash tools/deploy_scoring_fixes.sh --apply    # 실제로 올린다
#
# ## 무엇을 올리고 무엇을 안 올리는가
#
#   올린다   scripts/sim/score_policy.py            지표 7개를 되살린 고침
#            scripts/sim/mechanisms/__init__.py     범용 대체 + poi_restriction
#            scripts/sim/mechanisms/generic.py      서버에 아예 없다
#            data/experiments/scoring_table.json    P016 · 라운드2 결과 블록
#            scripts/report/audit_*.py              전수 점검 둘
#
#   **안 올린다**  scripts/sim/run_simulation.py
#       서버의 것은 **다른 세대**다. 경험 기억 층·소득·증거 무결성이 들어오기 전
#       기반에 사용처 표시 고침만 얹혀 있다. 정답지 라운드 결과는 전부 그 세대가
#       낸 것이다. 여기서 저장소 판으로 덮으면 런 파이프라인 전체가 바뀐다 —
#       채점기 배포가 아니라 이주(migration)이고, 그것은 따로 계획해야 한다.
#       (memory: 서버 작업 사본이 진짜)
#
# ## 채점 결과가 달라진다
#
# `fetch()` 에서 `actual_spent > 0` 조건이 빠졌다. 0원 쓴 것도 관측으로 들어와
# 쌍체 표본이 넓어진다. **이미 채점한 라운드를 새 채점기로 다시 매기면 숫자가
# 달라진다.** 그래서 라운드가 도는 중에는 올리지 않는다.
set -uo pipefail
cd "$(dirname "$0")/.."

KEY=${SIM_KEY:-/c/Users/Administrator/.ssh/outofmemory.pem}
HOST=${SIM_HOST:-outofmemory@123.37.28.167}
PORT=${SIM_PORT:-10022}
APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=25 $HOST"
say(){ echo "[$(date +%H:%M)] $*"; }

FILES="
scripts/sim/score_policy.py
scripts/sim/mechanisms/__init__.py
scripts/sim/mechanisms/generic.py
data/experiments/scoring_table.json
scripts/report/audit_indicator_coverage.py
scripts/report/audit_policy_delivery.py
"

# ------------------------------------------------------- ① 돌고 있는 런이 없는가
say "돌고 있는 런이 있는지 본다"
# 대괄호 한 글자로 **자기 자신을 세지 않게** 한다. 이것이 없으면 확인하러 보낸
# 명령줄 자체가 패턴에 걸려 런이 없는데도 1개로 나온다.
RUNNING=$($SSH 'pgrep -c -f "[r]un_simulation" || true' 2>/dev/null | tr -dc 0-9)
echo "  run_simulation 프로세스 ${RUNNING:-?}개"
if [ "${RUNNING:-0}" != "0" ]; then
  echo "  **런이 돌고 있다 — 채점 기준을 바꾸지 않는다. 끝나고 다시.**"
  [ $APPLY -eq 1 ] && exit 1
fi

# ---------------------------------------------------------------- ② 지금 상태
say "서버와 저장소의 sha256 을 맞대 본다"
printf '%-46s %-10s %-10s %s\n' 파일 서버 저장소 판정
printf -- '-%.0s' $(seq 1 92); echo
for f in $FILES; do
  [ -f "$f" ] || { printf '%-46s %-10s %-10s %s\n' "$f" - - "저장소에 없다"; continue; }
  L=$(sha256sum "$f" 2>/dev/null | cut -c1-8)
  R=$($SSH "sha256sum /data/repo/$f 2>/dev/null | cut -c1-8" 2>/dev/null | tr -d '\r')
  if [ -z "$R" ]; then V="서버에 없다 — 새로 올린다"
  elif [ "$L" = "$R" ]; then V="같다 — 건너뛴다"
  else V="**다르다 — 올린다**"; fi
  printf '%-46s %-10s %-10s %s\n' "$f" "${R:--}" "$L" "$V"
done

if [ $APPLY -eq 0 ]; then
  echo
  say "여기까지가 점검이다. 올리려면 --apply"
  exit 0
fi

# ---------------------------------------------------------------- ③ 백업 후 배포
STAMP=$(date +%Y%m%d_%H%M%S)
say "서버에 백업을 만든다 (/data/backup_deploy_$STAMP)"
$SSH "mkdir -p /data/backup_deploy_$STAMP && cd /data/repo && for f in $(echo $FILES | tr '\n' ' '); do [ -f \$f ] && install -D \$f /data/backup_deploy_$STAMP/\$f; done; find /data/backup_deploy_$STAMP -type f | wc -l"

for f in $FILES; do
  [ -f "$f" ] || continue
  $SSH "mkdir -p /data/repo/$(dirname $f)"
  if scp -q -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=no "$f" "$HOST:/data/repo/$f"; then
    echo "  → $f"
  else
    echo "  **실패 $f**"
  fi
done

# --------------------------------------------------------------- ④ 올린 뒤 점검
say "서버에서 전수 점검 둘을 돌린다"
$SSH 'cd /data/repo && source /data/venv/bin/activate && export PYTHONIOENCODING=utf-8 PYTHONPATH=/data/repo && \
  python scripts/report/audit_indicator_coverage.py --quiet; echo "---"; \
  python scripts/report/audit_policy_delivery.py | tail -6'

say "백업: /data/backup_deploy_$STAMP — 되돌리려면 거기서 복사한다"

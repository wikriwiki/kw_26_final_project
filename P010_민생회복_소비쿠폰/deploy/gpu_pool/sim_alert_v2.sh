#!/usr/bin/env bash
# 본런 경보 (2026-10-06) — A100 서버에서 혼자 돌며 이상이 생기면 텔레그램으로 바로 알린다.
#
#   ALERT_RUNS="p012_main p013_main" bash sim_alert.sh        (tmux simalert)
#
# 설정: /data/gpu_pool/telegram.env (권한 600) — TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
# 같은 경보는 상태가 바뀔 때만 한 번 보낸다(해소되면 '해소' 한 번). 6시간마다 진행 요약 한 번.
# 감시 원칙(실패를 크게): 확인할 수 없는 것도 경보다 — 로그가 없으면 '로그 없음', 중계가 답하지 않으면 '중계 무응답'.
set -u
ENVF=/data/gpu_pool/telegram.env
STATE=/data/gpu_pool/sim_alert.state
LOG=/data/gpu_pool/sim_alert.log
RUNS=${ALERT_RUNS:-}
BOLTS=${ALERT_BOLTS:-}    # 하루 검수용 "런:on_bolt:off_bolt ..."(예: "p012_main:7687:7688")
COLABS=${ALERT_COLAB:-}   # 경보 대상 Colab 일꾼(예: "colab1-a colab1-b"). 시험 계정은 넣지 않는다
INTERVAL=${ALERT_INTERVAL:-120}
STALL_MIN=${ALERT_STALL_MIN:-60}
SUMMARY_EVERY=${ALERT_SUMMARY_SEC:-21600}
touch "$STATE"
# Neo4j 암호(하루 검수용) — 실행기와 같은 파일
source /data/ab3w/neo4j_credentials.sh 2>/dev/null || true

send() {   # $1 = 본문
  # shellcheck disable=SC1090
  source "$ENVF" 2>/dev/null
  if [[ -z ${TELEGRAM_BOT_TOKEN:-} || -z ${TELEGRAM_CHAT_ID:-} ]]; then
    echo "$(date -Is) [보내지 못함: 토큰/채팅 ID 없음] $1" >> "$LOG"; return 1
  fi
  if curl -fsS -m 20 "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
       --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" --data-urlencode "text=[KW26 A100] $1" >/dev/null 2>>"$LOG"; then
    echo "$(date -Is) [보냄] $1" >> "$LOG"
  else
    echo "$(date -Is) [보내기 실패] $1" >> "$LOG"; return 1
  fi
}

flag() {   # $1 = 키, $2 = 1(문제)/0(정상), $3 = 문제 문구, $4 = 해소 문구
  local key=$1 bad=$2 was
  was=$(grep -c "^$key\$" "$STATE")
  if [[ $bad == 1 && $was == 0 ]]; then
    send "⚠ $3" && echo "$key" >> "$STATE"
  elif [[ $bad == 0 && $was != 0 ]]; then
    send "✓ 해소: ${4:-$3}" && sed -i "/^$key\$/d" "$STATE"
  fi
}

progress() {   # $1 = 런 이름(예: p012_main) → "정책 전 n/7 · 있음 n/7 · 없음 n/7"
  local B=/data/ab3w/$1 out=""
  for arm in pre on off; do
    out+="$arm $(ls "$B/$arm"/day_*.json 2>/dev/null | wc -l) "
  done
  echo "$out"
}

last_summary=$(date +%s)
send "경보 시작 — 감시 런: ${RUNS:-없음} · Colab: ${COLABS:-없음} · 간격 ${INTERVAL}s · 진행 멈춤 기준 ${STALL_MIN}분" || true
while true; do
  for r in $RUNS; do
    c=${r%_*}
    runlog=/data/ab3w/logs/${r}.run.log
    if [[ ! -f $runlog ]]; then flag "nolog_$r" 1 "$r 실행 로그가 없다($runlog)"; continue; fi
    flag "nolog_$r" 0 "$r 실행 로그 없음" "$r 실행 로그 생김"
    done_ok=$(grep -c "=== 끝:" "$runlog")
    failed=$(grep -cE "FAILED at line|멈춘다" "$runlog")
    # 실행기 프로세스는 환경으로만 정책을 알 수 있어 /proc 에서 찾는다
    alive=0
    for pid in $(pgrep -f "bash tools/run_ab3w.sh"); do
      tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -qx "AB_CASE=$c" && alive=1
    done
    flag "fail_$r" "$([[ $failed -gt 0 ]] && echo 1 || echo 0)" "$r 실행기가 멈췄다: $(grep -E 'FAILED at line|멈춘다' "$runlog" | tail -1 | cut -c1-200)"
    if [[ $done_ok -gt 0 ]]; then
      flag "done_$r" 1 "$r 끝 — $(progress "$r")" ; continue
    fi
    flag "dead_$r" "$([[ $alive == 0 ]] && echo 1 || echo 0)" "$r 실행기 프로세스가 없다(끝 표지도 없음) — $(tail -1 "$runlog" | cut -c1-160)" "$r 실행기 다시 돎"
    age=$(( ( $(date +%s) - $(stat -c %Y "$runlog") ) / 60 ))
    # 로그는 날마다 '시도' 줄을 남긴다. 하루가 길 수 있어 출력 폴더의 최근 변경도 함께 본다.
    newest=$(find /data/ab3w/$r -type f -mmin -"$STALL_MIN" 2>/dev/null | head -1)
    flag "stall_$r" "$([[ -z $newest && $age -ge $STALL_MIN ]] && echo 1 || echo 0)" "$r ${STALL_MIN}분 넘게 진행 없음 — $(progress "$r")" "$r 진행 다시 보임"
    retries=$(grep -cE "시도 [3-5]$" "$runlog")
    flag "retry_$r" "$([[ $retries -gt 0 ]] && echo 1 || echo 0)" "$r 같은 날을 3번 이상 다시 돌림: $(grep -E '시도 [3-5]$' "$runlog" | tail -1 | cut -c1-160)"
  done
  # 하루 검수 — 각 갈래의 하루가 끝나면(day_*.json) 결제 원장·이유·기억을 그래프에서 읽어 검사한다
  for spec in $BOLTS; do
    r=${spec%%:*}; rest=${spec#*:}; bon=${rest%%:*}; boff=${rest#*:}
    B=/data/ab3w/$r; mkdir -p "$B/audit"
    for arm in pre on off; do
      for f in "$B/$arm"/day_*.json; do
        [[ -e $f ]] || continue
        d=$(basename "$f" .json); d=${d#day_}
        [[ -e $B/audit/${arm}_${d}.json ]] && continue
        bolt=$([[ $arm == off ]] && echo "$boff" || echo "$bon")
        pw=$([[ $arm == off ]] && echo "${NEO4J_PASSWORD_OFF:-}" || echo "${NEO4J_PASSWORD_ON:-}")
        repo=/data/repo_ab3w_20261006; for m in ${ALERT_REPOS:-}; do if [[ ${m%%=*} == $r ]]; then repo=${m#*=}; fi; done
        out=$(cd "$repo" && NEO4J_URI=bolt://localhost:$bolt NEO4J_PASSWORD=$pw timeout 600               /data/venv/bin/python tools/ab3w_day_audit.py "$B" "$arm" "$d" --json "$B/audit/${arm}_${d}.json" 2>/dev/null)
        rc=$?
        echo "$(date -Is) 검수 $r $arm $d rc=$rc $(head -1 <<<"$out")" >> "$LOG"
        if [[ $rc == 2 ]]; then
          send "⚠ 하루 검수 하드 오류 $r $arm $d — $(sed -n 2p <<<"$out" | cut -c1-300)"
        elif [[ $rc != 0 ]]; then
          rm -f "$B/audit/${arm}_${d}.json"; send "⚠ 하루 검수를 돌리지 못함 $r $arm $d (rc=$rc) — 다음 주기에 다시 시도"
        fi
      done
    done
  done
  # 모델 서버·중계
  a100=$(curl -s -m 8 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health)
  flag a100 "$([[ $a100 == 200 ]] && echo 0 || echo 1)" "A100 SGLang /health 응답 $a100 — 재시작하면 시드가 바뀌어 Colab 이 모두 거부된다" "A100 SGLang 응답 정상"
  st=$(curl -s -m 8 http://127.0.0.1:30100/pool/status)
  if [[ -z $st ]]; then
    flag proxy 1 "GPU 풀 중계(30100)가 답하지 않는다"
  else
    flag proxy 0 "" "GPU 풀 중계 응답 정상"
    while read -r name healthy; do
      [[ " $COLABS " == *" $name "* ]] || continue
      # 한 번이라도 받아들여진 일꾼만 감시한다(아직 안 붙은 계정은 경보 대상이 아니다). 처음 붙으면 알린다.
      if [[ $healthy == True ]] && ! grep -qx "seen_$name" "$STATE"; then
        echo "seen_$name" >> "$STATE"; send "✓ Colab $name 붙음 — 중계가 받아들임"
      fi
      grep -qx "seen_$name" "$STATE" || continue
      flag "colab_$name" "$([[ $healthy == True ]] && echo 0 || echo 1)" "Colab $name 빠짐(끊김·24시간 한도·재시작 필요)" "Colab $name 다시 일함"
    done < <(python3 -c '
import json,sys
d=json.loads(sys.stdin.read())
for b in d["backends"]:
    if not b["local"]: print(b["name"], b["healthy"])
' <<<"$st" 2>/dev/null)
  fi
  free=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
  flag disk "$([[ ${free:-0} -lt 60 ]] && echo 1 || echo 0)" "/data 여유 ${free}GB" "/data 여유 ${free}GB"
  memfree=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  flag mem "$([[ ${memfree:-0} -lt 15 ]] && echo 1 || echo 0)" "메모리 여유 ${memfree}GB" "메모리 여유 ${memfree}GB"
  now=$(date +%s)
  if (( now - last_summary >= SUMMARY_EVERY )); then
    msg="진행 요약 $(date '+%m/%d %H:%M')"
    for r in $RUNS; do msg+=" | $r: $(progress "$r")"; done
    msg+=" | $(python3 -c '
import json,sys
d=json.loads(sys.stdin.read())
print("처리: " + ", ".join(f"{b[\"name\"]} {b[\"served\"]}" for b in d["backends"] if b["served"]))
' <<<"$st" 2>/dev/null)"
    send "$msg"; last_summary=$now
  fi
  sleep "$INTERVAL"
done

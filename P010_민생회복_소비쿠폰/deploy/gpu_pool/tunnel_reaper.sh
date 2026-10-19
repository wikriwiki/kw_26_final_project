#!/usr/bin/env bash
# 끊긴 Colab 터널 정리 (2026-10-06).
#
# 문제: Colab 런타임이 갑자기 꺼지면 서버 sshd 는 상대가 죽은 줄 모른다(ClientAliveInterval 0, TCP keepalive 7200초).
#       그 세션이 계정 포트(180N1·180N2)를 계속 쥐고 있어, Colab 을 다시 띄워도 'remote port forwarding failed' 로
#       최대 약 2시간 다시 붙지 못한다.
# 처리: 한 sshd 세션이 쥔 계정 포트가 **모두** 연속 FAILS 번(간격 INTERVAL 초) /health 에 답하지 않으면 그 세션만 끊는다.
#       서버 하나만 재기동 중이면(다른 하나는 답함) 끊지 않는다. 끊긴 쪽 진행 중 요청은 중계가 A100 으로 다시 보낸다.
# 사용자 권한으로 돈다(시스템 설정을 바꾸지 않는다). PID 로만 끊는다(pkill -f 금지 — 자기 명령줄을 물 수 있다).
set -u
INTERVAL=${INTERVAL:-20}
FAILS=${FAILS:-4}
LOG=${LOG:-/data/gpu_pool/tunnel_reaper.log}
declare -A fail
log() { echo "$(date -Is) $*" >> "$LOG"; }
# 터널 소켓을 쥔 sshd 세션은 /proc/<pid>/fd 를 막아 두어(덤프 불가) 사용자 권한 ss 로는 PID 가 안 보인다.
# PID 읽기만 sudo 로 한다(설정 변경 없음). 끊기는 사용자 권한 kill 이다(세션 주인이 이 사용자).
if sudo -n ss -ltnH >/dev/null 2>&1; then SS="sudo -n ss"; else SS="ss"; fi
log "start interval=${INTERVAL}s fails=$FAILS pid=$$ ss='$SS'"
[[ $SS == ss ]] && log "경고: sudo 없이 ss — 터널 PID 가 안 보여 정리하지 못한다"
while true; do
  declare -A pid_ports=() pid_bad=()
  # 계정 포트 18011~18062 중 듣고 있는 것과 그 sshd PID
  while read -r port pid; do
    [[ -z $pid ]] && continue
    if curl -s -m 4 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health" | grep -q '^200$'; then
      fail[$port]=0
    else
      fail[$port]=$(( ${fail[$port]:-0} + 1 ))
    fi
    pid_ports[$pid]+="$port "
    (( ${fail[$port]} >= FAILS )) && pid_bad[$pid]=$(( ${pid_bad[$pid]:-0} + 1 ))
  done < <($SS -ltnpH 2>/dev/null | awk '{split($4,a,":"); p=a[length(a)]; if (p ~ /^180[1-6][12]$/) {match($0,/pid=[0-9]+/); print p, substr($0,RSTART+4,RLENGTH-4)}}')
  for pid in "${!pid_ports[@]}"; do
    n=$(wc -w <<<"${pid_ports[$pid]}")
    if [[ ${pid_bad[$pid]:-0} -eq $n ]]; then
      comm=$(ps -o comm= -p "$pid" 2>/dev/null)
      owner=$(ps -o user= -p "$pid" 2>/dev/null)
      if [[ $comm == sshd* && $owner == "$(id -un)" ]]; then
        log "kill sshd pid=$pid ports=${pid_ports[$pid]}— 모든 포트가 ${FAILS}번 연속 응답 없음"
        kill "$pid" 2>>"$LOG" && for p in ${pid_ports[$pid]}; do fail[$p]=0; done
      else
        log "skip pid=$pid comm=$comm owner=$owner ports=${pid_ports[$pid]}— sshd 가 아니거나 내 것이 아님"
      fi
    fi
  done
  unset pid_ports pid_bad
  sleep "$INTERVAL"
done

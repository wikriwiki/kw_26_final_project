#!/usr/bin/env bash
# 본런이 끝나기를 기다렸다가 마무리를 잇는다. 본런이 실패하면 **크게** 적고 멈춘다.
# 기다림은 러너의 종료코드 파일로 판단한다 — 보이지 않으면 계속 본다(3일 한도).
# 경로는 환경변수로 바꿀 수 있다 — 띄우기 전에 판정을 시험하기 위해서다.
set -uo pipefail
EXITF=${CHAIN_EXITCODE:-/data/p012m.exitcode}
LOG=${CHAIN_LOG:-/data/p012m_chain.log}
STATUS=${CHAIN_STATUS:-/data/p012m_chain.status}
FINISH=${CHAIN_FINISH:-/data/pilot_repo_20260927/tools/run_p012_finish_20260929.sh}
RUNLOG=${CHAIN_RUNLOG:-/data/p012m.log}
NAP=${CHAIN_NAP:-60}
say() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
say "대기 시작 — $EXITF 를 본다"
for _ in $(seq 1 4320); do
  if [[ -s $EXITF ]]; then
    rc=$(cat "$EXITF")
    if [[ $rc == 0 ]]; then
      say "본런 성공 — 마무리 시작"
      if bash "$FINISH"; then
        say "CHAIN_OK"; echo OK > "$STATUS"
      else
        say "CHAIN_FINISH_FAILED"; echo FINISH_FAILED > "$STATUS"
      fi
    else
      say "CHAIN_RUN_FAILED exit=$rc — 마무리를 하지 않는다. 마지막 로그:"
      tail -8 "$RUNLOG" 2>/dev/null | tee -a "$LOG"
      echo "RUN_FAILED $rc" > "$STATUS"
    fi
    exit 0
  fi
  sleep "$NAP"
done
say "CHAIN_TIMEOUT — 3일이 지나도 본런 종료코드가 없다"
echo TIMEOUT > "$STATUS"

#!/bin/bash
# Qwen3.6-35B-A3B-AWQ 다운로드 완료 → 14B AWQ sim+vLLM kill → Qwen3.6 vLLM 시작 → sim resume
set -u

WD="/g/내 드라이브/Kw/final_project"
cd /c/Users/Administrator
LOG="$WD/logs/auto_swap_qwen36.log"
mkdir -p "$WD/logs"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== auto_swap_qwen36 시작 (PID=$$) ==="

HF_DIR_WSL="/home/administrator/.cache/huggingface/hub/models--QuantTrio--Qwen3.6-35B-A3B-AWQ"
TIMEOUT=$((SECONDS + 3600))
while true; do
    if [ "$SECONDS" -gt "$TIMEOUT" ]; then
        log "FATAL: 다운 30분 초과"
        exit 1
    fi
    INC=$(wsl.exe -e bash -c "find $HF_DIR_WSL/blobs -name '*.incomplete' 2>/dev/null | wc -l" 2>/dev/null | tr -d '\r\n' | grep -oE '^[0-9]+' || echo 99)
    SIZE_KB=$(wsl.exe -e bash -c "du -s $HF_DIR_WSL 2>/dev/null | awk '{print \$1}'" 2>/dev/null | tr -d '\r\n' | grep -oE '^[0-9]+' || echo 0)
    SIZE_GB=$(awk -v s=$SIZE_KB 'BEGIN{printf "%.2f", s/1024/1024}')
    if [ "$INC" = "0" ] && [ "$SIZE_KB" -gt 22000000 ]; then
        log "STEP 1 완료: Qwen3.6 ${SIZE_GB}GB"
        break
    fi
    log "STEP 1: ${SIZE_GB}GB incomplete=$INC"
    sleep 30
done

log "STEP 2: 14B AWQ sim+vLLM kill"
ps -ef 2>/dev/null | grep -E "sim_resume_14b" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
taskkill.exe //F //IM python3.13.exe 2>&1 | tail -2
wsl.exe -e bash -c "pkill -9 -f 'vllm serve' 2>&1; sleep 3"
sleep 5
rm -f /tmp/sim_v9.lock

log "STEP 3: Qwen3.6 vLLM 시작"
wsl.exe -e bash -c "
source /home/administrator/miniconda3/etc/profile.d/conda.sh
conda activate vllm
nohup vllm serve QuantTrio/Qwen3.6-35B-A3B-AWQ \
    --quantization awq_marlin \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --port 8000 \
    --enable-prefix-caching \
    --language-model-only \
    --trust-remote-code > /tmp/vllm_qwen36.log 2>&1 < /dev/null &
disown
"

log "STEP 4: vLLM 준비 대기 (15분)"
W_TIMEOUT=$((SECONDS + 900))
while [ "$SECONDS" -lt "$W_TIMEOUT" ]; do
    M=$(curl -sS -m 5 http://localhost:8000/v1/models 2>/dev/null | python3 -c "
import sys, json
try: print(json.loads(sys.stdin.read())['data'][0]['id'])
except: pass
" 2>/dev/null)
    if [ "$M" = "QuantTrio/Qwen3.6-35B-A3B-AWQ" ]; then
        log "vLLM 준비: $M"
        break
    fi
    sleep 15
done

log "STEP 5: sim resume (qwen36, workers=32)"
cat > /tmp/sim_resume_qwen36.sh << 'INNER'
#!/bin/bash
set -u
LOCK="/tmp/sim_v9.lock"
if [ -f "$LOCK" ]; then
    OWNER=$(cat "$LOCK" 2>/dev/null)
    if [ -n "$OWNER" ] && kill -0 "$OWNER" 2>/dev/null; then exit 1; fi
    rm -f "$LOCK"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
WD="/g/내 드라이브/Kw/final_project"
cd "$WD"
LOG="$WD/logs/auto_pipeline_v9_simonly.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "=== START qwen36 workers=32 (PID=$$) ==="
export LLM_MODE=qwen36_35b_a3b_awq
export PYTHONUNBUFFERED=1
ATTEMPT=0; MAX=10
while [ "$ATTEMPT" -lt "$MAX" ]; do
    ATTEMPT=$((ATTEMPT+1))
    SIMLOG="$WD/logs/sim_v9_qwen36_attempt${ATTEMPT}.log"
    log "attempt $ATTEMPT"
    python3 -u "$WD/scripts/sim/run_simulation.py" --start 2026-05-25 --days 3 --workers 24 > "$SIMLOG" 2>&1
    RC=$?
    log "exit=$RC"
    LAST="/c/Users/Administrator/sim_output/metrics/day_2026-05-27.jsonl"
    if [ -f "$LAST" ] && [ "$(wc -l < "$LAST")" -ge 14560 ]; then log "5/27 완료"; break; fi
    [ "$RC" -eq 0 ] && break
    sleep 30
done
log "=== STEP 4 visited Memory ==="
python3 "$WD/scripts/sim/recover_visited_memory.py" --start 2026-05-25 --days 3 >> "$LOG" 2>&1
log "=== STEP 5 보고서 ==="
LLM_MODE=qwen36_35b_a3b_awq python3 "$WD/scripts/sim/generate_final_report.py" --start 2026-05-25 --days 3 --policy-from 2026-05-27 --out "$WD/output/sim/report/FINAL_REPORT_3D_P009.md" --skip-interview >> "$LOG" 2>&1
LLM_MODE=qwen36_35b_a3b_awq python3 "$WD/scripts/sim/generate_final_report.py" --start 2026-05-25 --days 3 --policy-from 2026-05-27 --out "$WD/output/sim/report/FINAL_REPORT_3D_P009_FULL.md" >> "$LOG" 2>&1
python3 "$WD/scripts/sim/analyze_policy_spend.py" --start 2026-05-25 --days 3 --out "$WD/output/sim/report/POLICY_SPEND_ANALYSIS.md" >> "$LOG" 2>&1
python3 "$WD/scripts/sim/export_visualization.py" --start 2026-05-25 --days 3 >> "$LOG" 2>&1
python3 "$WD/scripts/sim/build_standalone_html.py" >> "$LOG" 2>&1
log "=== ALL DONE Qwen3.6 ==="
INNER
chmod +x /tmp/sim_resume_qwen36.sh
nohup bash /tmp/sim_resume_qwen36.sh > "$WD/logs/auto_pipeline_v9_simonly.out" 2>&1 &
log "sim PID=$!"

log "=== auto_swap_qwen36 종료 — 35B로 계속 진행 (복귀 안 함) ==="

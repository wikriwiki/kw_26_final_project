#!/bin/bash
# Qwen3.5-9B-AWQ가 multimodal로 확인 → 폐기.
# 1순위: Qwen/Qwen3-8B-AWQ (공식, text-only)
# Fallback: Qwen/Qwen3-8B (BF16, 현재 사용 중)
set -u

WD="/g/내 드라이브/Kw/final_project"
cd "$WD"
LOG="$WD/logs/auto_model_swap.log"
mkdir -p "$WD/logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
notify() { log "PUSH: $*"; }

log "=== auto_model_swap v2 시작 (PID=$$) ==="

# STEP 1: Qwen3-8B-AWQ 다운로드 완료 대기 (30분)
HF_DIR="/c/Users/Administrator/.cache/huggingface/hub/models--Qwen--Qwen3-8B-AWQ"
DOWNLOAD_TIMEOUT=$((SECONDS + 1800))
while true; do
    if [ "$SECONDS" -gt "$DOWNLOAD_TIMEOUT" ]; then
        notify "WARN: Qwen3-8B-AWQ 30분 초과 — 8B BF16 유지 결정"
        FORCE_FALLBACK=1
        break
    fi
    if [ ! -d "$HF_DIR" ]; then
        log "STEP 1: 다운로드 디렉토리 미생성"
        sleep 30
        continue
    fi
    INCOMPLETE=$(find "$HF_DIR/blobs" -name "*.incomplete" 2>/dev/null | wc -l)
    SIZE_KB=$(du -s "$HF_DIR" 2>/dev/null | awk '{print $1}')
    SIZE_GB=$(awk -v s=$SIZE_KB 'BEGIN{printf "%.2f", s/1024/1024}')
    if [ "$INCOMPLETE" = "0" ] && [ -n "$SIZE_KB" ]; then
        # AWQ 8B는 ~5GB
        IS_DONE=$(awk -v s=$SIZE_GB 'BEGIN{print (s >= 4.5) ? 1 : 0}')
        if [ "$IS_DONE" = "1" ]; then
            log "STEP 1 완료: Qwen3-8B-AWQ ${SIZE_GB}GB"
            break
        fi
    fi
    log "STEP 1: Qwen3-8B-AWQ ${SIZE_GB}GB, incomplete=$INCOMPLETE"
    sleep 30
done

# STEP 2: Day 2 시작 직전 잡기
DAY2_JSONL="/c/Users/Administrator/sim_output/metrics/day_2026-05-26.jsonl"
while true; do
    if [ -f "$DAY2_JSONL" ]; then
        log "STEP 2 완료: Day2 jsonl 감지"
        break
    fi
    log "STEP 2 대기: Day2 jsonl 없음"
    sleep 60
done

# STEP 3: sim + vLLM kill
log "STEP 3: sim + vLLM kill"
ps -ef 2>/dev/null | grep -E "auto_pipeline_v9_simonly|auto_pipeline_v9_qwen35" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
ps -ef 2>/dev/null | grep -E "run_simulation.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
taskkill.exe //F //IM python3.13.exe 2>&1 | tail -3
taskkill.exe //F //IM python.exe 2>&1 | tail -3
sleep 8
rm -f /tmp/sim_v9.lock

if [ -n "${FORCE_FALLBACK:-}" ]; then
    log "FORCE_FALLBACK 활성 → 8B BF16로 바로"
    LLM_MODE_OVERRIDE=qwen8b
    VLLM_SCRIPT="$WD/scripts/serve/run_vllm.sh"
    EXPECTED="Qwen/Qwen3-8B"
else
    LLM_MODE_OVERRIDE=qwen3_8b_awq
    VLLM_SCRIPT="$WD/scripts/serve/run_vllm_qwen3_8b_awq.sh"
    EXPECTED="Qwen/Qwen3-8B-AWQ"
fi

# STEP 4: vLLM 시작
log "STEP 4: vLLM $EXPECTED 시작"
nohup bash -c "
source ~/.bashrc 2>/dev/null
source /c/Users/Administrator/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate vllm 2>/dev/null
bash $VLLM_SCRIPT
" > "$WD/logs/vllm_new.log" 2>&1 &
log "vLLM bash PID=$!"

# STEP 5: 준비 대기 (15분)
log "STEP 5: vLLM 준비 대기 (1순위 $EXPECTED)"
WAIT_TIMEOUT=$((SECONDS + 900))
while true; do
    if [ "$SECONDS" -gt "$WAIT_TIMEOUT" ]; then
        notify "WARN: $EXPECTED 15분 초과. 옵션 재시도 (max-len 8192)"
        # 1차 옵션 재시도: max-len 줄임
        ps -ef 2>/dev/null | grep -E "vllm" | grep -v grep | awk '{print $2}' | xargs -r kill -9
        taskkill.exe //F //IM python3.13.exe 2>&1 | tail -3
        sleep 10
        nohup bash -c "
source ~/.bashrc 2>/dev/null
source /c/Users/Administrator/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate vllm 2>/dev/null
vllm serve $EXPECTED \
    --quantization awq_marlin \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --port 8000 \
    --enable-prefix-caching \
    --trust-remote-code
" > "$WD/logs/vllm_retry.log" 2>&1 &
        R_TIMEOUT=$((SECONDS + 600))
        RETRY_OK=0
        while [ "$SECONDS" -lt "$R_TIMEOUT" ]; do
            R_MODEL=$(curl -sS -m 5 http://localhost:8000/v1/models 2>/dev/null | python3 -c "
import sys, json
try: print(json.loads(sys.stdin.read())['data'][0]['id'])
except: pass
" 2>/dev/null)
            if [ "$R_MODEL" = "$EXPECTED" ]; then
                log "Retry OK: $EXPECTED"
                RETRY_OK=1
                break
            fi
            sleep 10
        done
        if [ "$RETRY_OK" = "0" ]; then
            notify "FATAL: $EXPECTED 재시도 실패. 8B BF16 복귀."
            ps -ef 2>/dev/null | grep -E "vllm" | grep -v grep | awk '{print $2}' | xargs -r kill -9
            taskkill.exe //F //IM python3.13.exe 2>&1 | tail -3
            sleep 10
            nohup bash -c "
source ~/.bashrc 2>/dev/null
source /c/Users/Administrator/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate vllm 2>/dev/null
bash $WD/scripts/serve/run_vllm.sh
" > "$WD/logs/vllm_fallback_final.log" 2>&1 &
            sleep 180
            LLM_MODE_OVERRIDE=qwen8b
        fi
        break
    fi
    MODEL=$(curl -sS -m 5 http://localhost:8000/v1/models 2>/dev/null | python3 -c "
import sys, json
try: print(json.loads(sys.stdin.read())['data'][0]['id'])
except: pass
" 2>/dev/null)
    if [ "$MODEL" = "$EXPECTED" ]; then
        log "vLLM 준비 완료: $MODEL"
        break
    elif [ -n "$MODEL" ]; then
        log "vLLM 모델: $MODEL (대기 — 기대값 $EXPECTED)"
    fi
    sleep 10
done

# STEP 6: sim 재시작 (workers=80, fallback 시 60)
WORKERS=80
if [ "$LLM_MODE_OVERRIDE" = "qwen8b" ]; then
    WORKERS=48
fi
log "STEP 6: sim 재시작 (mode=$LLM_MODE_OVERRIDE, workers=$WORKERS)"

# auto_pipeline_v9_qwen35.sh의 --workers를 동적으로 변경하기 위해 inline 새 wrapper
cat > /tmp/sim_resume_inline.sh <<EOF
#!/bin/bash
set -u
LOCK="/tmp/sim_v9.lock"
if [ -f "\$LOCK" ]; then
    OWNER=\$(cat "\$LOCK" 2>/dev/null)
    if [ -n "\$OWNER" ] && kill -0 "\$OWNER" 2>/dev/null; then
        echo "already running: \$OWNER" >&2; exit 1
    fi
    rm -f "\$LOCK"
fi
echo \$\$ > "\$LOCK"
trap 'rm -f "\$LOCK"' EXIT

WD="$WD"
cd "\$WD"
LOG="\$WD/logs/auto_pipeline_v9_simonly.log"
log() { echo "[\$(date '+%Y-%m-%d %H:%M:%S')] \$*" | tee -a "\$LOG"; }
log "=== START $LLM_MODE_OVERRIDE workers=$WORKERS (PID=\$\$) ==="

export LLM_MODE=$LLM_MODE_OVERRIDE
export PYTHONUNBUFFERED=1
ATTEMPT=0; MAX=10
while [ "\$ATTEMPT" -lt "\$MAX" ]; do
    ATTEMPT=\$((ATTEMPT+1))
    log "attempt \$ATTEMPT"
    SIMLOG="\$WD/logs/sim_v9_${LLM_MODE_OVERRIDE}_attempt\${ATTEMPT}.log"
    python3 -u "\$WD/scripts/sim/run_simulation.py" --start 2026-05-25 --days 3 --workers $WORKERS > "\$SIMLOG" 2>&1
    RC=\$?
    log "exit=\$RC"
    LAST="/c/Users/Administrator/sim_output/metrics/day_2026-05-27.jsonl"
    if [ -f "\$LAST" ] && [ "\$(wc -l < "\$LAST")" -ge 14560 ]; then
        log "5/27 완료"; break
    fi
    if [ "\$RC" -eq 0 ]; then log "exit=0 incomplete"; break; fi
    log "resume in 30s"; sleep 30
done

log "=== STEP 4 visited Memory ==="
python3 "\$WD/scripts/sim/recover_visited_memory.py" --start 2026-05-25 --days 3 >> "\$LOG" 2>&1

log "=== STEP 5 보고서 ==="
LLM_MODE=$LLM_MODE_OVERRIDE python3 "\$WD/scripts/sim/generate_final_report.py" --start 2026-05-25 --days 3 --policy-from 2026-05-27 --out "\$WD/output/sim/report/FINAL_REPORT_3D_P009.md" --skip-interview >> "\$LOG" 2>&1
LLM_MODE=$LLM_MODE_OVERRIDE python3 "\$WD/scripts/sim/generate_final_report.py" --start 2026-05-25 --days 3 --policy-from 2026-05-27 --out "\$WD/output/sim/report/FINAL_REPORT_3D_P009_FULL.md" >> "\$LOG" 2>&1
python3 "\$WD/scripts/sim/analyze_policy_spend.py" --start 2026-05-25 --days 3 --out "\$WD/output/sim/report/POLICY_SPEND_ANALYSIS.md" >> "\$LOG" 2>&1
python3 "\$WD/scripts/sim/export_visualization.py" --start 2026-05-25 --days 3 >> "\$LOG" 2>&1
python3 "\$WD/scripts/sim/build_standalone_html.py" >> "\$LOG" 2>&1
log "=== ALL DONE ==="
EOF
chmod +x /tmp/sim_resume_inline.sh
nohup bash /tmp/sim_resume_inline.sh > "$WD/logs/auto_pipeline_v9_simonly.out" 2>&1 &
log "sim 재시작 PID=$!"

notify "=== Model swap 완료 → $LLM_MODE_OVERRIDE workers=$WORKERS ==="
log "=== auto_model_swap 종료 ==="

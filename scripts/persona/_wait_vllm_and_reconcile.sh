#!/bin/bash
# EXAONE 4.5 vLLM 준비 대기 → 7,500 + 30,000 페르소나 LLM 봉합 자동 실행

cd "/g/내 드라이브/Kw/final_project"

LOG="/c/Users/Administrator/sim_output_9d/persona_reconcile.log"
mkdir -p "$(dirname $LOG)"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== auto reconcile 시작 (PID=$$) ==="

# Step 1: vLLM 응답 대기 (모델 다운로드 + 로드 ~25분)
log "Step 1: vLLM HTTP 200 대기..."
W_TIMEOUT=$((SECONDS + 3600))
while [ "$SECONDS" -lt "$W_TIMEOUT" ]; do
    M=$(curl -sS -m 5 http://localhost:8000/v1/models 2>/dev/null | python3 -c "
import sys, json
try: print(json.loads(sys.stdin.read())['data'][0]['id'])
except: pass
" 2>/dev/null)
    if [ -n "$M" ]; then
        log "✅ vLLM 준비 완료: $M"
        break
    fi
    sleep 30
done

# Step 2: 응답 형식 sanity check (thinking off 작동 확인)
log "Step 2: chat_template_kwargs 검증..."
RESP=$(curl -sS -m 30 -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"LGAI-EXAONE/EXAONE-4.0-32B-AWQ","messages":[{"role":"user","content":"Reply with exactly: {\"ok\":true}"}],"max_tokens":30,"chat_template_kwargs":{"enable_thinking":false}}' 2>&1)
log "vLLM 응답 샘플: $(echo $RESP | head -c 300)"

# Step 3: 7,500 reconcile
log "Step 3: 7,500 페르소나 LLM 봉합 시작..."
LLM_MODE=exaone40_32b_awq python3 -u -m scripts.persona.build_rank_coupling \
    --limit 7500 --seed 42 \
    --llm-reconcile \
    --llm-mode exaone40_32b_awq \
    --out output/personas/full_7500_exaone40/A_rank_coupling_bdc_nvidia_v2.jsonl \
    --jsonl 2>&1 | tee -a "$LOG"

if [ -f output/personas/full_7500_exaone40/A_rank_coupling_bdc_nvidia_v2.jsonl ]; then
    log "✅ 7,500 봉합 완료: $(wc -l < output/personas/full_7500_exaone40/A_rank_coupling_bdc_nvidia_v2.jsonl) lines"
else
    log "⚠️ 7,500 봉합 fail"
fi

# Step 4: 30,000 reconcile
log "Step 4: 30,000 페르소나 LLM 봉합 시작..."
LLM_MODE=exaone40_32b_awq python3 -u -m scripts.persona.build_rank_coupling \
    --limit 30000 --seed 42 --multiplier 3 \
    --llm-reconcile \
    --llm-mode exaone40_32b_awq \
    --out output/personas/full_30000_exaone40/A_rank_coupling_bdc_nvidia_v2.jsonl \
    --jsonl 2>&1 | tee -a "$LOG"

if [ -f output/personas/full_30000_exaone40/A_rank_coupling_bdc_nvidia_v2.jsonl ]; then
    log "✅ 30,000 봉합 완료: $(wc -l < output/personas/full_30000_exaone40/A_rank_coupling_bdc_nvidia_v2.jsonl) lines"
else
    log "⚠️ 30,000 봉합 fail"
fi

log "=== DONE ==="

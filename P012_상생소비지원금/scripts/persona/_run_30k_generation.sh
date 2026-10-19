#!/bin/bash
# 30,000 페르소나 LLM 봉합 생성 — EXAONE-4.0-32B-AWQ (Combo A)
# 출력: C: 드라이브 (Drive Stream 손상 회피용)

cd "/g/내 드라이브/Kw/final_project"

# 출력 경로: C: 드라이브 (로컬 파일시스템, Drive Stream 우회)
OUT_DIR="/c/Users/Administrator/personas_30000_exaone40"
OUT_FILE="$OUT_DIR/A_rank_coupling_bdc_nvidia_v2.jsonl"
LOG="$OUT_DIR/build.log"
mkdir -p "$OUT_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

T0=$(date +%s)
log "===== 30,000 페르소나 LLM 봉합 시작 (EXAONE-4.0-32B-AWQ) ====="

# vLLM HTTP 응답 대기
log "[Step 1] vLLM HTTP 응답 대기"
until curl -sS -m 5 http://localhost:8000/v1/models 2>/dev/null | grep -q "EXAONE"; do
    sleep 30
done
log "vLLM 준비 완료"

# 응답 형식 sanity check
log "[Step 2] chat_template_kwargs 검증"
RESP=$(curl -sS -m 30 -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"LGAI-EXAONE/EXAONE-4.0-32B-AWQ","messages":[{"role":"user","content":"Reply with exactly: {\"ok\":true}"}],"max_tokens":30,"chat_template_kwargs":{"enable_thinking":false}}' 2>&1)
log "vLLM 응답: $(echo $RESP | head -c 200)"

# 30,000 reconcile
log "[Step 3] 30,000 페르소나 build_rank_coupling --multiplier 3 시작"
NVIDIA_PERSONA_PATH="C:/Users/Administrator/personas_30000_exaone40/nvidia_seoul_full.jsonl" \
LLM_MODE=exaone40_32b_awq '/c/ProgramData/Anaconda3/python.exe' -u -m scripts.persona.build_rank_coupling \
    --limit 30000 --seed 42 --multiplier 3 \
    --llm-reconcile \
    --llm-mode exaone40_32b_awq \
    --out "$OUT_FILE" \
    --jsonl 2>&1 | tee -a "$LOG"

T1=$(date +%s)
ELAPSED=$((T1 - T0))

if [ -f "$OUT_FILE" ]; then
    LINES=$(wc -l < "$OUT_FILE")
    SIZE=$(ls -la "$OUT_FILE" | awk '{print $5}')
    log "✅ 30,000 봉합 완료: $LINES lines, ${SIZE} bytes"
    log "총 소요: ${ELAPSED}s = $((ELAPSED/3600))h $(((ELAPSED%3600)/60))m"
else
    log "⚠️ 30,000 봉합 fail"
fi

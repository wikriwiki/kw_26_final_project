#!/usr/bin/env bash
# SGLang 재기동 — KV 캐시 확대판.
# 기존 자동 산정치 mem_fraction_static=0.634 는 80GB 카드에서 58GB만 쓰고 22GB를 놀렸다.
# 0.88 로 올려 남는 메모리를 KV 풀로 돌린다(30만 → 약 45만 토큰 기대).
set -u
source /data/venv_sgl/bin/activate
export NCCL_CUMEM_ENABLE=1
export PYTHONUNBUFFERED=1
exec python -m sglang.launch_server \
  --model-path LGAI-EXAONE/EXAONE-4.5-33B-AWQ \
  --port 8000 --host 0.0.0.0 \
  --tp-size 2 \
  --attention-backend triton \
  --trust-remote-code \
  --mem-fraction-static 0.88 \
  >> /data/sglang.log 2>&1

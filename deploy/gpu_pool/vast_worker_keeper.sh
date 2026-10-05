#!/usr/bin/env bash
# Keeps the inference-only SGLang server and its reverse tunnel to the experiment box alive.
# The model command is the experiment server's recorded argv (server-config.json), unchanged.
cd /workspace/gpu-worker
exec 9>/workspace/gpu-worker/keeper.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) keeper already running"; exit 0; }
export PATH=/workspace/venv_sgl/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
ARGS=(--model-path LGAI-EXAONE/EXAONE-4.5-33B-AWQ --host 127.0.0.1 --port 8000
      --served-model-name LGAI-EXAONE/EXAONE-4.5-33B-AWQ --dtype auto --context-length 16384
      --max-running-requests 16 --tp-size 1 --mem-fraction-static 0.88 --attention-backend triton
      --revision 31e6a965d0661bbe4a8b895e22a77f8271772ba0 --random-seed 42 --trust-remote-code
      --grammar-backend outlines --constrained-json-whitespace-pattern '[\n\t ]*')
healthy() { curl -s -o /dev/null -m 3 -w '%{http_code}' http://127.0.0.1:8000/health | grep -q 200; }
tunnel_loop() {
  while true; do
    ssh -N -i /root/.ssh/no_smoking_gpu_pool -p ${TUNNEL_PORT:-22} -o StrictHostKeyChecking=yes \
        -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
        -o HostKeyAlias=[95.3.33.46]:45025 -R 127.0.0.1:18003:127.0.0.1:8000 root@${TUNNEL_HOST:-172.17.0.8}
    echo "$(date -u +%FT%TZ) tunnel exited $?"; sleep 5
  done
}
tunnel_loop >> tunnel.log 2>&1 9>&- &
while true; do
  if ! pgrep -f "sglang.launch_server.*--port 8000" >/dev/null; then
    echo "$(date -u +%FT%TZ) starting sglang"
    setsid /workspace/venv_sgl/bin/python -m sglang.launch_server "${ARGS[@]}" >> sglang.log 2>&1 < /dev/null 9>&- &
  fi
  sleep 30
done

#!/usr/bin/env bash
# 중계 프로그램이 죽으면 2초 뒤 다시 띄운다. DISABLED 파일이 있으면 멈춘다.
cd /data/gpu_pool
while [[ ! -e /data/gpu_pool/DISABLED ]]; do
  /data/venv/bin/python gpu_pool_proxy.py --config config.json >> proxy.out 2>&1
  echo "[$(date -Is)] proxy exited rc=$?" >> proxy.out
  sleep 2
done

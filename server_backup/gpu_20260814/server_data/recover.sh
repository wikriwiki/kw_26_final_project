#!/usr/bin/env bash
# EXP-001 복구 — 컨테이너 재생성으로 /data(임시)가 소실됨. 이번엔 NFS(/home/ubuntu/data)에
# 설치하고 /data는 심볼릭 링크로 기존 경로 호환을 유지한다 → 다음 재생성에도 생존.
set -uo pipefail
NAS=/home/ubuntu/data
LOG=$NAS/recover.log
exec >>"$LOG" 2>&1
echo "===== 복구 시작 $(date -u +%FT%TZ) ====="

step() { echo "--- [$1] $(date -u +%T) ---"; }

step "디렉토리 + /data 심볼릭"
mkdir -p $NAS/exp001_repo $NAS/exp001 $NAS/models $NAS/neo4j_data
[ -L /data ] || sudo ln -sfn $NAS /data
ls -ld /data && ls -1 /data | head

step "JDK 17"
if ! command -v java >/dev/null; then
  sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openjdk-17-jdk-headless
fi
java -version 2>&1 | head -1

step "Neo4j 5.26"
if [ ! -d $NAS/neo4j-community-5.26.0 ]; then
  cd $NAS && wget -q https://dist.neo4j.org/neo4j-community-5.26.0-unix.tar.gz && tar xzf neo4j-community-5.26.0-unix.tar.gz && rm -f neo4j-community-5.26.0-unix.tar.gz
fi
ls -d $NAS/neo4j-community-5.26.0 && echo "  neo4j 준비"

step "시뮬 venv"
if [ ! -x $NAS/venv/bin/python ]; then
  python3 -m venv $NAS/venv
fi
source $NAS/venv/bin/activate
pip install -q -U pip
pip install -q neo4j pydantic requests python-dateutil pandas numpy huggingface_hub
python -c "import neo4j,pydantic,requests,pandas,numpy;print('  sim venv OK')"
deactivate

step "모델 다운로드 (EXAONE-4.5-33B-AWQ)"
source $NAS/venv/bin/activate
export HF_HOME=$NAS/models
python - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download("LGAI-EXAONE/EXAONE-4.5-33B-AWQ", max_workers=8)
print("  model at", p)
PY
deactivate

step "SGLang venv (EXAONE-4.5 포크)"
if [ ! -x $NAS/venv_sgl/bin/python ]; then
  python3 -m venv $NAS/venv_sgl
fi
source $NAS/venv_sgl/bin/activate
pip install -q -U pip
pip install "sglang[all] @ git+https://github.com/lkm2835/sglang.git@add-exaone4_5#subdirectory=python"
pip install -q --no-deps "transformers==5.8.0" "kernels==0.10.0"
python -c "import sglang, transformers, kernels; from transformers.models.auto.configuration_auto import CONFIG_MAPPING; assert 'exaone4_5' in CONFIG_MAPPING; print('  sglang', sglang.__version__, '| transformers', transformers.__version__, '| exaone4_5 OK')"
deactivate

echo "===== 복구 종료 $(date -u +%FT%TZ) ====="

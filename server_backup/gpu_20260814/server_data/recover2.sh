#!/usr/bin/env bash
# 복구 2단계 — venv 재생성(ensurepip 설치 후) + 패키지 + 모델. 실패 시 즉시 중단.
set -euo pipefail
NAS=/home/ubuntu/data
exec >>"$NAS/recover2.log" 2>&1
echo "===== 2단계 시작 $(date -u +%FT%TZ) ====="
step(){ echo "--- [$1] $(date -u +%T) ---"; }

step "시뮬 venv 재생성"
rm -rf $NAS/venv $NAS/venv_test
python3 -m venv $NAS/venv
test -x $NAS/venv/bin/pip || { echo "FATAL: venv pip 없음"; exit 1; }
source $NAS/venv/bin/activate
python -c "import sys; assert sys.prefix.startswith('$NAS/venv'), sys.prefix; print('  venv 활성 OK', sys.prefix)"
pip install -q -U pip
pip install -q neo4j pydantic requests python-dateutil pandas numpy huggingface_hub hf_transfer
python -c "import neo4j,pydantic,requests,pandas,numpy,huggingface_hub;print('  sim venv 패키지 OK')"

step "모델 다운로드 (EXAONE-4.5-33B-AWQ)"
export HF_HOME=$NAS/models HF_HUB_ENABLE_HF_TRANSFER=1
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("LGAI-EXAONE/EXAONE-4.5-33B-AWQ", max_workers=8)
print("  model at", p)
PY
du -sh $NAS/models
deactivate

step "SGLang venv 재생성"
rm -rf $NAS/venv_sgl
python3 -m venv $NAS/venv_sgl
test -x $NAS/venv_sgl/bin/pip || { echo "FATAL: venv_sgl pip 없음"; exit 1; }
source $NAS/venv_sgl/bin/activate
python -c "import sys; assert sys.prefix.startswith('$NAS/venv_sgl'), sys.prefix; print('  venv_sgl 활성 OK')"
pip install -q -U pip
pip install "sglang[all] @ git+https://github.com/lkm2835/sglang.git@add-exaone4_5#subdirectory=python"
pip install -q --no-deps "transformers==5.8.0" "kernels==0.10.0"
python -c "import sglang, transformers, kernels; from transformers.models.auto.configuration_auto import CONFIG_MAPPING; assert 'exaone4_5' in CONFIG_MAPPING; print('  sglang', sglang.__version__, '| transformers', transformers.__version__, '| exaone4_5 OK')"
deactivate

echo "RECOVER2_DONE $(date -u +%FT%TZ)"

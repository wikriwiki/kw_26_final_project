#!/usr/bin/env bash
# Rebuild the exact Vast SGLang environment and fetch the pinned model revision.
set -euo pipefail
cd /workspace/gpu-worker
echo "$(date -u +%FT%TZ) setup start"
sha256sum vast_venv_sgl_freeze.txt
[ -x /workspace/venv_sgl/bin/python ] || /opt/conda/bin/python3 -m venv /workspace/venv_sgl
/workspace/venv_sgl/bin/python -m pip install -q uv
# Vast installed sglang, then upgraded transformers to 5.8.0; the freeze is the exact full set.
/workspace/venv_sgl/bin/python -m uv pip install --no-deps --python /workspace/venv_sgl/bin/python -r vast_venv_sgl_freeze.txt
/workspace/venv_sgl/bin/python -c 'import importlib.metadata as md, torch; print("sglang", md.version("sglang"), "torch", torch.__version__, "transformers", md.version("transformers"), "outlines", md.version("outlines"), torch.cuda.get_device_name(0))'
/workspace/venv_sgl/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('LGAI-EXAONE/EXAONE-4.5-33B-AWQ', revision='31e6a965d0661bbe4a8b895e22a77f8271772ba0'))"
echo "$(date -u +%FT%TZ) setup done"

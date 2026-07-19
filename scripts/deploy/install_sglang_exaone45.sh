#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# EXP-001 SGLang 설치 — EXAONE-4.5-33B-AWQ 서빙용 별도 venv(/data/venv_sgl)
#
# 표준 SGLang은 EXAONE-4.5 아키텍처를 아직 지원하지 않아, add-exaone4_5 포크를
# 설치한다. transformers는 EXAONE-4.5 config를 읽을 수 있는 최신(>=5.8.0)이 필요.
#
# 사용: bash scripts/deploy/install_sglang_exaone45.sh
# 이후: bash scripts/serve/serve_exaone45_sglang_a100x2.sh 로 기동.
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

VENV="${VENV:-/data/venv_sgl}"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q -U pip
# EXAONE-4.5 지원 SGLang 포크 (표준 릴리스에 병합 전)
pip install "sglang[all] @ git+https://github.com/lkm2835/sglang.git@add-exaone4_5#subdirectory=python"
# EXAONE-4.5 config 로드에 필요한 최신 transformers
pip install -U "transformers>=5.8.0" || pip install -U --pre transformers
python -c "import sglang, transformers; print('sglang', sglang.__version__, '| transformers', transformers.__version__)"
echo "SGL_INSTALL_DONE — 다음: bash scripts/serve/serve_exaone45_sglang_a100x2.sh"

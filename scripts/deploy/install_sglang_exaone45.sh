#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# EXP-001 SGLang 설치 — EXAONE-4.5-33B-AWQ 서빙용 별도 venv(/data/venv_sgl)
#
# 표준 SGLang은 EXAONE-4.5 아키텍처를 아직 지원하지 않아, add-exaone4_5 포크를
# 설치한다. 모델 카드가 안내한 포크의 commit과 기존 작동 의존성 조합을 고정한다.
#
# 사용: bash scripts/deploy/install_sglang_exaone45.sh
# 이후: bash scripts/serve/serve_exaone45_sglang_a100x2.sh 로 기동.
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

VENV="${VENV:-/data/venv_sgl}"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q -U pip
# LG 모델 카드의 add-exaone4_5 포크: 2026-05-07 commit, 검토일 2026-09-22.
# 브랜치 이름 대신 commit을 사용하여 재설치 시 코드가 바뀌지 않게 한다.
pip install "sglang[all] @ git+https://github.com/lkm2835/sglang.git@6757c9f904cdb8ae9028a394a2108d079b9e088c#subdirectory=python"
# ── 버전 핀 (2026-07-20 실측으로 확정한 유일 작동 조합) ────────────────────
# SGLang 포크가 고정하는 transformers==5.3.0 은 exaone4_5 아키텍처를 모른다.
# → transformers 를 5.8.0 으로 올려야 EXAONE-4.5 config 를 인식한다.
# 그런데 5.8.0 은 최신 kernels(0.16)의 LayerRepository(version 필수) API 와 충돌해
# 서빙이 import 단계에서 죽는다(hub_kernels 모듈 로드 시 즉시 예외).
# → kernels 를 0.10.0 으로 낮추면 충돌이 사라진다. --no-deps 로 다른 핀을 안 건드린다.
pip install -q --no-deps "transformers==5.8.0" "kernels==0.10.0"
python -c "import sglang, transformers, kernels; from transformers.models.auto.configuration_auto import CONFIG_MAPPING; assert 'exaone4_5' in CONFIG_MAPPING; print('sglang', sglang.__version__, '| transformers', transformers.__version__, '| exaone4_5 OK')"
echo "SGL_INSTALL_DONE — 다음: bash scripts/serve/serve_exaone45_sglang_a100x2.sh"

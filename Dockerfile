# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# 시뮬레이션 앱 이미지 (CI 에서 빌드 → GHCR push)
#  - CPU 전용: LLM 은 외부 vLLM/SGLang(LLM_BASE_URL), Neo4j 는 별도 서비스.
#  - 런타임 의존성만 슬림 설치 (datasets/pandas 등 불필요 패키지 제외).
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

RUN pip install --no-cache-dir \
        "neo4j>=5.20,<7" \
        "openai>=1.54,<2" \
        "pydantic>=2.7,<3" \
        "pyyaml>=6.0" \
        "openpyxl>=3.1" \
        "requests>=2.31"

WORKDIR /app
COPY . /app

ENV SIM_OUTPUT_DIR=/output \
    PYTHONUNBUFFERED=1 \
    LLM_MODE=qwen8b
RUN mkdir -p /output

# ENTRYPOINT=python → 인자로 임의 스크립트 실행 가능.
#   기본(CMD)   : 시뮬레이션
#   그래프 적재 : docker compose run --rm app scripts/neo4j_load/run_all.py
ENTRYPOINT ["python", "-u"]
CMD ["scripts/sim/run_simulation.py", "--start", "2026-05-25", "--days", "3", "--workers", "8"]

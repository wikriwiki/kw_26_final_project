# 서울 상권 소비행동 시뮬레이션 — 프로토타입

> 2026 서울시 빅데이터 활용 경진대회 — 분석 부문

서울시 빅데이터(카드소비, KT 유동인구, 상권발달지수)를 기반으로 소비자 에이전트를 생성하고,
LLM(Qwen3) + 룰 엔진 + GraphRAG를 결합한 하이브리드 ABM 시뮬레이션을 수행합니다.

## 구조

```
prototype/
├── src/          # 시뮬레이션 소스 코드
├── docs/         # 아키텍처 및 설계 문서
└── output/       # 시뮬레이션 결과 (리포트, 지도, 그래프)
```

## 빠른 시작

```bash
cd src

# LLM 없이 룰 기반만 (빠른 테스트)
python simulation.py --no-llm --agents 100 --weeks 4

# Qwen3 사용 (Ollama 실행 필요)
ollama run qwen3:30b
python simulation.py --agents 300 --weeks 24
```

## 데이터 모드

```python
# src/config.py
USE_SYNTHETIC = True   # 합성 데이터 (기본, 별도 데이터 불필요)
USE_SYNTHETIC = False  # 실제 데이터 (빅데이터캠퍼스 풀 데이터 필요)
```

## 의존성 설치

```bash
pip install -r requirements.txt
```

## 출력

| 파일 | 설명 |
|------|------|
| `output/simulation_animation.html` | 에이전트 이동 인터랙티브 지도 |
| `output/reports/week_XX.md` | 주간 소비행동 리포트 |
| `output/knowledge_graph.graphml` | GraphRAG 지식그래프 |

## 문서

| 문서 | 내용 |
|------|------|
| `docs/01_architecture.md` | 시스템 아키텍처 + Hybrid D 엔진 |
| `docs/02_data.md` | ETL 파이프라인 + 데이터 이슈 |
| `docs/03_agents.md` | 에이전트 시스템 + GraphRAG + LLM |
| `docs/04_roadmap_v2.md` | v2 확장 계획 |

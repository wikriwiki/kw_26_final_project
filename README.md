# 서울 상권 소비행동 시뮬레이션 — KW 2026

> 2026 서울시 빅데이터 활용 경진대회 — 분석 부문

LLM(vLLM/SGLang) + 룰 엔진 + Neo4j/Graphiti 기반 하이브리드 ABM 시뮬레이션.

---

## 📁 디렉토리 구조 (팀 코딩 가이드)

**새 코드를 작성하기 전에 반드시 해당 폴더의 `README.md`를 먼저 읽으세요.**
각 폴더는 명확한 책임 경계를 가지며, 폴더를 잘못 선택하면 PR 리뷰에서 위치 변경을 요청합니다.

```
.
├── prototype/              # 기존 프로토타입 (참고용, 점진 이관)
│
├── src/                    # 신규 아키텍처 소스 코드
│   ├── core/               # 도메인 모델 (Pydantic) — 비즈니스 로직 없음
│   ├── phases/
│   │   ├── dawn/           # Phase 1: 자정에 Day t Plan 생성
│   │   └── night/          # Phase 2: 하루 마감 후 상호작용 & 메모리
│   ├── policy_pipeline/    # Async 정책 주입 (Watchdog 기반)
│   ├── infra/
│   │   ├── llm/            # vLLM/SGLang 엔진, 배치 컨트롤러
│   │   └── cache/          # Prompt Prefix Cache, Community Summary
│   ├── graph/
│   │   ├── migrations/     # Neo4j 스키마 .cypher 파일
│   │   └── queries/        # 도메인별 Cypher 쿼리 모듈
│   └── prompts/            # Jinja2 LLM 프롬프트 템플릿
│
├── data/
│   ├── policies/inbox/     # 새 정책 파일 드랍 위치
│   ├── policies/processed/ # 처리 완료 정책 보관
│   └── seed/               # 행정동/POI/Persona 초기 데이터
│
├── tests/
│   ├── unit/               # 외부 의존성 없는 단위 테스트
│   ├── integration/        # Neo4j/LLM 포함 테스트
│   └── fixtures/           # 샘플 데이터
│
├── scripts/                # 운영 스크립트 (init_neo4j, seed_data 등)
└── output/                 # 런타임 산출물 (gitignored)
    ├── logs/
    └── plans/
```

---

## 🚦 팀 작업 규칙

### 1. 새 파일 추가 전 README 읽기
모든 1차 폴더에 README가 있습니다. **무슨 파일이 어디로 가야 하는지 정의되어 있습니다.**
규칙을 어긴 PR은 위치 변경 후 다시 받습니다.

### 2. 의존 방향 (위 → 아래만 허용)
```
phases / policy_pipeline   ←  비즈니스 로직 (LLM 호출 OK)
        ↓
graph / infra / prompts    ←  인프라 레이어 (도메인 모름)
        ↓
core                       ←  순수 도메인 모델
```
- `core/`는 어떤 것도 import 하지 않음
- `infra/`, `graph/`는 `core/`만 import
- `phases/`, `policy_pipeline/`은 `core/`, `infra/`, `graph/`, `prompts/` 모두 import 가능

### 3. 직접 호출 금지
| 대상 | 금지 | 사용해야 할 것 |
|------|------|----------------|
| Neo4j 드라이버 직접 호출 | `neo4j.GraphDatabase.driver(...)` | `src/graph/queries/*.py` |
| LLM HTTP 직접 호출 | `httpx.post("http://vllm/...")` | `src/infra/llm/engine_client.py` |
| 프롬프트 문자열 하드코딩 | `f"You are an agent... {persona}"` | `src/prompts/*.jinja2` |

### 4. 커밋/PR 규칙
- PR 제목: `<scope>: <한 줄 요약>` — scope는 폴더명 (`dawn:`, `graph:`, `policy_pipeline:` 등)
- 한 PR은 한 폴더에 집중. 여러 폴더가 변하면 책임이 섞이고 있다는 신호
- 새 의존성 추가는 별도 PR로

---

## 🛠 개발 시작

```bash
pip install -r requirements.txt

# Neo4j 초기화 (최초 1회)
python -m scripts.init_neo4j

# 시드 데이터 로딩
python -m scripts.seed_data

# 시뮬레이션 실행
python -m src.cli --day 1
```

---

## 📚 추가 문서

- `prototype/docs/01_architecture.md` — Hybrid 시뮬레이션 엔진 (기존 프로토타입 기준)
- `prototype/docs/03_agents.md` — 에이전트 시스템
- `prototype/docs/04_roadmap_v2.md` — v2 확장 계획

---

## Agent Persona Pipeline

서울시 빅데이터를 기반으로 소비자 에이전트 페르소나를 생성하는 파이프라인입니다.  
전체 흐름: **전처리 → 통계 산출 → 에이전트 생성 → 검증**

---

### 파일 설명

#### 1. `preprocess_join.py` — 데이터 전처리 및 조인

원본 CSV 데이터셋들을 읽어 공통 키(행정동코드, 성별, 연령대)로 조인하고, 분석 가능한 형태로 가공합니다.

- **입력**: `data/raw/` 내 원본 CSV (텔레콤 29종 지표, 카드소비, KT 유동인구, 집계구 결제 등)
- **출력**: `output/original/` 또는 `output/synthetic/`
  - `joined_persona_base.csv` — (행정동, 성별, 연령대) 기준 인구통계 + 텔레콤 + 소비 + 유동인구 통합 테이블
  - `joined_dong_context.csv` — 행정동 단위 상권 환경 데이터 (상권지수, 유입비율 등)
- **주요 기능**: 행정동코드 매핑, 성별·연령대 정규화, 가중평균 집계, Left Join

```bash
python preprocess_join.py              # original + synthetic 모두
python preprocess_join.py original     # 원본 데이터만
```

---

#### 2. `analyze_stats.py` — 통계 산출 (에이전트 생성 입력)

전처리된 데이터를 분석하여 LLM 에이전트 생성에 필요한 통계 JSON 파일들을 생성합니다.

- **입력**: `output/synthetic/` (또는 `output/original/`) 내 조인된 CSV + `data/raw/` 원본
- **출력**: `output/stats/` (총 7개 JSON)
  - `agent_profiles.json` — 그룹별 소비수준(10분위), 이동활발도, 업종 소비비율
  - `dong_context.json` — 행정동별 상권 환경 지표
  - `workplace_flow.json` — 거주지→직장 이동 확률분포
  - `workplace_population.json` — 행정동별 직장인구 (성별×연령대)
  - `consumption_detail.json` — 평일/주말별 업종 소비비중
  - `global_distributions.json` — 서울 전체 소비/이동 패턴
  - `agent_allocation.json` — 그룹별 에이전트 할당 수량
- **주요 파라미터**: `TARGET_AGENTS = 15000` (생성할 총 에이전트 수)

```bash
python analyze_stats.py                # synthetic 데이터 기준
python analyze_stats.py --source original
```

---

#### 3. `generate_agents.py` — LLM 기반 에이전트 생성

vLLM 서버(Qwen3-32B-AWQ)를 호출하여 통계 기반의 소비자 에이전트 페르소나를 대량 생성합니다.

- **입력**: `output/stats/` 내 통계 JSON 파일들
- **출력**: `output/agents/agents_final.json`
- **사전 조건**: WSL에서 vLLM 서버가 실행 중이어야 함
- **주요 기능**:
  - 그룹별(행정동×성별×연령대) 통계를 프롬프트로 구성
  - 비동기 병렬 요청으로 대량 생성
  - 중단 후 `--resume`으로 이어서 생성 가능
  - 에이전트 스키마: 거주지, 인적사항, 직장, 소비패턴, 행동지표, 성격

```bash
python generate_agents.py --limit 5            # 시범 생성
python generate_agents.py --max-concurrent 8   # 전체 생성
python generate_agents.py --resume             # 중단 후 재개
```

---

#### 4. `validate_vs_raw.py` — 에이전트 검증

생성된 에이전트의 분포가 원본 데이터의 통계와 얼마나 일치하는지 검증합니다.

- **입력**: `output/agents/agents_final.json` + `data/raw/telecom_29.csv` + `output/stats/*.json`
- **출력**: 콘솔 검증 리포트
- **검증 항목**:
  1. 텔레콤 지표 비교 — 출근시간, 배달일수, 이동거리 등 raw 평균 vs 에이전트 평균
  2. 성별×연령대 인구 분포 비교
  3. 자치구별 분포 비교
  4. 통계 평균/표준편차 vs 에이전트 분포 비교
  5. 전체 요약 (커버리지, 직업 다양성, 성비 등)

```bash
python validate_vs_raw.py
python validate_vs_raw.py --agents output/agents/agents_final.json
```

---

### 파이프라인 실행 순서

```
1. preprocess_join.py   →  원본 데이터 전처리 및 조인
2. analyze_stats.py     →  통계 JSON 생성
3. generate_agents.py   →  vLLM으로 에이전트 대량 생성
4. validate_vs_raw.py   →  생성 결과 검증
```

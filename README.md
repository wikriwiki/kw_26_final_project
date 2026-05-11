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

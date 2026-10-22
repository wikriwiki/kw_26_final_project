# Agent Persona Pipeline

서울시 빅데이터를 기반으로 소비자 에이전트 페르소나를 생성하는 파이프라인입니다.
전체 흐름: **전처리 → 통계 산출 → 에이전트 생성 → 검증**

> 모든 명령어는 **프로젝트 루트 디렉토리**에서 실행한다고 가정합니다.
> 각 스크립트는 헤더에 `PROJECT_ROOT = Path(__file__).resolve().parents[2]` 가
> 정의돼 있어 어느 cwd에서 호출해도 `data/`, `output/` 경로가 정확히 해석됩니다.

---

## 파일 설명

### 1. `preprocess_join.py` — 데이터 전처리 및 조인

원본 CSV 데이터셋들을 읽어 공통 키(행정동코드, 성별, 연령대)로 조인하고, 분석 가능한 형태로 가공합니다.

- **입력**: `data/raw/` 내 원본 CSV (텔레콤 29종 지표, 카드소비, KT 유동인구, 집계구 결제 등)
- **출력**: `output/original/` 또는 `output/synthetic/`
  - `joined_persona_base.csv` — (행정동, 성별, 연령대) 기준 인구통계 + 텔레콤 + 소비 + 유동인구 통합 테이블
  - `joined_dong_context.csv` — 행정동 단위 상권 환경 데이터 (상권지수, 유입비율 등)
- **주요 기능**: 행정동코드 매핑, 성별·연령대 정규화, 가중평균 집계, Left Join

```bash
python scripts/bdc/preprocess_join.py              # original + synthetic 모두
python scripts/bdc/preprocess_join.py original     # 원본 데이터만
```

---

### 2. `analyze_stats.py` — 통계 산출 (에이전트 생성 입력)

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
python scripts/bdc/analyze_stats.py                # synthetic 데이터 기준
python scripts/bdc/analyze_stats.py --source original
```

---

### 3. `generate_agents.py` — LLM 기반 에이전트 생성

vLLM 서버(Qwen3-32B-AWQ)를 호출하여 통계 기반의 소비자 에이전트 페르소나를 대량 생성합니다.

- **입력**: `output/stats/` 내 통계 JSON 파일들
- **출력**: `output/agents/agents_final.json`
- **사전 조건**: WSL에서 vLLM 서버가 실행 중이어야 함 (`scripts/serve/serve_qwen32b.sh`)
- **주요 기능**:
  - 그룹별(행정동×성별×연령대) 통계를 프롬프트로 구성
  - 비동기 병렬 요청으로 대량 생성
  - 중단 후 `--resume`으로 이어서 생성 가능
  - 에이전트 스키마: 거주지, 인적사항, 직장, 소비패턴, 행동지표, 성격

```bash
python scripts/bdc/generate_agents.py --limit 5            # 시범 생성
python scripts/bdc/generate_agents.py --max-concurrent 8   # 전체 생성
python scripts/bdc/generate_agents.py --resume             # 중단 후 재개
```

---

### 4. `validate_vs_raw.py` — 에이전트 검증

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
python scripts/bdc/validate_vs_raw.py
python scripts/bdc/validate_vs_raw.py --agents output/agents/agents_final.json
```

---

## 파이프라인 실행 순서

```
1. scripts/bdc/preprocess_join.py   →  원본 데이터 전처리 및 조인
2. scripts/bdc/analyze_stats.py     →  통계 JSON 생성
3. scripts/bdc/generate_agents.py   →  vLLM으로 에이전트 대량 생성
4. scripts/bdc/validate_vs_raw.py   →  생성 결과 검증
```

---

## 보조 스크립트 (로컬 전용, .gitignore)

이 폴더에는 위 4개 메인 파이프라인 외에 다음 보조 스크립트들도 함께 있습니다.
.gitignore에 들어있어 트래킹되지 않으며, BDC 데이터를 다시 다룰 때 필요에 따라
사용합니다.

| 파일 | 역할 |
|---|---|
| `file_discovery.py` | data/raw/ 내 압축·CSV 자동 탐지 (FileEntry 추상화). preprocess/analyze가 import |
| `patch_failed_joins.py` | preprocess_join에서 실패한 행 사후 보정 |
| `synthetic_generator.py` | BDC 미반입 시 사용할 합성 데이터 생성 (data/synthetic/) |
| `compare_stats.py`, `compare_to_json.py` | 통계 vs 에이전트 분포 비교 보조 |
| `validate_agents.py` | 페르소나 스키마·필수필드 검증 |
| `validate_pandas_analyze.py`, `validate_pandas_workers.py` | preprocess/analyze 결과 무결성 검증 |
| `quick_validate.py` | 가벼운 sanity check (CLI: agents JSON 경로 인자) |
| `inspect_personas.py` | Nemotron 페르소나 raw 데이터 확인 (옛 참조용) |
| `count_seoul.py` | Nemotron parquet 내 서울 거주자 수 카운트 |
| `assign_income_bucket.py` | Nemotron 페르소나의 소득 버킷 배정 (Nemotron 통합 폐기됨, 보존만) |
| `extract_ksco_codes.py`, `match_occupation_to_ksco.py` | KSCO 직업분류 코드표 추출·매칭 (BDC job 정규화 reference용) |

# Agent Persona Pipeline

서울시 빅데이터를 기반으로 소비자 에이전트 페르소나를 생성하는 파이프라인입니다.  
전체 흐름: **전처리 → 통계 산출 → 에이전트 생성 → 검증**

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
python preprocess_join.py              # original + synthetic 모두
python preprocess_join.py original     # 원본 데이터만
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
python analyze_stats.py                # synthetic 데이터 기준
python analyze_stats.py --source original
```

---

### 3. `generate_agents.py` — LLM 기반 에이전트 생성

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
python validate_vs_raw.py
python validate_vs_raw.py --agents output/agents/agents_final.json
```

---

## 파이프라인 실행 순서

```
1. preprocess_join.py   →  원본 데이터 전처리 및 조인
2. analyze_stats.py     →  통계 JSON 생성
3. generate_agents.py   →  vLLM으로 에이전트 대량 생성
4. validate_vs_raw.py   →  생성 결과 검증
```

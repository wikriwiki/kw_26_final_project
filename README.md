# Agent Persona Pipeline

서울시 빅데이터를 기반으로 소비자 에이전트 페르소나를 생성하는 파이프라인입니다.
전체 흐름: **전처리 → 통계 산출 → 에이전트 생성 → 검증**

> 🔧 **이번 브랜치(`feat/sglang-migration`) 변경 요약**
> - LLM 백엔드: **vLLM → SGLang** (OpenAI 호환 API)
> - **3-way 모델 선택**: `qwen32b`(기본/AWQ) · `qwen9b`(개발용) · `exaone`(국내 대회용 FP8)
> - 프롬프트를 **5-layer 구조**로 재배치하여 SGLang RadixAttention prefix cache 적중률 극대화
> - `generate_agents.py` 안의 모놀리식 프롬프트 빌더 → `prompt_layers.py` / `sglang_client.py` 로 분리
>
> 자세한 변경 내역은 §[SGLang 이전 상세](#sglang-이전-상세)를 참고하세요.

---

## 파일 설명

### 1. `preprocess_join.py` — 데이터 전처리 및 조인

원본 CSV 데이터셋들을 읽어 공통 키(행정동코드, 성별, 연령대)로 조인하고, 분석 가능한 형태로 가공합니다.

- **입력**: `data/raw/` 내 원본 CSV (텔레콤 29종 지표, 카드소비, KT 유동인구, 집계구 결제 등)
- **출력**: `output/original/` 또는 `output/synthetic/`
  - `joined_persona_base.csv` — (행정동, 성별, 연령대) 기준 인구통계 + 텔레콤 + 소비 + 유동인구 통합 테이블
  - `joined_dong_context.csv` — 행정동 단위 상권 환경 데이터 (상권지수, 유입비율 등)

```bash
python preprocess_join.py              # original + synthetic 모두
python preprocess_join.py original     # 원본 데이터만
```

---

### 2. `analyze_stats.py` — 통계 산출 (에이전트 생성 입력)

전처리된 데이터를 분석하여 LLM 에이전트 생성에 필요한 통계 JSON 파일들을 생성합니다.

- **입력**: `output/synthetic/` (또는 `output/original/`) 내 조인된 CSV + `data/raw/` 원본
- **출력**: `output/stats/` (총 7개 JSON: `agent_profiles`, `dong_context`, `workplace_flow`,
  `workplace_population`, `consumption_detail`, `global_distributions`, `agent_allocation`)
- **주요 파라미터**: `TARGET_AGENTS = 15000`

```bash
python analyze_stats.py
python analyze_stats.py --source original
```

---

### 3. `generate_agents.py` — SGLang 기반 에이전트 생성

SGLang 서버에 OpenAI 호환 API로 호출해 통계 기반 페르소나를 대량 생성합니다.

- **입력**: `output/stats/` 내 통계 JSON 파일들
- **출력**: `output/agents/agents_final.json` (+ `output/agents/partial/batch_*.json`)
- **사전 조건**: SGLang 서버가 미리 떠 있어야 함 (§[설치 및 구동](#설치-및-구동))
- **신규 보조 모듈**
  - `sglang_client.py` — 모델 레지스트리(`MODELS`) + SGLang 호출 래퍼
  - `prompt_layers.py` — 5-layer 프롬프트 빌더 (캐시 친화 정렬)

```bash
# 가장 작은 단위 확인 (LLM 호출 없이 프롬프트만 출력)
python generate_agents.py --dry-run --model qwen9b --limit 1

# 시범 생성
python generate_agents.py --model qwen9b --limit 20 --max-concurrent 4

# 본 실행
python generate_agents.py --model qwen32b --max-concurrent 16
python generate_agents.py --resume          # 중단 후 재개
```

---

### 4. `validate_vs_raw.py` — 에이전트 검증

생성된 에이전트의 분포가 원본 데이터의 통계와 얼마나 일치하는지 검증합니다.

```bash
python validate_vs_raw.py
python validate_vs_raw.py --agents output/agents/agents_final.json
```

---

## 파이프라인 실행 순서

```
1. preprocess_join.py   →  원본 데이터 전처리 및 조인
2. analyze_stats.py     →  통계 JSON 생성
3. generate_agents.py   →  SGLang 으로 에이전트 대량 생성
4. validate_vs_raw.py   →  생성 결과 검증
```

---

## SGLang 이전 상세

### 왜 SGLang 인가

기존 vLLM은 prefix cache가 **기본 활성이 아니고**, 같은 시스템 프롬프트와 통계 컨텍스트가
수천 번 반복되는 본 워크로드에서 캐시 미적중이 컸습니다. SGLang의 **RadixAttention** 은
요청 간 공통 토큰 prefix를 자동으로 재사용하여, 두 번째 호출부터 prefill 토큰의 상당 부분을
건너뜁니다. 본 파이프라인처럼 *공유되는 통계 + 그룹별 가변 정보* 패턴에 매우 적합합니다.

### 모델 레지스트리 (`sglang_client.MODELS`)

| 모드 키 | HuggingFace ID | 용도 | 비고 |
|---------|----------------|------|------|
| `qwen32b` | `Qwen/Qwen3-32B-AWQ` | 기존 기본값 (대용량 정확도) | AWQ 4-bit, A100 80GB 1장 |
| `qwen9b` | `Qwen/Qwen3.5-9B` | 빠른 개발/디버깅 | 메모리 여유, 빠른 응답 |
| `exaone` | `LGAI-EXAONE/EXAONE-4.5-33B-FP8` | 국내 대회용 | FP8, A100 80GB 1장 |

선택 우선순위: **CLI `--model` > 환경변수 `LLM_MODE` > 기본값 `qwen32b`**.

```bash
python generate_agents.py --model exaone     # CLI 지정
export LLM_MODE=qwen9b && python generate_agents.py   # env 지정
```

Qwen 계열 모델은 chat template의 thinking 모드가 켜진 경우 `<think>...</think>` 토큰을
길게 뱉어 비용/지연이 크므로, `sglang_client.py` 가 자동으로 `enable_thinking=False`를
`extra_body` 로 주입합니다.

### 5-layer 프롬프트 (캐시 핵심)

`prompt_layers.build_layers()` 가 한 그룹의 입력을 다음 순서로 조립합니다 — **공유 범위가
넓은 레이어를 앞에** 둬야 RadixAttention 이 prefix를 재사용할 수 있습니다.

| 층 | 내용 | 공유 범위 |
|----|------|-----------|
| L1 | system: 12-규칙 + 출력 스키마 | 전체 15,000명 |
| L2 | 서울 전체 분포 (weekday/weekend) | 전체 |
| L3 | 동 상권 환경 + 직장 확률분포 | 같은 동의 모든 코호트 (≈12개) |
| L4 | 코호트(성별×연령) 통계 + 업종 비율 | 같은 코호트의 모든 동 |
| L5 | 그룹 고유 (profile, consumption_detail, count) | 호출별 가변 |

추가로 `generate_agents.order_keys_for_cache()` 가 그룹 키를 **(동 → 코호트)** 순으로
정렬하여 호출 시퀀스가 prefix를 자연스럽게 누적 적중하도록 합니다 (한 동 안에서는 L1~L3
prefix가 12회 연속 적중, 다음 동으로 넘어가도 L1~L2는 계속 적중).

캐시 적중률은 SGLang `/metrics` 엔드포인트로 직접 확인할 수 있습니다:

```bash
curl -s http://localhost:30000/metrics | grep -E 'cache_hit|prefill'
```

---

## 설치 및 구동

### 1) 요구 환경

- Python 3.10+
- NVIDIA GPU (A100 80GB 권장) + CUDA 12.x
- 디스크 ~100GB (모델 가중치)

### 2) 의존성 설치 (venv 두 개 권장)

**서버용 venv** (SGLang 본체):
```bash
python -m venv .venv-server
source .venv-server/bin/activate
pip install --upgrade pip
pip install "sglang[all]"
```

**클라이언트용 venv** (`generate_agents.py` 등):
```bash
python -m venv .venv-client
source .venv-client/bin/activate
pip install -r requirements.txt
```

> 두 패키지(SGLang 본체와 OpenAI SDK)는 의존성이 충돌할 수 있으므로 venv 분리를 권장합니다.

### 3) 환경변수

```bash
cp .env.example .env
# .env 편집 후
set -a; source .env; set +a
```

또는 셸에서 직접:
```bash
export LLM_MODE=qwen32b
export SGLANG_BASE_URL=http://localhost:30000/v1
```

### 4) SGLang 서버 기동 (서버용 venv 활성화 상태에서)

```bash
bash scripts/serve_qwen32b.sh     # 기본 (Qwen3-32B-AWQ)
# 또는
bash scripts/serve_qwen9b.sh      # 개발용 (Qwen3.5-9B)
bash scripts/serve_exaone.sh      # 대회용 (EXAONE-4.5-33B-FP8)
```

서버가 떠 있는지 확인:
```bash
curl -s http://localhost:30000/v1/models | python -m json.tool
```

환경변수로 포트/메모리비 조정:
```bash
PORT=30001 MEM_FRAC=0.85 bash scripts/serve_qwen32b.sh
```

### 5) 에이전트 생성 (클라이언트용 venv 활성화 상태에서)

```bash
# 1) 프롬프트 레이어 점검 (LLM 호출 없음)
python generate_agents.py --dry-run --model qwen9b --limit 1

# 2) 소량 스모크 테스트
python generate_agents.py --model qwen9b --limit 20 --max-concurrent 4

# 3) 본 실행
python generate_agents.py --model qwen32b --max-concurrent 16

# 4) 중단 후 재개
python generate_agents.py --resume
```

### 6) 캐시 적중률 측정

```bash
# 시뮬레이션 전후 cache_hit 값 비교
curl -s http://localhost:30000/metrics | grep -E 'cache|prefill'
```

`order_keys_for_cache()` 가 적용된 두 번째 그룹 호출 이후 cache hit rate 가
크게 상승하면 정상입니다.

---

## 트러블슈팅

| 증상 | 원인 / 해결 |
|------|-------------|
| `Connection refused` to localhost:30000 | SGLang 서버가 떠 있지 않음. `scripts/serve_*.sh` 먼저 실행 |
| Qwen 응답에 `<think>...</think>` 가 그대로 나옴 | 구버전 SGLang. `extract_json_from_text()` 가 자동 제거하지만, 비용 절감 위해 서버 업데이트 권장 |
| OOM on A100 80GB | `MEM_FRAC=0.80 bash scripts/serve_*.sh` 로 낮추거나 `MAX_LEN=4096` 으로 축소 |
| `validate_vs_raw.py` 분포 불일치 | `TEMPERATURE` 를 0.7~0.8 로 낮추거나 그룹당 `count` 를 줄여 재시도 |

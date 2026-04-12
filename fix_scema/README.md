

# 📊 Agent Persona Pipeline (PySpark Edition)

본 프로젝트는 서울시 빅데이터 캠퍼스의 대용량 데이터(통신, 금융, 이동)를 활용하여 데이터 주도형(Data-driven) 소비자 에이전트 페르소나를 대규모로 자동 생성하고 검증하는 파이프라인입니다. 

기존의 순차 처리 방식을 탈피하고 **PySpark 기반의 단일 노드 분산 처리 아키텍처**를 적용하여, 캠퍼스 VDI 환경의 메모리 초과(OOM) 문제를 방지하고 처리 속도를 극대화했습니다.

---

## 🛠 1. 환경 설정 및 주의사항 (빅데이터 캠퍼스 VDI)

서울시 빅데이터 캠퍼스는 폐쇄망 환경이므로 외부 인터넷 연결 및 자유로운 라이브러리 설치(`pip install`)가 제한됩니다.

* **사전 설치 요건:** 캠퍼스 분석 환경에 기본 탑재된 `PySpark`, `Python 3.x` 환경을 그대로 활용합니다.
* **LLM 구동 환경:** 3단계 스크립트는 vLLM(Qwen3-32B-AWQ) 서버 통신을 요구합니다. GPU가 할당된 내부 서버에서 vLLM을 백그라운드로 띄워두거나(`localhost:8000`), 접속 가능한 내부 API 엔드포인트를 `03_agent_generation_and_validation.py`의 `VLLM_URL`에 맞춰 수정해야 합니다.
* **대소문자 엄격 구분:** 리눅스 기반 환경이므로 데이터 확장자(`.CSV`, `.txt`)의 대소문자를 정확히 일치시켜야 합니다.

---

## 📂 2. 디렉토리 구조 및 데이터 배치

스크립트를 실행하기 전, 작업 디렉토리 최상단에 다음과 같이 폴더를 구성하고 원본 데이터를 배치하십시오. (데이터 관리번호 및 스키마에 맞춰 정확한 파일명을 유지해야 합니다.)

```text
project_root/
├── 01_data_preprocess_spark.py
├── 02_statistical_profiling_spark.py
├── 03_agent_generation_and_validation.py
├── data/
│   ├── raw/
│   │   ├── PURPOSE_250M_XXXXXX.CSV         # [B078] 수도권 생활이동 데이터
│   │   ├── wlk_자치구_YYYYMM.txt             # [B009] KT 유동인구 (시간대별_성X연령대별)
│   │   └── 블록_성별연령대별_YYYYMM.txt      # [B063] 카드소비 패턴 데이터
│   └── mapping/
│       └── code_mapping_mopas_nso.csv      # 행정동/법정동 코드 매핑 테이블
└── output/
    ├── parquet/                            # [자동생성] 1단계 전처리 결과물 (Parquet)
    ├── stats/                              # [자동생성] 2단계 통계 프로필 (JSON)
    └── agents/                             # [자동생성] 3단계 최종 에이전트 (JSON)
```

> 💡 **Tip:** 전체 데이터를 한 번에 돌리기 전에, `data/raw/`에 각 파일 유형별로 1개의 샘플 파일만 넣고 전체 파이프라인 테스트를 진행하는 것을 강력히 권장합니다.

---

## 🚀 3. 파이프라인 실행 가이드

반드시 아래의 순서대로 스크립트를 실행해야 합니다.

### Step 1. 데이터 전처리 및 스키마 정규화 (ETL)
이기종 원본 데이터의 Wide/Long 포맷을 정규화하고, 파이프(`|`) 구분자를 처리하여 고속 병렬 처리에 최적화된 Parquet 포맷으로 변환합니다.

```bash
# Spark 환경 변수가 설정된 터미널에서 실행
python 01_data_preprocess_spark.py
```
* **출력결과:** `output/parquet/` 디렉토리 내에 `b078_mobility.parquet`, `b009_wlk.parquet`, `b063_consumption.parquet` 생성

### Step 2. 통계 집계 및 프로파일링
1단계에서 생성된 Parquet 데이터를 로드하여 그룹별(성별, 연령대 등) 분산 집계 연산을 수행하고, LLM에 주입할 경량화된 통계 JSON을 추출합니다.

```bash
python 02_statistical_profiling_spark.py
```
* **출력결과:** `output/stats/` 디렉토리 내에 `agent_spending_deciles.json`, `agent_industry_ratios.json`, `global_temporal_activity.json` 생성

### Step 3. 에이전트 생성 및 실시간 검증
산출된 통계 프로필을 바탕으로 vLLM 서버와 비동기 통신하여 에이전트를 대규모로 생성합니다. 생성 직후 비율 합계(1.0) 등 통계적 제약 준수 여부를 즉시 검증 및 교정합니다.

```bash
# vLLM 서버가 가동 중인지 확인 후 실행
python 03_agent_generation_and_validation.py
```
* **출력결과:** `output/agents/agents_final.json` (시뮬레이션에 즉각 투입 가능한 페르소나 데이터)

---

## ⚠️ 4. 트러블슈팅 (Troubleshooting)

1. **메모리 부족 (Java Heap Space / OOM) 에러 발생 시:**
   * 캠퍼스 VDI에 할당된 메모리에 따라 스크립트 상단의 Spark 세션 메모리 설정을 조절하십시오.
   * `01_data_preprocess_spark.py` 내부: `.config("spark.driver.memory", "16g")` ➡️ VDI 사양에 맞춰 `8g` 또는 `32g` 등으로 변경
2. **한글 깨짐 또는 파싱 에러 발생 시:**
   * 서울시 공공데이터 중 일부 텍스트 파일은 UTF-8이 아닌 CP949로 인코딩되어 있을 수 있습니다.
   * 파일 읽기 오류 발생 시 코드의 `spark.read.option("header", "true")` 뒤에 `.option("encoding", "cp949")`를 추가해 보십시오.
3. **vLLM Connection Error 발생 시:**
   * 3단계 실행 시 서버 연결 에러가 발생하면, VDI 내 방화벽 설정이나 포트(8000) 충돌 여부를 확인하고 `VLLM_URL` 변수를 수정하십시오.
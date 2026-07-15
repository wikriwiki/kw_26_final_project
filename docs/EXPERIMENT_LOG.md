# 실험 일지 (Experiment Log)

> 규칙: 모든 런은 실험 ID(EXP-xxx)를 부여하고, **환경·기간·정책·파라미터·타이밍**을 장표로 기록한다.
> 런 전에 장표를 먼저 채우고(사전 등록), 런 후 결과 칸만 추가한다.

---

## EXP-001 — 민생회복 소비쿠폰(P010) 14일 런 · EXAONE 4.5 · GPU LIVE

**상태**: 🟡 준비 완료 (서버 접속 대기 — PEM 키) · **목적**: 쿠폰 백테스트 1차 ON 런 (사전 7일 + 시행 7일)

### ① 인프라

| 항목 | 값 |
|---|---|
| 플랫폼 | 루키트랙 **GPU LIVE** 워크로드 (단일 노드·인터랙티브) |
| 접속 | `ssh -i 019f6020-e2b4-7cce-9098-ea9078a5046f.pem ubuntu@proxy.tta-gpu.gov-nhncloud.com -p 30043` |
| GPU | **NVIDIA A100 80GB × 2** |
| 스토리지 | `/data` (NAS 마운트 — 코드·모델캐시·Neo4j data·산출물 전부 여기) |
| 세팅 스크립트 | `scripts/deploy/setup_gpulive_exp001.sh` (s1~s9 단계형) |
| 산출물 경로 | `/data/exp001/` (run.log·llm.log·sim_output·versions.txt) |

### ② 소프트웨어 (s1 실행 후 versions.txt 값으로 확정 기입)

| 항목 | 값 |
|---|---|
| 코드 커밋 | main `e0d4814` + EXP-001 세팅 커밋 (푸시 해시: ____) |
| Python / vLLM | 3.10+ / **최신 설치** (기입: ____) — AWQ schema 지원 필요 |
| Neo4j | **Community 5.26.0 (신규 설치)**, data=`/data/neo4j_data` |
| Java | OpenJDK 17 (Neo4j 요구) |

### ③ LLM

| 항목 | 값 | 비고 |
|---|---|---|
| 모델 | **LGAI-EXAONE/EXAONE-4.5-33B-AWQ** | 🇰🇷 국산(LG) — **외국(중국 포함) 모델 사용 금지 제약** 준수 |
| 서빙 | vLLM, **TP=2 (A100 2장 전부 사용)**, port 8000 | `serve_exaone45_awq_a100x2.sh` |
| 파라미터 | gpu_util 0.92 · max_len 8192 · served_name=HF id | LLM_MODE=`exaone_4_5` |
| ⚠️ 리스크 | vllm 0.11에서 이 AWQ quantization schema 미지원 이력 | **폴백**: 동일 모델 FP8(`EXAONE-4.5-33B-FP8`, 국산 유지) — 발동 시 여기에 기록 |

### ④ 시뮬레이션 기간·파라미터

| 항목 | 값 |
|---|---|
| 기간 | **2026-05-20 ~ 2026-06-02 (14일)** |
| 구조 | 사전(정책 無) 7일: 05-20~26 → **P010 지급** 05-27 → 사후 7일: 05-27~06-02 |
| Day0 시드 | `DAY_ZERO=2026-05-19` (`08_initial_state.py`) |
| 에이전트 | 15,000 (전체) · workers **32** (초기값 — 처리량 보고 조정, 변경 시 기록) |
| 소비 모델 | `CONSUMPTION_MODEL=propensity` (봉투 포함) |
| 데이터 베이스 | 덤프 `neo4j_3day_p009_20260601_1515.dump`(553MB) 로드 → **97_reset(런 산출물 제거)** → clean Day0 |

### ⑤ 정책 (주입 내용 — preflight READY 필수)

| 항목 | 값 |
|---|---|
| 정책 | **P010 민생회복 소비쿠폰 1차 (백테스트)** — `data/neo4j_load/policies/P010.json` |
| 지급 | 전 tier 균등 **150,000원** (excluded_income 없음) · effective 2026-05-27 |
| 사용처 제한 | `poi_restricted=true` → [쿠폰] 매장 전용 (봉투 배분 + 하드검증) |
| 사용처 라벨 | 서울사랑상품권 가맹점 140,140건 실측 조인 + 룰 fallback (`09_coupon_eligibility.py`, 매칭률 기입: ____%) |
| 근사·한계 | 기초·차상위 가산 미반영(균등 15만) · 사용기한 환수 미구현(프롬프트 유인만) — P010.json notes |
| 하이퍼파라미터 원칙 | **정책 효과 관련 파라미터 튜닝 없음** — 전부 기저(정책 無) 캘리브레이션 값 그대로 (BOK_ALIGNMENT §2) |

### ⑥ 일별 소요 시간 (런 후 `bash setup_gpulive_exp001.sh timing` 출력으로 기입)

| Day | 날짜 | 소요(s) | ok / err | 비고 |
|---|---|---|---|---|
| 0 | 05-20 | | | 사전기간 시작 |
| 1 | 05-21 | | | |
| 2 | 05-22 | | | |
| 3 | 05-23 | | | |
| 4 | 05-24 | | | |
| 5 | 05-25 | | | |
| 6 | 05-26 | | | 사전기간 끝 |
| 7 | 05-27 | | | **P010 지급일** |
| 8 | 05-28 | | | |
| 9 | 05-29 | | | |
| 10 | 05-30 | | | |
| 11 | 05-31 | | | |
| 12 | 06-01 | | | |
| 13 | 06-02 | | | 종료 |
| **합계** | | | | GPU 사용률 평균(nvidia-smi): ____% |

### ⑦ 실행 체크리스트

```
[로컬] □ PEM 키 다운로드 (GPU LIVE 콘솔 → 인스턴스 상세)      ← 현재 대기 지점
       □ 코드 업로드:  scp -i {pem} -P 30043 -r kw_26_final_project ubuntu@proxy...:/data/
       □ 덤프 업로드:  scp -i {pem} -P 30043 neo4j_3day_p009_*.dump ubuntu@proxy...:/data/dumps/neo4j.dump
[서버] □ s1 deps (vllm 버전 → ②에 기입)   □ s2 모델 프리페치
       □ s3 Neo4j 신규+덤프 로드          □ s4 리셋(clean Day0 확인 ✔)
       □ s5 Day0 시드                     □ s6 P010+쿠폰 백필(매칭률 → ⑤ 기입)
       □ s7 LLM TP2 기동(/v1/models OK)   □ s8 preflight READY + LLM 스모크
       □ s9 14일 런 → ⑥ 타이밍 기입
[종료] □ 산출물 /data 보존 확인  □ 결과 분석(validate_*)  □ 본 장표 결과란 확정
```

### ⑧ 이슈·변경 이력

| 일시 | 내용 |
|---|---|
| 2026-07-15 | EXP-001 장표 작성. PEM 키 미보유로 서버 작업 대기. 세팅 스크립트·서빙 스크립트·리셋 도구 커밋 |
| | |

---

## (템플릿) EXP-00X — 제목

①인프라 / ②소프트웨어 / ③LLM / ④기간·파라미터 / ⑤정책 / ⑥일별 타이밍 / ⑦체크리스트 / ⑧이슈 — 위 구조 복사.

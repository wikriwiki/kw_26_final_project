# 실험 일지 (Experiment Log)

> 규칙: 모든 런은 실험 ID(EXP-xxx)를 부여하고, **환경·기간·정책·파라미터·타이밍**을 장표로 기록한다.
> 런 전에 장표를 먼저 채우고(사전 등록), 런 후 결과 칸만 추가한다.

---

## EXP-001 — 민생회복 소비쿠폰(P010) 7일 백테스트 런 · EXAONE 4.0(폴백) · GPU LIVE

**상태**: 🟢 실행 중 (2026-07-16, 7일 창 · start=05-24) · **목적**: 쿠폰 백테스트 ON 런 (사전 3일 + P010 + 사후 3일) — DiD 사용처 vs 비사용처 관측

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

### ③ LLM  — ⚠️ 실측 후 모델·토폴로지 변경 (2026-07-16)

| 항목 | 값 | 비고 |
|---|---|---|
| **요청 모델** | LGAI-EXAONE/EXAONE-4.5-33B-**AWQ** | ❌ **구동 불가** — 아래 사유 |
| 불가 사유 | (1) `model_type=exaone4_5`는 transformers **5.6.dev**만 인식 (설치본 4.57) (2) **비전-언어(VLM)** 모델 — `Exaone4_5_ForConditionalGeneration` (3) vLLM 0.18.1에 **해당 아키텍처 실행 클래스 없음** (지원: ExaoneForCausalLM/Exaone4ForCausalLM/ExaoneMoE) | FP8도 동일 아키텍처라 무의미 — 폴백 불가 |
| **채택 모델** | **LGAI-EXAONE/EXAONE-4.0-32B-AWQ** (`Exaone4ForCausalLM`) | 🇰🇷 국산(LG)·**text-only**·vLLM 정식지원 → **외국/중국 모델 금지 제약 준수** |
| 서빙 | vLLM 0.18.1, **DP=2 (데이터패러럴, GPU 1장당 풀 레플리카 1개 = 2장 전부 사용)** | 모델 18GB로 A100 1장에 적재됨 → TP=2는 all-reduce 오버헤드로 오히려 느림. DP=2가 처리량 우위(벤치 2362 vs TP2 500-800 tok/s) |
| 커널/정밀도 | **awq_marlin** (A100 sm80 가속) · **dtype float16** (awq는 bf16 불가) | 초기 `--quantization awq`는 느린 커널 강제됨 → marlin 전환 |
| 배칭 | `--max-num-batched-tokens 16384` · `--max-num-seqs 64` | batched-tokens 2048(기본)은 롱프롬프트 prefill이 decode를 굶김 / seqs 256은 롱컨텍스트로 KV 99% 스래싱 → 64로 안정화(KV~45%) |
| 파라미터 | gpu_util 0.92 · max_len 8192 · served_name=HF id | LLM_MODE=`exaone` |
| 서빙 스크립트(실측판) | `/data/exp001/run_llm.sh` (레포 `serve_exaone45_awq_a100x2.sh`는 4.5 전제라 미사용) | |

### ④ 시뮬레이션 기간·파라미터

| 항목 | 값 |
|---|---|
| 기간 | **2026-05-24 ~ 2026-05-30 (7일)** — 처리량 재산정(~6일→~3일)으로 백테스트 창 축소 (사용자 승인) |
| 구조 | 사전(정책 無) 3일: 05-24~26 → **P010 지급** 05-27 → 사후 3일: 05-28~30 |
| Day0 시드 | `DAY_ZERO=2026-05-23` (`08_initial_state.py`) |
| 에이전트 | **14,560** 처리 대상 (State 시드 14,881 중 조건 충족분) · workers **96** (실측 튜닝값, 32→128→192→64→96) |
| 처리량(실측, 지속) | **~23–27 agent/분** (초반 버스트 41–48이나 지속률은 낮음) · GPU 2장 100% |
| 병목(진단) | **롱프롬프트 prefill이 저동시성(≈14 req)에서도 GPU 100% 포화** — 프롬프트에 페르소나+정책+POI 후보목록(4–6k tok). 동시성↑ 무효(GPU 여유 0), 프롬프트 축소는 sim 코드 변경(범위 밖) |
| **일별 실측 예상** | **~9–10 시간/일** → **14일 ≈ 130–150시간(~5.5–6일) 벽시계** (Day0 완료 시 ⑥ 실측 기입) |
| 중단 내성 | **`/data/exp001/resume.sh`** — 중단 후 한 줄로 재개(멱등: 완료 agent skip, grant 중복적용 가드, per-day checkpoint) → 6일 런도 인터럽트 생존 |
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
| 0 | 05-24 | | | 사전기간 시작 |
| 1 | 05-25 | | | |
| 2 | 05-26 | | | 사전기간 끝 |
| 3 | 05-27 | | | **P010 지급일** |
| 4 | 05-28 | | | 사후기간 시작 |
| 5 | 05-29 | | | |
| 6 | 05-30 | | | 종료 |
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
| 2026-07-16 | PEM 확보→서버 접속. s1~s3 완료(vllm 0.18.1, A100×2, Neo4j 5.26 덤프 2.9GiB 로드). 코드 실경로=worktree(최상위 checkout은 구커밋)→REPO 재지정 |
| 2026-07-16 | **모델 변경**: EXAONE-4.5-33B-AWQ 구동 불가(VLM·transformers5.6dev·vLLM 미지원) → **EXAONE-4.0-32B-AWQ**(국산·text-only·정식지원) 채택 |
| 2026-07-16 | **서빙 튜닝**: TP2→**DP2**(모델 18GB 1장 적재, all-reduce 회피) · awq→**awq_marlin** · dtype **float16**(awq bf16불가) · batched-tokens 2048→**16384** · seqs 256(KV99%스래싱)→**64** |
| 2026-07-16 | s4~s6 완료(리셋 clean·14,881 시드·P010 적재·coupon 537,489 라벨). preflight READY·LLM 스모크 OK. **P010 effective_until 05-28→2026-09-30**(실제 다개월 사용기한 반영, 관측창 전체 유효) |
| 2026-07-16 | s9 14일 런 개시. 실측 지속처리량 ~25 agent/분(롱프롬프트 prefill GPU포화 병목) → 일 ~10h·14일 ~6일 재산정. `resume.sh`로 중단내성 확보 |
| 2026-07-16 | 처리량 재산정 반영 **7일 백테스트 창으로 축소(사용자 승인)**: start=05-24·days=7·Day0=05-23. 사전3(24~26)+P010(27)+사후3(28~30). 예상 ~3일 |

---

## (템플릿) EXP-00X — 제목

①인프라 / ②소프트웨어 / ③LLM / ④기간·파라미터 / ⑤정책 / ⑥일별 타이밍 / ⑦체크리스트 / ⑧이슈 — 위 구조 복사.

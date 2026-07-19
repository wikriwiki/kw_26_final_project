# EXP-001 실행 런북 — 민생회복 소비쿠폰 백테스트 (A100×2 / GPU LIVE)

> 대상: 메가존클라우드 GPU LIVE (루키 트랙) · A100 80GB × 2
> 이 문서만 따라가면 끝. 상세 배경은 `docs/GPU_LIVE_SETUP.md`.

---

## 0. 실험 설계 (확정)

| 항목 | 값 |
|---|---|
| 정책 | **P010 민생회복 소비쿠폰 1차** (행안부, 실제 정책) |
| 정책 시행일 | **2025-07-21 (월)** — 실제 1차 신청·지급 개시 |
| 시뮬 기간 | **2025-07-14(월) ~ 07-27(일)** = 14일 (시행 전 7일 + 후 7일, 월~일 2주 정렬) |
| 지급액 | 소비 **1분위 40만** / **2분위 30만** / **3~10분위 15만** |
| 사용처 | 지역사랑상품권 가맹점 + 연매출 30억 이하 (`poi_restricted=true`) |
| 에이전트 | **7,500명** (소비 10분위 비례 층화표본) |
| 모델 | **LGAI-EXAONE/EXAONE-4.5-33B-AWQ**, TP=2 (A100×2 전부) |

**지급액 근거** — 실제 1차: 일반국민 15만 / 차상위·한부모 30만 / 기초수급자 40만.
서울=수도권이라 비수도권(+3만)·농어촌 인구감소(+5만) 가산 미적용 → 최대 40만.
2차(9/22~, +10만)는 시뮬 기간 밖. 계층을 시뮬의 최소 계층 단위인 소비 10분위에 대응.

**표본 검증** (모집단 14,683 → 7,500):

| 차원 | 최대 편차 |
|---|---|
| 소비 10분위 / 소득 5분위 | ±0.01%p |
| 연령 | ±0.35%p |
| 성별 | ±0.42%p |
| 자치구 | 25/25개 커버, 평균 0.10%p |

---

## 0.5 재현 체크리스트 (다른 팀원이 같은 클라우드에서 돌리려면)

**git clone 만으로는 부족하다.** 아래 3가지가 추가로 필요하다.

| 필요물 | git에 있나 | 조달 방법 |
|---|---|---|
| 시뮬 코드·정책(P010.json)·쿠폰 CSV(`data/coupon/`) | ✅ | `git clone` |
| 운영 스크립트(verify/snapshot/launch) | ✅ (이 커밋부터 `scripts/exp/`) | `git clone` |
| SGLang 서빙 스택 | ✅ (이 커밋부터, 아래) | `scripts/deploy/install_sglang_exaone45.sh` |
| **베이스 Neo4j 덤프**(Agent·POI·KNOWS 그래프) | ❌ **불가**(용량·개인정보) | NAS `/home/ubuntu/data/dumps/` 에서 회수하거나 별도 공유 |

> **베이스 그래프는 git에 넣을 수 없다.** 에이전트는 BDC 실측 데이터 + 페르소나 생성으로
> 만든 것이라 원본(BDC)이 없으면 `scripts/neo4j_load/` 파이프라인으로도 재생성 불가.
> 실무 경로 = **덤프 파일 1개를 별도로 받아 `/data/dumps/neo4j.dump` 로 업로드.**
> EXP-001 산출 덤프: NAS `/home/ubuntu/data/dumps/neo4j_baseline_pre_p010_20250720.dump`
> (정책 직전 순수 baseline) 또는 `neo4j_base_day0.dump`(Day0 시작 상태).

**서빙은 vLLM이 아니라 SGLang이다.** 이 EXAONE-4.5 AWQ(compressed-tensors int4)는 vLLM이
로드하지 못한 이력이 있어(§3 s7의 vLLM 폴백 체인은 실패함), EXP-001은 EXAONE-4.5 지원
SGLang 포크로 서빙했다. 실전에서 돈 경로:

```bash
bash scripts/deploy/install_sglang_exaone45.sh          # /data/venv_sgl 구성 (~10분)
export HF_HOME=/data/hf_cache
nohup bash scripts/serve/serve_exaone45_sglang_a100x2.sh > /data/exp001/llm.log 2>&1 &
# 헬스: curl -sf localhost:8000/v1/models
```

**본런·검증·baseline 덤프** (setup_gpulive_exp001.sh 의 s7 vLLM 대신 위 SGLang 사용):

```bash
# 본런 (결과는 NAS로 직접 기록 → 컨테이너 소실에도 resume 안전)
WORKERS=64 nohup bash scripts/exp/launch_exp001.sh > /home/ubuntu/data/exp001/logs/run.log 2>&1 &

# 3시간마다 상세 검증(속도·품질·추출완전성·소비·이동·정책·사회·무결성 8차원)
python scripts/exp/verify_exp001.py

# 정책 시행 직전 순수 baseline 덤프 (시뮬 중단→오염노드+산출물 제거→덤프→resume)
bash scripts/exp/snapshot_baseline.sh
```

---

## 1. 콘솔 작업 (웹 UI — 사람이 해야 함)

접속: `https://iam.tta-gpu.gov-nhncloud.com/login`

1. **스토리지 생성** → 상태가 **BOUND** 확인 (이게 없으면 워크로드 삭제 시 전부 소실)
2. **워크로드 생성**
   - GPU: **A100 80GB × 2** (전부)
   - 유형: **단일 노드 / 인터랙티브** (SSH 접속용)
   - 실행 환경: **NVIDIA PyTorch 계열 기본 이미지** (CUDA 포함)
   - **공유 메모리: 16GB 이상** (vLLM이 shm 사용 — 부족하면 TP2에서 실패)
   - **스토리지 마운트: `/data`**
3. 상태 **실행 중** → 인스턴스 상세에서 **PEM 키 다운로드** + SSH 명령 확인

---

## 2. 업로드 (로컬 → 서버)

```bash
# SSH 접속 (콘솔에 표시된 명령)
ssh -i {키.pem} ubuntu@proxy-app.{도메인} -p {포트}

# 서버에서 디렉토리 준비
mkdir -p /data/dumps
```

로컬에서 2가지 업로드 (SFTP/FileZilla 또는 scp):

| 대상 | 로컬 경로 | 서버 경로 |
|---|---|---|
| **코드** | (git clone 권장 — 아래) | `/data/kw_26_final_project` |
| **베이스 덤프** | `output/sim/dumps/v8_baseline_before_p009/neo4j_v8_baseline_before_p009.dump` (774MB) | `/data/dumps/neo4j.dump` ← **파일명 반드시 `neo4j.dump`** |

```bash
# 코드는 git clone (업로드보다 빠르고 최신)
cd /data && git clone https://github.com/wikriwiki/kw_26_final_project.git
cd kw_26_final_project && git checkout main     # ed4c3f4 이상
```

```bash
# 덤프는 로컬에서 scp (파일명 변환 주의)
scp -i {키.pem} -P {포트} \
  "output/sim/dumps/v8_baseline_before_p009/neo4j_v8_baseline_before_p009.dump" \
  ubuntu@proxy-app.{도메인}:/data/dumps/neo4j.dump
```

> **덤프 선택 이유**: 이 덤프는 Agent 15,000 · POI 542,478 · KNOWS 197,676 ·
> **WORKS_AT 12,078**(직장 외출 재현에 필수) 시드를 모두 보유. 이전 런 산출물
> (Plan/State/Memory/Conversation)은 `s4_reset`이 걷어내 clean Day0으로 만든다.

---

## 3. 원샷 실행

```bash
cd /data/kw_26_final_project
bash scripts/deploy/setup_gpulive_exp001.sh all
```

단계별로 하려면 (권장 — 실패 지점 파악 쉬움):

| 단계 | 명령 | 소요 | 하는 일 |
|---|---|---|---|
| s1 | `bash scripts/deploy/setup_gpulive_exp001.sh s1_deps` | ~5분 | venv + vLLM + Java17 |
| s2 | `... s2_model` | ~20분 | EXAONE-4.5-33B-AWQ(+FP8 폴백) 다운로드 → `/data/hf_cache` |
| s3 | `... s3_neo4j` | ~5분 | Neo4j 5.26 설치 + 덤프 로드 + 기동 |
| s4 | `... s4_reset` | ~2분 | 이전 런 산출물 제거 → clean Day0 |
| s5 | `... s5_seed` | ~2분 | Day0 State 시드 (2025-07-13) |
| s6 | `... s6_policy` | ~5분 | **P010 적재 + 쿠폰 사용처 백필**(실측 가맹점 14만건 조인) |
| s7 | `... s7_llm` | ~10분 | vLLM TP=2 기동 (**awq_marlin → awq → fp8 자동 폴백**) — ⚠️ 이 모델은 vLLM 로드 실패 이력. **실전은 §0.5 SGLang 사용** |
| s8 | `... s8_preflight` | ~1분 | 정책 사전점검 + LLM 스모크 |
| s9 | `... s9_run` | **본런** | 14일 시뮬 (nohup) |

---

## 4. 모니터링

```bash
tail -f /data/exp001/run.log                              # 진행
bash scripts/deploy/setup_gpulive_exp001.sh timing        # 일별 소요
grep -c '"status": "ok"' /data/exp001/sim_output/metrics/day_2025-07-*.jsonl   # 일별 처리량
nvidia-smi                                                 # GPU (2장 다 물려야 정상)
cat /data/exp001/llm_quant.txt                            # 어떤 양자화로 떴는지
```

**정상 지표**: GPU util 90%+ / `num_requests_running` ≈ workers / err율 < 1%

---

## 5. 성능 튜닝 (첫 1시간 관찰 후)

`WORKERS=32`가 기본. 33B는 11B보다 토큰당 느리므로 실측 보고 조정:

```bash
# 처리율 확인 (agents/min)
grep "@" /data/exp001/run.log | tail -3

# 낮으면 workers 상향 후 재기동 (KV cache 여유: 33B AWQ ~18GB / A100 160GB)
pkill -f run_simulation
WORKERS=48 bash scripts/deploy/setup_gpulive_exp001.sh s9_run   # resume이 이어받음
```

> 이전 실측(RTX5090 · 11B AWQ+Marlin · workers 48) = 30~36 agents/min → 7,500명/일 ≈ 3.5~4시간.
> 33B·A100×2는 이보다 느릴 수 있음. **14일 완주는 40~60시간 예상** — 여유 갖고 시작.

---

## 6. 결과 보존 (⚠️ 워크로드 삭제 전 필수)

산출물이 **전부 `/data`(스토리지)** 에 있는지 확인. 마운트 밖(`/home/ubuntu`)은 삭제 시 소실.

```bash
ls /data/exp001/sim_output/metrics/     # day_2025-07-*.jsonl 14개
ls /data/exp001/run.log /data/exp001/llm.log

# 포스트런 덤프 (분석·재현용) — 반드시 생성
/data/neo4j-community-5.26.0/bin/neo4j stop
/data/neo4j-community-5.26.0/bin/neo4j-admin database dump neo4j --to-path=/data/dumps_out
ls -lh /data/dumps_out/            # 이 파일을 로컬로 회수
```

---

## 7. 함정 (겪은 것들)

| 증상 | 원인·해결 |
|---|---|
| vLLM이 GPU 1장만 사용 | `TP=2` 확인. `nvidia-smi`에 2장 다 떠야 정상 |
| AWQ 스키마 거부 | s7이 `awq_marlin → awq → fp8` 자동 폴백. 최종 실패 시 `tail -40 /data/exp001/llm.log` |
| 처리율이 비정상적으로 낮음 | 양자화 커널 확인(`cat /data/exp001/llm_quant.txt`). `awq`면 Marlin 대비 최대 5배 느림 |
| shm 부족 / TP2 hang | 워크로드 **공유 메모리 16GB↑** 로 재생성 |
| 덤프 load 실패 | 파일명이 `neo4j.dump` 여야 함 (Community는 `<db명>.dump` 규약) |
| 정책이 프롬프트에 안 뜸 | `python scripts/sim/policy_preflight.py data/neo4j_load/policies/P010.json` 로 진단 |
| 워크로드 삭제 후 데이터 없음 | `/data`(스토리지) 밖에 저장한 것 — §6 재확인 |

---

## 8. 완료 후 분석

```bash
# 소득분위별 쿠폰 소진율 + DID (대조군 없음 — P010은 전 국민 지급)
#  → baseline 7일 vs 정책 7일, 분위별 비교로 효과 추정
python scripts/sim/generate_final_report.py --help
```

> **주의**: P010은 **전 국민 지급**이라 소득 '상' 대조군이 없다(모두 최소 15만 수령).
> 따라서 P009식 DID(수혜 vs 미수혜) 불가 → **분위 간 차등(40/30/15만) 기반 용량반응(dose-response)**
> 과 **baseline 7일 대비 전후 비교**로 효과를 본다. 사용처 제한이 있으므로
> **쿠폰 가맹점 vs 비가맹점 매출 전환**도 핵심 지표.

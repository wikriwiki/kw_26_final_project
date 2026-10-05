# infra.md — 인프라와 비용

> 메인: [`./schedule_generation_plan.md`](./schedule_generation_plan.md). 현재 설치·서빙 절차는 [`../../SETUP.md`](../../SETUP.md), Vast 준비는 [`../../deploy/vast/README.md`](../../deploy/vast/README.md)를 따른다.

## 1. 실행 모델

기본은 **LGAI-EXAONE/EXAONE-4.5-33B-AWQ**, `LLM_MODE=exaone_4_5`다. `scripts/serve/serve_exaone45_sglang_a100x2.sh`로 기동하고 고정 checkpoint revision을 사용한다. 에이전트 계획·POI 선택·의도 분류가 같은 모델 설정을 공유한다.

AWQ 체크포인트는 W4A16g128 `compressed-tensors` 형식이므로 별도 `awq_marlin` 플래그를 강제하지 않는다. 공식 LG namespace만 서버 기본 진입점에서 허용한다. API 공급자 폴백은 이 실험의 기본 경로에 포함하지 않는다.

기존 SGLang TP=2 환경에서 소량 검증을 시작한다. 단일 32/48GB급 GPU는 검증 후보이며 실제 적재 가능 여부와 처리량은 서버에서 확인한다. 작업 크기를 늘릴 때 GPU 모델·메모리·서빙 버전·배치·프롬프트를 기록한다.

---

## 2. 토큰 예산 (호출당)

에이전트 1명 × 하루 = **2회 호출** (Stage 1 → Stage 2).

**Stage 1 — "무엇을 할까"**

| 블록 | 토큰 | 캐시? |
|---|---|---|
| System 지시 | 300 | ✅ 전역 |
| 페르소나 + 고정 장소 + 참조통계 + 카테고리 어휘 | 700 | ✅ per-agent |
| 날짜·요일 | 100 | ❌ |
| memory_context (30일 Top-5~7) | ≤500 | ❌ |
| policy_context (awareness≥0.3) | ≤200 | ❌ |
| social_context (14일 pending 포함) | ≤400 | ❌ |
| Intent zone 힌트 | ≤150 | ❌ |
| running_state (정책 수혜 시 쿠폰 포함) | ≤100 | ❌ |
| **Stage 1 입력** | **~1,950** | - |
| Stage 1 출력 (category 시퀀스) | ~300 | - |

**Stage 2 — "어디로 갈까"** (미결 event만 처리, pinned_poi는 Stage 2 우회)

| 블록 | 토큰 | 캐시? |
|---|---|---|
| 페르소나 + 고정 장소 | 450 | ✅ per-agent |
| 미결 이벤트 목록 (Stage 1 출력 중 pinned 제외) | ~200 | ❌ |
| Event별 POI Top-30 (KDTree 런타임 필터) | ~1,200 | ❌ |
| **Stage 2 입력** | **~1,850** | - |
| Stage 2 출력 (미결 event POI 확정) | ~250 | - |

**합계: 입력 ~3,800 / 출력 ~550** (2회 호출 합산, 하루 1~2 event 평균 pin 가정)

---


## 3. 비용 산정

위 토큰 표는 설계 당시의 예산이며 측정 처리량이 아니다. 현재 프롬프트의 실제 입력·출력 토큰과 재시도율을 먼저 측정한다.

- 총 GPU 비용 = 실제 서버 점유 시간 × 계약 시간당 요금 + 스토리지·전송 비용.
- 예상 agent-day 처리량은 동일 모델·서버로 50~100명 규모를 측정해 계산한다.
- 무료 GPU 제공이나 과거 API 가격을 현재 Vast 비용으로 적용하지 않는다.
- 이전 모델의 속도 기록을 LG 모델의 처리량으로 바꾸어 적지 않는다.

## 4. 후보 설정 비교

AWQ/FP8/BF16/GGUF의 속도 순위를 고정하지 않는다. 필요한 경우 동일 GPU와 동일 요청에서 첫 토큰 지연, 출력 tokens/s, agent-day/hour, 최대 VRAM, JSON 실패율을 함께 비교한다. 선택된 AWQ가 메모리와 동시성 요구를 만족하는지 먼저 확인하고, 후보 변경은 별도 실행으로 기록한다.

## 5. 모니터링

- 호출 수·성공률·재시도와 schema 실패 분포
- 입력·출력 토큰, prefix cache, 요청 지연과 GPU 사용량
- 동일 시드의 정책 OFF/ON 실행 및 모델·입력 해시
- 완료시간과 청구금액에 따른 실제 비용

체크포인트와 결과는 영구 볼륨에 저장한다. 여러 날짜를 한 번에 생성해 상태 연속성을 생략하는 최적화는 적용하지 않는다.

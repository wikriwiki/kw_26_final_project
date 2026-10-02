# P012 상생소비지원금 시뮬레이션 — 실행한 코드 그대로

이 브랜치는 2026-09-29 ~ 10-03 에 A100 서버에서 돌린 P012 본런의 코드다.
`scripts/sim/` 은 서버에서 실제로 돈 파일을 바이트 그대로 옮겼다.

## 무엇을 돌렸나

- 같은 서울 시민 에이전트 3,000명(`output/cohort/p012_main_3000.json`)을 같은 날짜
  2021-10-01 ~ 10-07(7일)로 두 번 시뮬레이션했다. 한 번은 상생소비지원금이 있고 한 번은 없다.
  두 결과의 차이를 KDI 「상생소비지원금의 소비 진작 효과」(2022.9) 실측과 맞댔다.
- 정책 파일: `data/experiments/P012_v53_compressed7_main_20260929.json`
  (한 달 문턱·한도를 7/31 로 줄인 판). 지표 계약서: `data/experiments/P012_indicator_contract.json`
- 모델: `LGAI-EXAONE/EXAONE-4.5-33B-AWQ` (SGLang, A100 1장), 프롬프트 v53, 환경 covid_2021
- 설정: `EXP_ELIGIBLE_CHANNEL=1`, `EXP_DAILY_INCOME=anchor:0.41667`, workers 64
- 걸린 시간: 지원금 있음 45시간 35분(9/29 15:49 ~ 10/1 13:24), 없음 38시간 12분(10/1 13:38 ~ 10/3 03:50)

## 코드가 실행본과 같다는 근거

- `scripts/sim/` 의 소스 지문(`experience_provenance.source_fingerprint`):
  `a78d3a2abf031a834df90cf04163f4fb940f91aea079be9c2a4f67bd94875d1b` — 이 브랜치에서 다시 계산해도 같다.
- 두 시뮬레이션 14일치 cohort 파일의 실행 지문이 모두 같다:
  `a8a1011c8eed925ba79fb80d5e0aaa023e806bec6912e2113c721095637986dc` (코드 + EXP_* 설정)
- 서버 `scripts/sim` 의 마지막 수정은 9/29 15:48, 본런 시작은 15:49.
- 이 브랜치의 부모 커밋과 다른 점: `run_simulation.py`·`dawn_context.py` 는 이 실행 뒤에 들어온
  인구 프로필 연동이 빠진 실행본이고(`population_profile.py` 도 없다), `night_intent_llm.py` 는
  대화 키를 쌍에서 가져오는 수정만 들어간 실행본이다.

## 실행 순서 (서버)

1. `tools/run_p012_main_20260929.sh` — 지원금 있는 7일 실행 (`tools/run_multi_v53_short_arm_20260928.sh p012m on`)
2. `tools/continue_p012m_20260929.sh` — 보존 → 지원금 없는 7일 → 보존 → 채점
3. `tools/chain_p012m_finish_20260929.sh` → `tools/run_p012_finish_20260929.sh` —
   보존 검사, 그래프 덤프 실제 복원 대조, 1대1 인터뷰, 보고서
4. 유사도: `scripts/report/similarity_p012.py` → `scripts/report/build_p012_similarity_html.py`

보존은 `tools/preserve_multi_v53_short_arm_20260928.sh` 가 한다: 결제 원장 2종
(`export_multi_policy_sector_ledger.py`, `export_cashback_month.py`), 에이전트별 기억·계획·상태
(`export_agent_dossier.py`), 그래프 덤프. 검사는 `verify_p012_preservation.py`,
`verify_graph_against_dossier.py`, `tools/verify_graph_restore_20260929.sh`.

## 결과

- 기록: `data/experiments/P012_MAIN_7D_RESULT.md`, 보고서: `output/p012m_main_20261003/P012_VALIDITY.md`,
  유사도: `output/p012m_main_20261003/similarity.json` · `p012_similarity.html`
- 방향 일치 9/11, 크기 유사도 0.395 (95% 0.304~0.471), 순위 상관 −0.02, 평균 절대 오차 15.1%p

## 데이터가 있는 곳 (저장소에는 없다)

- 서버: `/data/multipolicy_v53_20260928/p012m/` (원장·기억 모음·그래프 덤프·채점·보고서)
- 서버 밖 사본: `E:\p012m_preserved_20261002\` (두 그래프 덤프, 출발 그래프, 원장·기억 묶음,
  마무리 산출물 — 서버 체크섬과 대조함), `E:\server_data_bundle_20261003\` (서버 /data 전체 산출물 묶음)

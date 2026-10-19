# L1 기준선·일관성 평가 도구 (2026-10-19)

보존된 본 실행 결과(zip)를 읽어 기준선 모사·관찰 시험·인용 대조를 평가한 코드와 그 결과(집계값만, 원 데이터 없음).
설계와 판정 규칙: `data/experiments/L1_BASELINE_FIX_DESIGN_20261019.md`.

| 파일 | 하는 일 |
|---|---|
| extract_run.py / extract_run_v2.py | 본 실행 zip(dossier·원장)에서 일정·기억·상태·정책 상태를 뽑아 pickle 로(v2 는 세부 업종·동 코드 포함) |
| l1_multiscale.py | 어디서·언제·무엇에·누가·사람 위치를 단위별로 실측과 비교(균등·입력 배분·상가 수·완벽한 모형) |
| l1_baseline.py / l1_within_group.py | 패턴별 기준선 비교, 업종 안 시간대·요일 비교 |
| obs_consistency.py / cross_run.py / nsz_consistency.py | 관찰 시험(비슷한 조건 → 비슷한 행동), 실험 간 같은 사람, 금연구역 입장 일관성 |
| cite_threshold.py | 인용 대조 규칙(정확 일치 대 유사도 임계값)과 틀린 인용 통과율 |
| make_multiscale_fig.py | 단위 사다리 그림 |

경로는 작성자 PC 기준(`C:/Users/srdyh/...`)이다. 실측 CSV(상권분석 추정매출 2024, 생활인구 2026-03, 소상공인 상가정보 2025-12)와
본 실행 zip(팀 공유 드라이브 KW_시뮬결과_공유_20261011)은 저장소에 넣지 않았다.
새 런을 평가할 때는 extract_run_v2.py 로 zip 을 cache 로 뽑은 뒤 l1_multiscale.py 의 RUNS·경로를 바꿔 돌린다.

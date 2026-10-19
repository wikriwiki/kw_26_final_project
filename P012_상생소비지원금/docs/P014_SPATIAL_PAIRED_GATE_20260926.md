# P014 위치별 소비 쌍체 원장 관문 (2026-09-26)

현재 `price_discount` 실행 경로는 서울사랑상품권의 할인 구매와 실제 사용, 본인 현금 유출, 발행 보조금을 거래별로 정산하지 않는다. 정책 문구와 적격 매장 표시는 있지만, 그 사실만으로 구매력이나 할인 소진이 시뮬레이션에 구현됐다고 볼 수 없다. 이 결함을 해결하기 전에는 P014 결과를 외부 효과 크기 적중이나 총소비 무반응 통과로 판정하지 않는다.

새 `export_spatial_daily_ledger.py`는 각 ON/OFF 팔의 **완료된** 시민×날짜 원장에서 매장 총매출, 온라인 지출, 거주 행정동 매출, 거주 자치구 매출, 타 자치구 매출을 추출한다. 모든 양의 오프라인 거래에 거주지·매장 동 코드가 있어야 하며, `거주 자치구+타 자치구=전체 오프라인 매출`이어야 한다. 0원 시민도 남긴다. 등록 인원·날짜의 정상 metrics, Stage2 생성 품질, 실행 지문, 정책 파일과 그래프의 시행일·할인율·월 구매한도·사용처 규칙을 검증한 후 출력·매니페스트를 원자적으로 쓴다. 무정책 팔은 Policy 노드가 0개여야 한다.

현재 A100의 **P012** 10월 19일 완료 아카이브로 위치 필드의 구조만 점검했다. 양의 지출 거래 2,804건 전부에서 Agent 거주 동 코드와 매장 동 코드가 유효한 같은 길이의 8자리 코드였고, 읽기 전용 그래프 조회에서도 2,804건 모두 `LIVES_AT` 거주 POI·매장 POI 동 코드가 있었다. 이는 P014 ON/OFF 실행 결과나 상품권 정산 검증이 아니다.

`paired_local_voucher_effect.py`는 두 팔의 **동일 시민·동일 날짜** 원장을 필요로 한다. 효과 창을 결과 전에 지정하고 시민 단위로 재표집한다. `LV-1`은 오프라인 매장 총매출과 온라인 지출의 합이며 전체 차이·시민당 하루 차이·무정책 팔 대비 상대 변화를 낸다. `LV-2`는 전체 오프라인 매출 중 거주 **행정동** 매출 비중, `LV-3`은 타 **자치구** 매출 비중이다. 정책 조건과 직접 대응하는 거주 **자치구** 비중도 보조 지표로 낸다. 행정동과 자치구를 혼동하지 않는다. 온라인 지출에는 매장 위치가 없으므로 LV-2/3의 분모에 넣지 않는다. 분모가 0인 표본은 비율을 임의의 0으로 채우지 않는다.

이 산출물은 `comparison=indirect_proxy`, `prepaid_voucher_settlement_verified=false`, `external_magnitude_comparable=false`를 명시한다. 조세재정연구원의 원문은 같은 서울 합성 시민·같은 날짜의 매장 총매출 ON/OFF 효과가 아니며, LV-1의 외부 무효과 동등성 띠도 아직 근거 있게 등록되지 않았다. 따라서 방향·내부 이동량만 진단한다. 예전의 서로 다른 전후 날짜를 정책 효과로 읽는 방식은 채택하지 않는다.

원문 재감사에서는 세 지표 모두 **외부 실측 부호 적중의 대상이 아님**을 확인했다. 원문 방법(PDF 71쪽/인쇄 54쪽)은 지자체×연도×업종 패널의 가맹 소상공인 업종 매출 로그를 삼중차분한다. 표 VI-4(PDF 75쪽/인쇄 58쪽)의 매출 계수 비유의는 시민 총지출 0 또는 동등성을 뜻하지 않는다. 표 VI-6(PDF 79쪽/인쇄 62쪽)의 업종별 매출 계수는 거주 행정동 소비 비중이나 타 자치구 소비 비중이 아니다. `LV-1`은 `different_estimand`, `LV-2/3`은 `not_observed`로 채점표에 기록했다. 이 값들은 내부 기전만 진단하며 새 프롬프트 선발의 실측 점수에 더하지 않는다. 후속 외부 검증에는 원문과 대응하는 가맹점 업종별 매출, 발행액, 비교 지역·업종·연도의 대조 설계가 필요하다.

새 P014 실험을 시작하려면 진행 중인 A100 P012의 완결 백업, 별도 ON/OFF 그래프, 정책 전 같은 날짜 차이 진단, 양팔의 동일 예산 보충 맵과 프롬프트 해시가 먼저 필요하다. 프롬프트 후보마다 두 팔을 모두 실행한다. 추출은 **각 팔의 그래프를 초기화하기 전에** 수행한다. 실행 예시는 다음과 같으며 아직 실행 완료를 뜻하지 않는다.

```text
python scripts/report/export_spatial_daily_ledger.py --roster <frozen_roster.json> --start <start> --end <end> --arm on --policy-file data/neo4j_load/policies/P014.json --metrics-dir <ON>/metrics --out <ON>/p014_spatial.jsonl
python scripts/report/export_spatial_daily_ledger.py --roster <frozen_roster.json> --start <start> --end <end> --arm off --policy-file data/neo4j_load/policies/P014.json --metrics-dir <OFF>/metrics --out <OFF>/p014_spatial.jsonl
python scripts/report/paired_local_voucher_effect.py --on <ON>/p014_spatial.jsonl --off <OFF>/p014_spatial.jsonl --roster <frozen_roster.json> --start <start> --end <end> --effect-start <registered_effect_start> --effect-end <registered_effect_end> --json-out <result.json>
```

외부 크기 검증에는 추가로 할인 상품권의 구매 결정을 시민 행동으로 기록하고, 본인 현금과 상품권 액면 잔액, 할인 보조금, 적격 매장 사용액을 거래별로 보존·대조해야 한다. 총소비가 늘지 않는다는 결과를 프롬프트나 소비 엔진의 규칙으로 고정하지 않는다. 원문 측정대상·관측 기간·분모·대조군도 별도로 맞춰야 한다.

2026-09-26 사용처 배선 보강: 세 Stage2 후보 조회가 `p.upjong_l3`를 전달하지 않아 범용 적격 평가기는 실제 후보에서 코드 기준 규칙을 사용할 수 없었다. 조회 필드를 추가했고, `P014.json`의 백화점·면세점은 코드가 있으면 기존 `exclude.subs` fallback을 건너뛰는 문제를 막기 위해 `exclude.subs_always`로 선언했다. 이는 앞으로의 후보·적격 판정 정합성을 위한 소스 변경이며 이미 보관된 P014 런을 새 코드의 결과로 재해석하지 않는다. 기존 P012 적립 규칙과의 323개 사례 대조는 불일치 0개다. 실제 P014 상품권 결제 회계와 원문 추정량의 공백은 그대로다.

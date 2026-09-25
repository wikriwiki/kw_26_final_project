# P014 위치별 소비 쌍체 원장 관문 (2026-09-26)

현재 `price_discount` 실행 경로는 서울사랑상품권의 할인 구매와 실제 사용, 본인 현금 유출, 발행 보조금을 거래별로 정산하지 않는다. 정책 문구와 적격 매장 표시는 있지만, 그 사실만으로 구매력이나 할인 소진이 시뮬레이션에 구현됐다고 볼 수 없다. 이 결함을 해결하기 전에는 P014 결과를 외부 효과 크기 적중이나 총소비 무반응 통과로 판정하지 않는다.

새 `export_spatial_daily_ledger.py`는 각 ON/OFF 팔의 **완료된** 시민×날짜 원장에서 매장 총매출, 온라인 지출, 거주 행정동 매출, 거주 자치구 매출, 타 자치구 매출을 추출한다. 모든 양의 오프라인 거래에 거주지·매장 동 코드가 있어야 하며, `거주 자치구+타 자치구=전체 오프라인 매출`이어야 한다. 0원 시민도 남긴다. 등록 인원·날짜의 정상 metrics, Stage2 생성 품질, 실행 지문, 정책 파일과 그래프의 시행일·할인율·월 구매한도·사용처 규칙을 검증한 후 출력·매니페스트를 원자적으로 쓴다. 무정책 팔은 Policy 노드가 0개여야 한다.

`paired_local_voucher_effect.py`는 두 팔의 **동일 시민·동일 날짜** 원장을 필요로 한다. 효과 창을 결과 전에 지정하고 시민 단위로 재표집한다. `LV-1`은 오프라인 매장 총매출과 온라인 지출의 합이며 전체 차이·시민당 하루 차이·무정책 팔 대비 상대 변화를 낸다. `LV-2`는 전체 오프라인 매출 중 거주 **행정동** 매출 비중, `LV-3`은 타 **자치구** 매출 비중이다. 정책 조건과 직접 대응하는 거주 **자치구** 비중도 보조 지표로 낸다. 행정동과 자치구를 혼동하지 않는다. 온라인 지출에는 매장 위치가 없으므로 LV-2/3의 분모에 넣지 않는다. 분모가 0인 표본은 비율을 임의의 0으로 채우지 않는다.

이 산출물은 `comparison=indirect_proxy`, `prepaid_voucher_settlement_verified=false`, `external_magnitude_comparable=false`를 명시한다. 조세재정연구원의 원문은 같은 서울 합성 시민·같은 날짜의 매장 총매출 ON/OFF 효과가 아니며, LV-1의 외부 무효과 동등성 띠도 아직 근거 있게 등록되지 않았다. 따라서 방향·내부 이동량만 진단한다. 예전의 서로 다른 전후 날짜를 정책 효과로 읽는 방식은 채택하지 않는다.

새 P014 실험을 시작하려면 진행 중인 A100 P012의 완결 백업, 별도 ON/OFF 그래프, 정책 전 같은 날짜 차이 진단, 양팔의 동일 예산 보충 맵과 프롬프트 해시가 먼저 필요하다. 프롬프트 후보마다 두 팔을 모두 실행한다. 추출은 **각 팔의 그래프를 초기화하기 전에** 수행한다. 실행 예시는 다음과 같으며 아직 실행 완료를 뜻하지 않는다.

```text
python scripts/report/export_spatial_daily_ledger.py --roster <frozen_roster.json> --start <start> --end <end> --arm on --policy-file data/neo4j_load/policies/P014.json --metrics-dir <ON>/metrics --out <ON>/p014_spatial.jsonl
python scripts/report/export_spatial_daily_ledger.py --roster <frozen_roster.json> --start <start> --end <end> --arm off --policy-file data/neo4j_load/policies/P014.json --metrics-dir <OFF>/metrics --out <OFF>/p014_spatial.jsonl
python scripts/report/paired_local_voucher_effect.py --on <ON>/p014_spatial.jsonl --off <OFF>/p014_spatial.jsonl --roster <frozen_roster.json> --start <start> --end <end> --effect-start <registered_effect_start> --effect-end <registered_effect_end> --json-out <result.json>
```

외부 크기 검증에는 추가로 할인 상품권의 구매 결정을 시민 행동으로 기록하고, 본인 현금과 상품권 액면 잔액, 할인 보조금, 적격 매장 사용액을 거래별로 보존·대조해야 한다. 총소비가 늘지 않는다는 결과를 프롬프트나 소비 엔진의 규칙으로 고정하지 않는다. 원문 측정대상·관측 기간·분모·대조군도 별도로 맞춰야 한다.

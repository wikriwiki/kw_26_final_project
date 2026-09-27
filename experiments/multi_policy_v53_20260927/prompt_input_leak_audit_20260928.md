# v53 정책 실측값 입력 누수 감사 (2026-09-28)

## 판정과 범위

동결된 범용 Stage1 v53 및 중립 Stage2 시스템 프롬프트, 정책 사실/개인 상태 렌더 경로, 정책 입력 JSON, 사회 배경 렌더 경로를 정적으로 점검했다. [재구성 렌더 스냅샷](prompt_input_static_snapshots_20260928.json)은 P010·P012·P013·P014·P016, 거리두기 ON/OFF의 정책 공통 사실과 해당일 사회 배경을 담는다. 정책별 **정책 조건 숫자**(지급액·할인율·한도·사용기간)와 감염 상황은 입력에 있다. 등록 지표의 **실측 성과값**, 출처 논문의 효과 계수, 목표 방향·허용 오차, 평가 결과는 이 스냅샷과 확인한 시스템 프롬프트에 없다. `empirical_registry.json`, `score_multi_policy_proxies.py`, 원문 정답지는 별도 평가 경로이며 `scripts/sim`의 Stage1/Stage2 입력 구성 경로에서 호출하지 않는다.

P010 ON 보존 원장의 `sector.ledger.jsonl.manifest.json`은 범용 Stage1 SHA256 `7ccde1a4ea451a1c18c0b27a4ba30df43afe6c9b7056111fbb7f60280c7e710f`, Stage2 SHA256 `d1996f5013f0c4c948693ec0624b165322227a6330c1e950d262452dcc9d18bf`를 기록한다. 로컬 `prompts.v53.SYSTEM_PROMPT`와 `stage2_poi.NEUTRAL_SYSTEM_PROMPT`의 UTF-8 SHA가 각각 일치한다. P010 동결 정책 파일 SHA256은 `09c54cdfaf26a00f6e4ee91e2d51947b71f75800066c765e6b439c3e9ad1a3ba`이다. 이후 팔은 각 보존 manifest의 입력 지문을 별도로 확인해야 한다.

## 실제 모델에 닿는 경로

`stage1_intent._format_dawn_blocks`는 `DawnContext.to_prompt_blocks`와 `prompts.p012.format_dawn_blocks`를 통해 `policy_facts`, `environment`, 개인 `policy` 상태를 사용자 메시지에 넣는다. `dawn_context.POLICY_CYPHER`가 Graph Policy에서 읽는 열은 이름·유형·**description**·기전 파라미터·효력기간·적용 지역·대상 업종 등이며 `notes`, `render_mode`, `_notes`, 평가 원장은 읽지 않는다. `_format_policy_facts`는 description과 기전의 `facts`를, `_format_policy_status`는 개인 자격·잔액·조건을 조합한다. Stage2도 같은 정책 사실과 상태를 사용한다. `environments.registry`는 날짜별 사회 배경만 반환한다.

| 입력 | 확인한 모델 노출 내용 | 평가값 누수 판정 |
|---|---|---|
| P010 | 계층별 15/30/40만원, 매장 자격, 소멸일 | BOK 소비유발률·업종 사용 비중 없음. `_notes`의 BOK 언급은 Graph 입력에서 제외됨 |
| P012 | 3% 문턱, 초과분 10% 환급, 월 10만원 한도, 적립 업종 | KDI 회귀계수·지급 평균·상한 도달률 없음 |
| P013 | 지원금 제도와 적격 규칙 | KDI 매출 성장률·품목별 효과 없음 |
| P014 | 상품권 10% 할인 구매, 자기 돈 선불, 자치구 제한 | 문헌의 슈퍼마켓/소매 로그계수 없음. `notes`의 해석/기존 시뮬 결과는 렌더되지 않음 |
| P016 | 농축산물 20% 즉시 할인, 1만원 상한, 대상 업종 | 연구의 판매액 효과·품목 비중 수치 없음. `notes`의 연구 설명/기존 결과는 렌더되지 않음 |
| 거리두기 | 정책 팔의 사회 배경은 당시 방역 사실·확진 상황, 대조 팔은 방역 완화 가정 | 서울연구원 업종·상권 매출 감소율 없음 |

이 표의 지급액·할인율은 **정책의 원인/처치 정의**이지 검증하려는 **성과/결과값**이 아니다. Stage2의 동네×업종 평시 단가 앵커도 기초 행동 입력이며 정책효과 정답지는 아니다. 소스의 주석·docstring에는 과거 실험 결과나 실측 논의가 일부 있다(`mechanisms.price_discount`, `mechanisms.sector_voucher`, `stage2_poi` 등). 주석은 프롬프트 문자열로 직렬화되지 않으며, 해당 기전의 반환 문자열을 별도로 확인했다.

## 재현 방법과 한계

스냅샷은 동결/등록 정책 JSON을 Graph loader `10_load_grant_policy.py`와 같은 핵심/기전 파라미터 분리 규칙으로 행으로 바꾸고 `_format_policy_facts`, `environments.registry.build_environment`, `_format_environment`를 호출한 **정적 재구성**이다. 문서의 P010 ON 시스템/정책 SHA는 별도 보존 manifest와 대조했다. 반면 Graph에 실제 붙은 지역·Category 엣지, 개별 시민의 Persona·State·POI 후보, 당시 정확한 완성 HTTP 요청은 이 파일에 들어 있지 않다. 따라서 이 감사는 **확인한 코드 경로와 동결 입력에 평가 수치를 직접 주입하지 않았다는 판정**이지 모든 시민·모든 호출의 전체 프롬프트 바이트 보존 증명은 아니다.

또한 이름이 명시된 과거 정책은 모델 사전학습 지식으로 알려진 결과를 연상시킬 수 있다. 그 가능성은 소스 코드의 직접 주입과 다른 문제이며, 이번 정적 감사만으로 배제할 수 없다. v53의 범용 문구에는 지원금이 미뤄 둔 구매를 앞당길 수도 있다는 일반적 행동 예시가 있어, 성과 숫자 누수는 없어도 행동 방향의 사전 편향을 완전히 배제하지 못한다. 향후 독립 검증에는 정책 이름 익명화 조건과 원시 모델 요청/응답 보존이 필요하다. 이 감사 결과로 현재 숫자 proxy를 실측과 같은 추정량으로 승격하지 않는다.

# 시뮬 뒤 반드시 남아야 하는 것 — 바람이 아니라 검사로

세 가지를 요구받았다. 각각을 **검사**로 바꿨다. 검사가 실패하면 다음 단계로 넘어가지
않는다 — 조용히 반쪽을 남기는 것이 가장 나쁘다.

## 1. 그래프 및 결제 원장

| 무엇 | 어디 | 검사 |
|---|---|---|
| 그래프 전체 | `<arm>/graph_backup/{neo4j,system}.dump` + `config_plugins.tar.gz` | `flock` 으로 중복 방지 · `SHA256SUMS` 생성 후 **즉시 재검증** · 실패 시 `trap` 이 Neo4j 재기동 |
| 업종별 결제 | `<arm>/sector.ledger.jsonl` + manifest | 오프라인 = 업종합 + 미분류 · 적립분 ≤ 적립총액 · 대조군에 정책 결제 0 |
| 캐시백 | `<arm>/cashback.ledger.jsonl` + manifest | 명부 × 날 수 = 행 수 · 부분 달이면 `partial_month_accepted` 기록 |

**"전과정"이 한 덤프에 들어오는 근거**: `:Memory`·`:Plan`·`:State` 노드는 날마다
**누적**되고 덮이지 않는다. 검수 런에서 팔이 끝난 그래프를 **한 번** 읽어 7일치
상태 280건(=40명×7일)이 전부 나왔다. 그것이 증거다.

## 2. 에이전트별 기억메모리 · 과거 지출 · 스케줄

`scripts/report/export_agent_dossier.py` → `<arm>/dossier.jsonl` (+ manifest)

한 사람이 한 줄이고, 그 안에:

    profile   직업·소득·거주동·직장동·소비앵커·업종구성·생활서술
    memories  [:REMEMBERS]->(:Memory)  날짜·요약·왜·무엇을 골랐는지·만족도·금액·정책결제여부
    plans     (:Plan)-[:INCLUDES]->(:POI)  시각·순서·의도·가게·실지출·선택이유·구매상태
    states    날짜별 잔액·이달 누적·적립대상 누적·기분·피로·정책 단계

**검사 (통과 못 하면 파일을 쓰지 않는다)**
- 명부의 사람이 하나라도 그래프에 없으면 중단
- 요구한 날 수만큼 `State` 가 없으면 **중단** (`--allow-missing-days` 로만 기록 후 진행)
- `states == 명부 × 일수` 를 보존 스크립트가 다시 확인하고, 어긋나면 다음 팔로
  **넘어가지 않는다**
- 기억이 0건인 사람 수를 요약에 낸다(그날 외출이 없으면 정상이므로 중단하지 않는다)

**순서가 핵심**: dossier 를 **그래프 덤프·초기화 전에** 뽑는다. 두 팔 설계에서 뒤쪽
팔의 복원이 앞쪽 팔의 그래프를 덮기 때문이다. 확인된 사실 — 라이브 그래프에 앞선
케이스의 State 가 하나도 남아 있지 않았다.

**중간에 죽는 경우**: `tools/watch_dossier_increments.sh` 가 하루가 끝나는 대로 그
하루치를 `<arm>/daily/dossier_<날>.jsonl` 로 읽기 전용으로 내리고 체크섬을 쌓는다.
시뮬을 멈추지 않는다. 런이 5일째 죽어도 4일치가 남는다.

## 3. 검증 지표 비교에 필요한 출력

| 출력 | 무엇을 위해 |
|---|---|
| `score/score_full.json` | 20개 지표 × 자 셋(부호확실성·쌍체부호p·수준) + 필요표본 |
| `eligible_by_sub` / `eligible_by_l1` | 업종별 적립/제외 분해 — 없으면 K10 을 셀 수 없다 |
| `online_by_kdi_attributed` | 제외분의 업종 구성(BDC 안분) — 안분이므로 독립 정보가 아니라고 표에 적는다 |
| `interviews/interviews.jsonl` | 인터뷰 + **원장에 없는 숫자 목록**(환각 검사) |
| `P012_VALIDITY.md` | 20개 표 · 방향 성적 · 한계 · 보존 증거 · 인터뷰를 원장 숫자와 나란히 |

**대조 불가도 행을 갖는다.** 미측정·해당없음·자료없음·판정불가는 이유와 함께 적고,
판정불가는 판정 가능해지는 표본 크기를 함께 낸다. 지우지 않는다.

## 확인 명령

    ssh -i outofmemory.pem -p 10022 outofmemory@123.37.28.167 \
      'B=/data/multipolicy_v53_20260928/p012m; tail -2 /data/p012m.log;
       ls $B/on/day_2021*.json | wc -l;
       ls $B/on/daily/dossier_*.jsonl 2>/dev/null | wc -l'

# generation.md — 매일 실행 로직

> 메인: [`./schedule_generation_plan.md`](./schedule_generation_plan.md)
> POI 준비물: [`data.md`](./data.md)

매일 자정에 실행되는 Daily Planner의 내부 동작.

---

## 1. 자정 배치 실행 흐름

```python
def midnight_batch(sim_date, agents):
    # ① 전날 이벤트 → [:VISITED {satisfaction, importance}] 엣지 MERGE
    finalize_yesterday_memories(sim_date - 1)

    # ② 정책·SNS·지인 캐스케이드 전파 → [:HEARD_RUMOR]·[:SAW_SNS]·[:HEARD_POLICY] 엣지 MERGE
    #    (에이전트간 NL 대화·정책 뉴스는 Graphiti 추출 후 그래프에 적재)
    propagate_sns_posts(sim_date)
    propagate_social_graph(sim_date)

    # ③ Agent.policy_state 갱신 (S0~S5 라이프사이클)
    update_policy_awareness(sim_date)

    # ④ 컨텍스트 빌더 (벌크 Cypher, 60K 대상)
    ctx = build_all_contexts(sim_date, agents)

    # ⑤ 2-Stage LLM 호출 (vLLM 병렬, max_concurrent=128)
    schedules = await gather([
        generate_daily_schedule(
            agent_persona=P[aid],
            fixed_pois=F[aid],
            date_context=dcx,
            memory_context=ctx.mem[aid],       # 30일 Top-5~7
            policy_context=ctx.pol[aid],       # awareness ≥ 0.3 (Agent.policy_state 기반)
            social_context=ctx.soc[aid],       # 14일 Top-3~5 (pending 포함)
            running_state=ctx.run[aid],        # 누적 상태 (쿠폰 조건부)
            zone_hints=ctx.zone[aid],          # intent zone 후보
            # Stage 2 후보는 사전계산 [:NEARBY] 엣지에서 Cypher 조회
            # Stage 2는 위 memory_context·social_context를 그대로 재주입받음
        )
        for aid in agents
        if not checkpoint.done(aid, sim_date)
    ], max_concurrent=128)

    # ⑥ 검증
    valid, failed = validate_batch(schedules)

    # ⑦ 저장
    save_to_db(sim_date, valid)         # agents.db.daily_plans
    save_json(sim_date, valid)          # output/schedules/daily/
    checkpoint.mark_done(sim_date, valid)
```

**⚠️ 순서 중요**: ①~③이 ④~⑤보다 **반드시 먼저** 완료돼야 정책 전파가 당일 계획에 반영된다.

---

## 2. 컨텍스트 빌더 (5종)

프로젝트 본질("정책에 따른 행동 변화")을 구현하는 핵심 단계. 각 에이전트별로 5종 컨텍스트를 수집해 프롬프트에 주입한다.

### 2.1 memory_context

- 소스: `(:Agent)-[:VISITED]->(:POI)` 엣지
- 필터: 전일 ~ **지난 30일**, `importance ≥ 2`
- 정렬: `importance × exp(-days_since/14)` 내림차순
- 선별: Top-5~7
- 포맷: `"2026-05-06 을지로골목(한식) 방문, 만족도 0.8"`
- 토큰 예산: ≤ 500

```cypher
MATCH (a:Agent {id:$aid})-[v:VISITED]->(p:POI)
WHERE v.date >= date()-30 AND v.importance >= 2
RETURN p, v
ORDER BY v.importance * exp(-duration.inDays(date(), v.date)/14) DESC
LIMIT 7
```

### 2.2 policy_context

- 소스 1: `(:Policy)-[:TARGETS]->(:Dong|:District)` 중 에이전트 거주·직장동 적용분
- 소스 2: `Agent.policy_state` JSON 속성 (`{P001:{stage:'S3', score:0.72, updated_at:'2026-05-06'}}`) — S0~S5 라이프사이클 누적
- 소스 3: `(:Agent)-[:HEARD_POLICY]->(:POI)` 엣지 최근 이력 (인지도 가산 근거)

**인지도(awareness) 계산**:
```
awareness = min(1.0,
    정책_경과일 × 도달률_baseline
    + 지인_노출횟수 × 전파계수
    + sns_노출여부 × sns_계수
)
```

- 필터: `awareness ≥ 0.3`만 주입
- 포맷: `"P001 | 강남구 소비쿠폰 10만원 | 음식점·카페 30% 환급 | 인지도 0.7"`
- 토큰 예산: ≤ 200

### 2.3 social_context

- 소스: `(:Agent)-[:HEARD_RUMOR|SAW_SNS]->(:POI)` 엣지, 지난 14일
- 필터: 동일 (Agent→POI) `[:VISITED]` 엣지 부재 → "pending 추천"만 (이미 가본 곳은 다시 추천으로 노출하지 않음)
- 정렬: `importance × exp(-days_since/7)` 내림차순
- 선별: Top-3~5
- 포맷: `"5/2 (D-5) 동료(agent_12003)가 'R_194832 역삼 새 맛집' 추천 — 미방문"`
- 토큰 예산: ≤ 400

```cypher
MATCH (a:Agent {id:$aid})-[r:HEARD_RUMOR|SAW_SNS]->(p:POI)
WHERE r.date >= date()-14
  AND NOT EXISTS { (a)-[:VISITED]->(p) }
RETURN r, p
ORDER BY r.importance * exp(-duration.inDays(date(), r.date)/7) DESC
LIMIT 5
```

> **왜 14일?** 오늘 들은 추천을 다음주에 실행하는 패턴을 허용. 24h 창은 지연 반응을 통째로 버렸음.

### 2.4 running_state

어제 종료 시점 기준 누적 상태. **정책 수혜 없으면 쿠폰 필드 생략**.

```json
{
  "월누적지출": 452000,
  "전일_만족도": 0.72
  // 정책 수혜 중일 때만 추가:
  // "잔여_쿠폰": {"P001": 70000}
}
```

- "최근방문 POI"는 `memory_context`와 중복이므로 **포함하지 않는다**
- 쿠폰 잔액은 `Agent.policy_state[<pid>].coupon_balance`에서 조회, 잔액 0 또는 정책 만료 시 키 전체 생략
- 토큰 예산: ≤ 100

### 2.5 성능 요건

- 60K 에이전트 컨텍스트 수집 목표 **< 5분**
- N+1 쿼리 금지 — 벌크 Cypher + `UNWIND $aids AS aid` 패턴 사용
- 컨텍스트는 메모리에 dict로 보관 후 ⑤ 단계에서 꺼내 씀

---

## 3. 프롬프트 구조 (캐시 블록 분리)

```
[SYSTEM — 캐시 블록 1, 전역 고정]
당신은 서울 시민 에이전트의 하루 동선을 설계합니다.
출력은 주어진 JSON 스키마를 따릅니다.
place 필드는 반드시 [후보 POI]의 이름/ID 중에서만 선택하세요.
새로운 장소를 지어내지 마세요.

이벤트 규칙:
- 첫·마지막 이벤트는 거주지입니다.
- 평일은 출근→근무→퇴근 블록을 반드시 포함 (무직 제외).
- 이벤트 간 최소 체류 20분.
- 요식업은 식사시간대(11:30~14:00, 18:00~21:00)에 집중.

정책·소식·기억 반영 원칙:
- 인지한 정책에 대해 성향·소득분위에 따라 수용/저항 결정.
- 부유층(소비분위 9~10)은 쿠폰성 정책 민감도 낮음.
- 지인 추천은 친밀도와 일치할 때만 반영.
- 전날 만족도 낮은 POI는 오늘 회피.

Pinned POI 사용 (Stage 1):
- `memory_context`의 고만족(≥ 0.7) POI를 재방문하고 싶을 때 → `pinned_poi`에 해당 `poi_id` 지정.
- `social_context`의 pending 추천을 오늘 실행할 때 → 같은 방식으로 pin.
- 그 외 일상 루틴(출근길 편의점·평범한 점심 등)은 pin하지 말 것. Stage 2가 거리순으로 선택.

[USER — 캐시 블록 2, 에이전트별 고정]
## 페르소나
  성별·연령·직업·소비분위·이동활발도·성격 키워드

## 고정 장소
  거주지 {name, lat, lon} / 직장 {name, lat, lon} / 근무시간

## 참조 통계
  평일·주말 업종 Top-5 / 본인 분위 일평균 이벤트 수

## 후보 POI
  [식사/거주근처]   C_... | ... | ... km  (15개)
  [식사/직장근처]   ...
  [카페/직장근처]   ...
  ...

[USER — 변동부, 매일 교체]
## 오늘
  날짜 / 요일 / 공휴일 / 급여일 오프셋

## 최근 경험 (memory_context — 30일)
## 인지한 정책 (policy_context)
## 지인 소식 (social_context — 14일, pending 포함)
## Running state (월누적지출·전일만족도, 정책 수혜 시 잔여쿠폰)
## Intent zone 힌트

→ 오늘 하루 이벤트 로그를 JSON 스키마로 생성하세요.
```

**캐싱 효과**:
- 2-Stage 합산: Stage 1 고정 블록(System+페르소나+고정장소+참조통계+카테고리 ≈ 1,000) + Stage 2 고정 블록(페르소나+고정장소 ≈ 450) — **총 ~1,150 토큰**이 60일 불변
- vLLM prefix cache가 이를 자동 재사용 → Day 2부터 prefill 75% 감소

프롬프트 **원문 스니펫**은 [`prompt.md`](./prompt.md) 참조.

---

## 4. 출력 스키마 (Pydantic)

### Stage 1 출력 — 의도·카테고리·선택적 POI 고정

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class Stage1Event(BaseModel):
    time: str                            # "HH:MM"
    category: str                        # 10 대카테고리
    sub_category: Optional[str] = None   # ~45 서브카테고리
    anchor: str                          # "residence" | "workplace" | "zone:<dong>" | "delivery"
    intent: str                          # ≤ 20자, 의도 한국어 명사/짧은 어구
    pinned_poi: Optional[str] = None     # social/memory context POI enum 중 하나

class Stage1Output(BaseModel):
    events: list[Stage1Event] = Field(min_length=3, max_length=10)
```

- `pinned_poi` 값역은 `guided_json`으로 **해당 에이전트의 memory_context∪social_context POI enum**으로 제한 → 환각 0%.
- pinned_poi가 있으면 Stage 2는 해당 event를 처리하지 않고, 병합기가 그대로 최종 출력에 반영.

### Stage 2 출력 — 미결 event POI 확정

```python
class Stage2Event(BaseModel):
    time: str
    poi_id: str                          # Top-30 enum 중 하나
    place: str
    purpose: str                         # ≤ 20자

class Stage2Output(BaseModel):
    events: list[Stage2Event]            # 미결 event만, Stage 1 순서 유지
```

### 최종 DailySchedule (병합 후)

```python
class Event(BaseModel):
    time: str
    poi_id: str
    place: str
    purpose: str

class DailySchedule(BaseModel):
    agent_id: int
    sim_date: str
    day_type: Literal["weekday","weekend","holiday"]
    events: list[Event] = Field(min_length=3, max_length=10)
    source_context: dict                 # 감사용, 입력 컨텍스트 요약
```

구조화 출력 강제 방식:
- **vLLM**: `guided_json={"schema": Stage1Output.schema()}` / `Stage2Output.schema()`
- **Qwen API**: `response_format={"type":"json_object"}` + Pydantic 재검증

---

## 5. 검증 규칙

| 규칙 | 위반 시 조치 |
|---|---|
| JSON 스키마 파싱 성공 | 재생성 |
| 모든 `poi_id`가 `:POI` 노드에 존재 (Cypher `MATCH`) | 재생성 |
| 모든 `poi_id`가 해당 에이전트의 `[:NEARBY]`·`[:LIVES_AT]`·`[:WORKS_AT]` 대상에 포함 | 재생성 |
| `time` 단조 증가 | 재생성 |
| 이벤트 간 간격 ≥ 20분 | 재생성 |
| 첫·마지막 이벤트 = 거주지 | 재생성 |
| 평일 + 직장유 → 09~18 중 직장 체류 ≥ 4h | 재생성 |
| 업종 운영시간 위반 | 시간 ±30분 이동 시도 → 실패 시 재생성 |
| 이벤트 수 범위 (평일 5~9, 주말 3~8) | 재생성 |

### 재시도 정책

- 최대 **3회**, temperature 0.7 → 0.9 → 1.0
- 최종 실패 → 빈 스케줄(거주지 체류만) + 경고 로그

---

## 6. 체크포인트·재개

`progress.sqlite`:

```sql
CREATE TABLE progress (
    agent_id     INTEGER,
    sim_date     TEXT,
    status       TEXT,        -- "done" | "failed" | "empty"
    retry_count  INTEGER,
    cost_tokens  INTEGER,
    error        TEXT,
    PRIMARY KEY (agent_id, sim_date)
);
```

- `--resume` 옵션: `status != 'done'` 조합만 재실행

---

## 7. 일일 완료 기준

- 60,000 중 **≥ 99% 검증 통과** → 다음 날 진행
- 실패 1%는 3회 재시도 후 빈 스케줄 기록 + 경고 로그

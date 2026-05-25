# NVIDIA 페르소나 결합 — 3가지 방식 정리 (팀 공유용)

> **TL;DR** — 기존 통계 페르소나가 "숫자만 있고 사람 냄새가 없다"는 문제를
> 풀려고 NVIDIA Nemotron-Personas-Korea(정성 서사)를 우리 BDC 통계(정량 소비·이동)에
> 붙였습니다. 붙이는 방법이 3가지이고, **LLM 호출 없이** 전부 결정적으로 생성됩니다.
> 결론부터: **방식 B(조건부 부여) 또는 C(B+봉합) 권장.** 방식 A(순위 짝짓기)는
> 작은 셀에서 모순 페르소나를 만들 위험이 있습니다.

---

## 1. 왜 이걸 하나 (배경)

지금까지 에이전트 페르소나는 **BDC 통계 기반 숫자 묶음**이었습니다.
소비분위·이동거리·배달일수 같은 정량값은 정확하지만:

- 에이전트의 **개성·서사가 없음** → 계획(스케줄) 생성 시 "왜 이 사람이 이렇게
  행동하는지" 설명력이 약함.
- LLM에게 "이 사람이 되어 하루를 계획하라"고 줄 때 **입력이 빈약** → 결과가
  전형적·획일적.

NVIDIA Nemotron-Personas-Korea는 그 반대입니다.

| | 우리 BDC 통계 | NVIDIA Nemotron-Personas-Korea |
|---|---|---|
| 강점 | 동·코호트 단위 **정량**(소비/이동/배달) | 1인 단위 **정성**(서사·취미·가치관·직업관) |
| 약점 | 개성·서사 없음 | 소비 금액·이동 같은 행동 수치 없음 |
| 규모 | 서울 행정동 × 성·연령 셀 | 약 100만 명(한국), KOSIS 인구통계 기반 |
| 라이선스 | 내부 | CC BY 4.0 (출처 표기 시 사용 가능) |

→ **둘은 상호보완.** NVIDIA의 "사람"에 우리 통계의 "소비·행동"을 입히면
빈약함이 해소됩니다. 문제는 *어떻게* 붙이느냐입니다.

---

## 2. 핵심 난제: 정성 ↔ 정량을 어떻게 짝지을까

두 데이터에는 공통 키(개인 ID)가 없습니다. 그래서 "이 NVIDIA 사람"과
"이 통계 셀"을 연결할 다리가 필요합니다. 우리가 쓰는 다리는 **SES proxy**
(사회경제적 지위 추정치, 0~1)입니다.

```
SES_proxy = 0.35 × 학력 + 0.40 × 직업tier + 0.25 × 주거유형   (→ 0~1 정규화)
```

NVIDIA 레코드에서 SES를 계산하고, 그걸로 소비 수준과 연결합니다.
**연결 방식의 차이가 곧 3가지 방식의 차이**입니다.

---

## 3. 세 가지 방식

### 방식 A — rank-coupling (순위 짝짓기)

```
NVIDIA 사람들을 SES로 줄 세움  ─┐
                               ├─→ 같은 등수끼리 짝짓기
통계 페르소나를 소비분위로 줄 세움 ─┘
```

- **작동**: (구·성·연령) 셀 안에서 NVIDIA 사람은 SES 순위, 통계 페르소나는
  소비 백분위로 각각 정렬한 뒤 **같은 등수끼리 매칭**. "돈 수준으로 줄 세워 짝짓기".
- **장점**: 통계 marginal(셀별 소비 분포)이 그대로 보존됨. 구현 단순.
- **단점 (치명적)**: 두 데이터를 **독립이라고 가정**하고 교차결합 → 작은 셀에서
  **모순 페르소나** 발생. 예) SES는 높게 나왔는데 그 셀의 소비분위가 낮으면
  "명품 좋아하는 서사 ↔ 편의점 소비" 같은 조합이 강제로 묶임.
- **메타**: `_match.method = rank-coupling`, `match_level`, `consume_percentile`

### 방식 B — conditional-graft (조건부 부여) ⭐ 권장

```
NVIDIA 사람 1명(진짜 사람)  ─→  그 사람의 동네/SES를 보고
                                "이 사람이라면 소비를 이만큼" 통계를 조건부로 입힘
```

- **작동**: NVIDIA 사람을 **base(진짜 한 명)** 로 두고, 거기서 파생.
  1. **동네**: 그 사람의 구 안에서 (성·연령) 인구비례로 행정동 추첨
  2. **소비**: 그 셀의 대표 소비분위 ± **SES 힌트**(`shift = round((SES−0.5)×4)`, 최대 ±2분위)
  3. **업종**: 셀 업종비율 + **취미 키워드 보정**(예: "와인·골프" → 여가/외식 가산)
  4. **행태**: 셀 telecom 분포에서 배달·이동·재택 샘플
- **장점**: 짝짓기가 아니라 *한 사람한테서 파생* → **모순 원천 차단**.
  SES·취미 보정은 on/off 옵션이라 끄면 셀 marginal 그대로(가장 안전).
- **단점**: SES 힌트가 ±2분위 보정뿐 → 극단값(전문직인데 저소비 동네)에서 **잔여 gap**이 남을 수 있음.
- **메타**: `_match.method = conditional-graft`, `dong_pick_level`, `ses_hint`, `hobby_adjust`

### 방식 C — hybrid (B + 규칙기반 모순 봉합) ⭐⭐ 가장 견고

```
방식 B 결과  ─→  규칙으로 모순 검출  ─→  소비분위를 SES 방향으로 봉합  ─→  잔여 경고 기록
```

- **작동**: B를 돌린 뒤 `reconcile` 레이어 추가.
  - **검출 규칙 3종**:
    1. SES ↔ 소비 gap (정규화 차이 > 0.4)
    2. 서사 ↔ 소비 모순 (고급 취미↔저소비 / 검소 취미↔고소비)
    3. 직업 ↔ 소비 (고SES 전문직인데 극저소비 — 단 구직/전직/은퇴는 제외)
  - **봉합**: gap 크면 소비분위를 SES 목표치 쪽으로 **최대 ±2** 당기고 금액·성향 재계산
  - **감사 로그**: 봉합 후에도 남는 모순은 `_match.warnings`에 기록 → 사후 검증 가능
- **장점**: B의 잔여 모순까지 교정 + **추적 가능**(어떤 페르소나가 봉합됐는지/경고가 남았는지).
- **단점**: 규칙·임계값(gap 0.4, max_pull 2)이 휴리스틱 → 튜닝 여지.
- **메타**: B와 동일 + `reconciled`(봉합 여부), `warnings`(잔여 모순)

---

## 4. 한눈 비교표

| 항목 | A · rank-coupling | B · conditional-graft | C · hybrid |
|------|-------------------|-----------------------|------------|
| 결합 원리 | SES 순위 짝짓기 | NVIDIA 1명서 파생 | B + 모순 봉합 |
| 독립성 가정 모순 | ⚠️ 있음(작은 셀 위험) | ✅ 없음 | ✅ 없음 |
| 통계 marginal 보존 | ✅ 강함 | ✅ (옵션 끄면 완전) | ✅ |
| 잔여 SES-소비 gap | 매칭에 의존 | △ 일부 남음 | ✅ 봉합 |
| 모순 추적/감사 | ✗ | ✗ | ✅ `warnings` |
| LLM 호출 | 0 | 0 | 0 |
| 결정성(seed) | ✅ | ✅ | ✅ |
| 구현 복잡도 | 낮음 | 중간 | 중간+ |
| **권장도** | △ | ⭐ | ⭐⭐ |

---

## 5. 실제 예시 (방식 C 출력 1명)

```json
{
  "agent_id": "AGT_11710720_M_20대_000",
  "residence": { "dong": "잠실7동", "gu": "송파구" },
  "personal": { "age_group": "20대", "gender": "M",
                "job": "전직 양식 조리사, 현재 구직중",
                "income_level": "중하", "life_stage": "사회초년생" },
  "spending": { "weekday_spending_level": 4, "daily_spending_weekday": 76770,
                "weekday_top_categories": {"편의점":0.41,"쇼핑":0.32,"식사":0.20} },
  "behavior": { "delivery_days": 15.8, "home_hours_weekday": 11.5, "mobility_level": 6 },
  "nvidia_persona": {
    "summary": "김찬민 씨는 정교한 양식 조리 기술을 가진 22세 구직자로, 송파구에서 홀로 지내며 정적인 휴식과 요리에 대한 자신만의 기준을 지키며 살아가는 청년입니다.",
    "hobbies": ["요리 유튜버 영상 분석", "동네 목욕탕 방문", "올림픽공원 가벼운 산책"]
  },
  "_match": { "method": "conditional-graft", "nvidia_ses": 0.675,
              "reconciled": true, "warnings": [] }
}
```

해석: NVIDIA가 준 **"양식 조리사 출신 구직 청년" 서사**에 우리 통계가
**소비분위 4 · 배달 월 15.8일 · 재택 11.5시간/일**을 입혔습니다.
구직 중이라 소득이 낮은 게 자연스럽고, 검출 규칙이 "구직"을 인식해
고SES-저소비 모순으로 **잘못 잡지 않았습니다**(`warnings: []`).

---

## 6. 페르소나 구조 & LLM 입력 정책

모든 방식이 동일한 출력 스키마를 따릅니다.

- `residence / personal / workplace / spending / behavior / personality` — 정량·기본
- `nvidia_persona` — **LLM 입력용** 정성 필드: `summary`, `hobbies`,
  `cultural_background`, `marital_status`, `housing_type`, `family_type`, `education_level`
- `nvidia_reserved` — **저장 전용(LLM 입력 X)**: `professional_persona`,
  `career_goals_and_ambitions` 등. *나중에 필요할 수 있어 확보만 해 둠.*
- `_match` — 방식별 추적 메타

> ⚠️ **정책**: 직업관·커리어목표(`nvidia_reserved`)는 페르소나에 **보관은 하되
> LLM 프롬프트에는 넣지 않습니다.** 스케줄 생성에 직접 필요하지 않아 토큰 절약 +
> 과적합 방지 목적.

---

## 7. 재현 방법

```bash
# 방식 A — rank-coupling
python scripts/persona/build_rank_coupling.py --limit 10

# 방식 B — conditional-graft
python scripts/persona/build_conditional.py --limit 10
#   옵션: --no-ses-hint (SES 보정 끔, 가장 안전) / --no-hobby-adjust

# 방식 C — hybrid (B + 봉합)
python scripts/persona/build_conditional.py --limit 10 --reconcile

# 전체(약 15,000명): --limit 생략
```

- 출력: `output/personas/samples/{A_rank_coupling, B_conditional_graft, C_hybrid}.json`
- 전 코드 **LLM 호출 0**, seed=42 결정적.
- 단위 테스트: `pytest tests/unit/persona/` (37개 통과)

---

## 8. 한계 & 후속

- **NVIDIA 서울 서브셋 필요**: 현재 fixture는 120명 샘플. 전체 생성 시
  `gu_sex_age` 매칭률을 올리려면 서울 서브셋(약 13만)을 받아야 함
  → `data/personas/README.md` 참고. (샘플이 작으면 `sex_age` 폴백 매칭이 늘어남.)
- **workplace 미정**: 직장 동/출퇴근은 현재 `null` (후속 단계에서 부여).
- **C의 임계값은 휴리스틱**: `gap_threshold=0.4`, `max_pull=2`는 실측으로 튜닝 가능.
- **출처 표기 의무**: NVIDIA 데이터는 CC BY 4.0 — 산출물 공개 시 출처 명시.

---

### 부록: 방식 선택 가이드

- **빠른 baseline / marginal 정확성만 중요** → B(옵션 끄고)
- **개성 살리되 모순 최소화** → B(옵션 켜고)
- **대회·발표용, 품질·검증 추적 중요** → **C 권장**
- A는 비교 실험용으로만 유지 (작은 셀 모순 위험 때문에 본 생성엔 비권장)

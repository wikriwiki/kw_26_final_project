# 소비행동 프롬프트 일반화 — 방법론 근거

우리가 하려는 것(**정책 여러 개의 실측 방향에 소비 프롬프트 하나를 맞추고, 정답을 열지
않은 정책 하나로 검증**)은 이름이 붙어 있는 방법이다. 즉흥 절차가 아니라 기존 문헌의
표준 설계에 해당하며, 문헌이 경고하는 함정 중 우리에게 실제로 걸리는 것이 둘 있다.

---

## 1. 우리 설계의 정체 — Pattern-Oriented Modeling (POM)

Grimm et al. (2005), *Pattern-Oriented Modeling of Agent-Based Complex Systems:
Lessons from Ecology*, Science.

POM 의 핵심 주장은 **하나의 출력 변수를 맞추는 것으로는 모형이 유효하지 않다**는 것이다.
서로 다른 수준·규모에서 관측된 **여러 패턴을 동시에** 재현해야 하며, 각 패턴이 후보 모형을
거르는 **필터**로 작동한다. 복수의 그럴듯한 모형을 만들어 패턴을 재현하는 것만 남긴다.

| POM | 우리 |
|---|---|
| 여러 수준·규모의 관측 패턴 | 정책 6개 × 지표 다수 (업종·상권·이동·분위) |
| 각 패턴 = 후보 모형 필터 | 각 지표 부호 = 후보 프롬프트 필터 |
| 복수 후보 모형 중 선택 | 후보 프롬프트 6개 중 선택 |
| 하나의 패턴만 맞추면 무효 | 한 정책만 맞추면 무효 |

**우리 설계는 POM 의 정책 버전이다.** Topping et al. (2012, PLOS One) 은 **이미 만들어진
대형 모형을 사후에 패턴으로 검정·조정**한 사례로, 시뮬레이터는 있고 프롬프트만 조정하는
우리 상황과 정확히 같다.

---

## 2. 수치가 아니라 부호를 맞추는 근거

계량경제학의 **적률법 시뮬레이션(MSM)** 과 **간접추론(II)** 은 모형 파라미터를 "시뮬 적률이
실측 적률과 맞도록" 고른다(Fagiolo, Moneta & Windrum 2007 의 표준 정리). 우리는 적률 대신
**부호와 순위**를 쓴다. 이유가 둘이다.

**① 우리 구조가 못 내는 크기가 있다.** 하루 총액이 페르소나 앵커에 묶여 있어 실측의
총소비 −14% 같은 값은 구조적으로 표현되지 않는다. 크기를 목표로 주면 프롬프트가 그걸
억지로 짜내고, 홀드아웃에서 무너진다.

**② 부호는 작은 표본에서도 안정적이지만 크기는 아니다.** 같은 코드로 돌린 두 런에서
1분위 MPC 가 0.216 과 0.045 로 갈린 적이 있다. 크기를 맞추려 들면 노이즈를 맞추게 된다.

지표 k개 × 정책 6개 = 6k 개 부호 비교이고 무작위면 일치율 50% 다. 80% 를 맞추면
이항검정으로 유의하다고 말할 수 있다.

> Fagiolo 계열 문헌이 공통으로 요구하는 것: **요약통계를 사전에 선정할 것.**
> 이것이 채점표 사전등록의 근거다. 사후에 지표를 고르면 추정량의 성질이 보장되지 않는다.

---

## 3. 후보 평가 예산 배분 — Sequential Halving / TRIPLE ★적용

프롬프트 선택을 **고정예산 최적팔 식별(fixed-budget best arm identification)** 로 푸는
연구가 있다 — TRIPLE (*Efficient Prompt Optimization Through the Lens of Best Arm
Identification*, arXiv:2402.09723). 후보 평가 비용을 명시적 예산 제약으로 다룬다.

**Sequential Halving**: 예산을 후보에 균등 배분 → 평가 → 상위 절반만 다음 단계로 승격 →
나머지 가지치기 → 반복. 유망한 후보에 **지수적으로 많은 예산**이 몰린다.

우리는 후보 6개 × 정책 6개 = 36 런(약 54시간)을 균등하게 돌릴 이유가 없다.

| 단계 | 후보 | 정책 | 런 길이 | 비용 |
|---|---|---|---|---|
| 1 | 6 | 2 (기전이 가장 다른 둘) | 최단 (100명·4일) | 12런 |
| 2 | 3 | 4 | 짧게 (150명·5일) | 12런 |
| 3 | 1~2 | 6 전부 | 표준 (150명·6일) | 6~12런 |

같은 예산으로 최종 후보를 훨씬 신뢰도 높게 고를 수 있다. **이건 지금 계획을 바꿔야 할
첫 번째 지점이다.**

**분산 축소 — 공통난수(Common Random Numbers)**: 후보 간 비교는 **같은 에이전트 표본·
같은 시드**로 돌려 쌍대 비교로 만든다. 후보마다 다른 표본을 쓰면 프롬프트 차이와 표본
차이가 섞인다. 시뮬레이션 비교의 기본 기법이며 비용이 0이다.

---

## 4. 자동 프롬프트 최적화 — 못 쓰는 것과 가져올 것

| 기법 | 방식 |
|---|---|
| **OPRO** (Yang et al. 2023) | LLM 이 프롬프트를 제안 → 점수 보고 개선 |
| **APO** (Pryzant et al. 2023) | 텍스트 "gradient" + beam search |
| **EvoPrompt** | 진화 알고리즘 — 변이·선택 |
| **DSPy** (Stanford) | 파이프라인을 컴파일러가 최적화 |
| **TextGrad** (Yuksekgonul et al. 2024) | 텍스트 피드백을 gradient 처럼 역전파 |

**전자동 탐색은 못 쓴다.** 이 기법들은 후보 하나를 초 단위로 평가할 수 있다는 전제 위에
있는데 우리는 후보 1회 평가에 수 시간이고, DSPy·TextGrad 는 **샘플 단위 정답 레이블**을
요구하는데 우리 정답은 집계 지표다(에이전트 한 명의 하루 계획에는 정답이 없다).

**그래도 가져올 것이 셋 있다.**

**① OPRO 의 제안 루프를 사람 손으로 돌린다.** OPRO 의 핵심은 "이전 프롬프트들과 그 점수를
같이 보여 주고 다음 후보를 제안하게 한다"는 것이다. 후보를 6개 한꺼번에 적지 말고,
**1단계 결과(어떤 문구가 어느 정책에서 어떤 부호를 냈는지)를 정리해 다음 후보를 짓는
근거로 쓴다.** 자동화가 아니라 절차의 차용이다.

**② APO 의 텍스트 gradient — "무엇이 틀렸는지"를 문장으로 남긴다.** 실패한 후보마다
"어느 정책·어느 지표에서 어느 방향으로 틀렸는가"를 기록하고, 그 진단이 다음 후보의
수정 방향이 된다. 실패 로그가 자산이 된다.

**③ 후보 수를 미리 못박는 것은 여전히 유효하다.** 실효 자유도는 글자 수가 아니라
시도한 후보 수다. 단계별로 후보를 추가할 때마다 자유도가 늘어난다는 점을 기록한다.

---

## 5. 견고성 시험 — 파라미터 섭동 ★적용

**BATprompt** (적대적 섭동으로 프롬프트를 견고화) 의 발상을 우리 쪽으로 옮기면,
**정책 파라미터를 조금 흔들어도 방향이 유지되는가**를 본다.

    캐시백률   10%   → 8% / 12%
    문턱      1.03  → 1.02 / 1.05
    한도      10만원 → 8만 / 12만

작은 파라미터 변화에 방향이 뒤집히면 그 프롬프트는 취약하고 홀드아웃에서 버티지 못한다.
런 길이가 같아 비용이 늘지 않으며, 최종 후보 1~2개에만 적용하면 충분하다.

> 문헌의 경고 하나: **프롬프트는 최적화에 쓴 모델에 과적합된다.** 다른 모델로 옮기면
> 이득이 크게 줄어든다. 모델을 EXAONE-4.5 로 고정하는 것은 제약이 아니라 필수 조건이다.
> (*Concentrate Attention: Towards Domain-Generalizable Prompt Optimization*, arXiv:2406.10584)

---

## 6. Lookahead bias — 우리에게 실제로 걸리는 함정 ★★

Sarkar & Vafa, *Lookahead Bias in Pretrained Language Models* (ICML 2025).

사전학습 데이터에 **미래 정보가 들어 있어**, 과거 정보만 써야 할 분석에 그것이 샌다.
저자들이 기업 실적 발표 위험요인 예측과 선거 결과 예측에서 실증했고, 결론이 매섭다.

> **과거 시점 경계를 지키라고 명시적으로 지시해도 막히지 않는다.** 모델은 여전히
> 암기 수준(recall-level) 정확도를 낸다. 백테스트에서 LLM 이 큰 사건을 맞혔을 때,
> 그것이 당시 정보에서 나온 것인지 사전학습에서 본 결과인지 구분할 수 없다.

**우리는 2020~2021 정책을 2026년 모델로 백테스트한다.** 긴급재난지원금·거리두기·
상생소비지원금의 결과는 모두 EXAONE 학습 시점 이전에 공개됐다. 정면으로 해당한다.

문헌의 정공법은 **시점 경계 모델**(DatedGPT, Chronologically Consistent LLMs — 지식
컷오프 이전 데이터로만 학습)인데, 우리는 모델을 바꿀 수 없다. 대신 셋을 한다.

**① 정책 이름을 주지 않는다.** "상생소비지원금"이 아니라 기전만 준다 — "이번 달 카드
사용액이 기준을 넘으면 초과분의 10%를 다음 달에 돌려받는다". 정책 JSON 을 기전
파라미터로 쪼개는 작업(§기전 모듈)이 그 자체로 완화책이 된다. **이름·연구기관·보도
문구가 프롬프트에 들어가지 않는지 렌더 결과로 확인한다.**

**② 플라세보 정책 검정** (§7). 실재하지 않은 가짜 정책에도 기전에 맞게 반응하면
기전을 처리하는 것이고, 실제 정책에만 반응하면 답을 아는 것이다.

**③ 직접 질의로 노출 정도를 재고 보고서에 적는다.** 모델에게 각 정책의 효과를 직접
물어 무엇을 아는지 기록한다. 감추는 것보다 명시하는 편이 방어가 된다.

> 이건 감점 요인이 아니라 **우리가 먼저 짚었다는 점이 가점 요인**이다. 이 분야 논문
> 다수가 이 문제를 다루지 않는다.

---

## 7. 플라세보·음성대조 (negative control)

인과추론의 표준 도구다 — 가정이 맞다면 **효과가 없어야 하는 자리**에 주 분석을 그대로
적용해 본다. 효과가 나오면 설계가 틀린 것이다.

| 위약 | 설계 | 무엇을 잡나 |
|---|---|---|
| **시점 위약** | 정책 시행 **전** 구간에 가짜 발효일 | 교란·워밍업 추세. **A6 평행추세 검정이 이미 이 형태다** |
| **가짜 정책 위약** | 실재하지 않은 기전 주입 (예: 문구·도서 20% 환급) | **lookahead bias** — 기전 처리인가 암기인가 |
| **무효 지표** | 정답이 "유의하지 않음"인 지표 | "정책이면 효과가 난다"로 수렴한 프롬프트 |

세 번째는 이미 확보돼 있다 — P012 제외업종(+2.85% ns), 사적모임 인원 제한(ns),
지역사랑상품권 총액(무영향). **무효가 정답인 지표가 채점표의 방어선이다.**

---

## 8. 선행연구에서 우리 위치

| 연구 | 한 일 | 한계 / 우리와의 관계 |
|---|---|---|
| Horton (2023) *Homo Silicus* (ACM EC 2024) | LLM 에 부존·정보·선호를 주고 경제 실험을 in silico 재현 | **실험실 실험 재현이지 실제 정책 결과 대조가 아님** |
| **EconAgent** (ACL 2024) | LLM 에이전트 거시경제 시뮬 | 저자 명시 한계 — **"미묘한 정책 변화에 대한 현실적 행동 반응 부족"** |
| 생성 에이전트 검증 연구 | 실험실 실험과 r = 0.85~0.90 | 출력이 편향·변동성 부족·**프롬프트 민감** |
| **비판적 리뷰** (AI Review 2025) *Validation is the central challenge for generative social simulation* | LLM-ABM 35편 체계적 리뷰 | 아래 |

비판적 리뷰가 든 이 분야의 세 가지 병폐가 우리 설계의 좌표를 그대로 찍어 준다.

> **35편 중 15편이 주관적 '그럴듯함(believability)' 평가에만 의존**했고, 22편이 그것을
> 주된 검증 방법으로 삼았다. 실증 데이터와의 정량 비교는 드물다.
> **거의 모든 연구가 단일 시뮬레이션 런의 결과만 보고**한다 — 사례 하나로 결론을 내리는 것과 같다.
> LLM 은 블랙박스성·문화 편향·확률적 출력 때문에 ABM 검증 난제를 **완화하는 게 아니라 악화**시킨다.

| 리뷰가 지적한 병폐 | 우리 대응 |
|---|---|
| 주관적 그럴듯함 평가 | 실측 정책 결과와 **부호 정량 대조** |
| 단일 런 보고 | **다중 시드**, 공통난수 쌍대 비교 |
| 단일 사례 | **정책 6개 + 홀드아웃 1개** |
| 기전과 느슨하게 연결된 결과 지표 | 기전 모듈별 파라미터 → 지표 대응을 명시 |

**두 번째 줄이 지금 계획을 바꿔야 할 두 번째 지점이다.** 지금까지 우리는 사이클마다
런을 한 번씩만 돌렸다. 최종 후보 판정은 **시드를 바꾼 복수 런**으로 해야 한다.

---

## 9. 실행 절차

1. **패턴 선정** — 부호가 명확하고 우리 구조가 산출 가능한 지표만. 못 내는 지표는 사전
   제외(`docs/POLICY_ANSWERKEY_MATRIX.md` §3). 무효가 정답인 지표를 반드시 포함.
2. **후보 집합 고정** — 소비 프롬프트 후보를 **먼저 적어 둔다.** 단계별로 추가하면
   자유도가 늘어난다는 점을 기록.
3. **Sequential Halving 으로 거른다** — 6후보×2정책 최단런 → 3후보×4정책 → 1~2후보×6정책.
   후보 간 비교는 **공통난수**(같은 에이전트·같은 시드)로.
4. **견고성 시험** — 최종 후보에 정책 파라미터 섭동을 걸어 방향이 유지되는지 본다.
5. **위약 검정** — 시점 위약 + 가짜 정책 위약(lookahead bias).
6. **다중 시드** — 최종 판정은 시드를 바꾼 복수 런으로.
7. **홀드아웃** — 8대 소비쿠폰을 **딱 한 번**. 돌려보고 고치면 홀드아웃이 아니다.
8. **보고** — 실패해도 그대로. 정직한 실패가 한 정책에 맞춘 성공보다 낫다.

---

## 참고문헌

**모형 검증·보정**
- Grimm, V. et al. (2005). Pattern-Oriented Modeling of Agent-Based Complex Systems:
  Lessons from Ecology. *Science* 310(5750), 987-991.
- Topping, C. J. et al. (2012). Post-Hoc Pattern-Oriented Testing and Tuning of an
  Existing Large Model. *PLOS One* 7(9), e45872.
- Piou, C. et al. (2009). Pattern-oriented modelling: a 'multi-scope' for predictive
  systems ecology. *Phil. Trans. R. Soc. B*.
- Fagiolo, G., Moneta, A. & Windrum, P. (2007). A Critical Guide to Empirical Validation
  of Agent-Based Models in Economics. *Computational Economics* 30, 195-226.

**LLM 사회 시뮬레이션**
- Horton, J. J. (2023). Large Language Models as Simulated Economic Agents: What Can
  We Learn from Homo Silicus? *ACM EC 2024*. arXiv:2301.07543
- Li, N. et al. (2024). EconAgent: LLM-Empowered Agents for Simulating Macroeconomic
  Activities. *ACL 2024*. arXiv:2310.10436
- (2025). Validation is the central challenge for generative social simulation: a
  critical review of LLMs in agent-based modeling. *Artificial Intelligence Review*.
  doi:10.1007/s10462-025-11412-6
- (2025). Towards Operational Validation of LLM-Agent Social Simulations. arXiv:2508.21740

**프롬프트 최적화·선택**
- Yang, C. et al. (2023). Large Language Models as Optimizers (OPRO). arXiv:2309.03409
- Pryzant, R. et al. (2023). Automatic Prompt Optimization with "Gradient Descent" and
  Beam Search. arXiv:2305.03495
- Yuksekgonul, M. et al. (2024). TextGrad: Automatic "Differentiation" via Text. arXiv:2406.07496
- Shi, C. et al. (2024). Efficient Prompt Optimization Through the Lens of Best Arm
  Identification (TRIPLE). arXiv:2402.09723
- Jamieson, K. & Talwalkar, A. (2016). Non-stochastic Best Arm Identification and
  Hyperparameter Optimization. *AISTATS*. (Successive Halving)
- (2024). Concentrate Attention: Towards Domain-Generalizable Prompt Optimization.
  arXiv:2406.10584
- Ramnath, K. et al. (2025). A Systematic Survey of Automatic Prompt Optimization
  Techniques. *EMNLP 2025*. arXiv:2502.16923

**시간 누출**
- Sarkar, S. K. & Vafa, K. (2025). Lookahead Bias in Pretrained Language Models.
  *ICML 2025*. SSRN 4754678

# 소비행동 프롬프트 일반화 — 방법론 근거

우리가 하려는 것(**정책 여러 개의 실측 방향에 소비 프롬프트 하나를 맞추고, 정답을 열지
않은 정책 하나로 검증**)은 이름이 붙어 있는 방법이다. 즉흥적으로 만든 절차가 아니라
기존 문헌의 표준 설계에 해당한다는 점을 정리해 둔다.

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

즉 **우리 설계는 POM 의 정책 버전**이다. "P012 하나를 잘 맞췄다"가 왜 불충분한지,
"6개 정책의 방향을 동시에 맞춘다"가 왜 강한 주장인지가 이 틀에서 바로 설명된다.

관련: Piou et al. (2009) *Pattern-oriented modelling: a 'multi-scope' for predictive
systems ecology* · Topping et al. (2012) *Post-Hoc Pattern-Oriented Testing and Tuning
of an Existing Large Model*, PLOS One — **이미 만들어진 대형 모형을 사후에 패턴으로
검정·조정**하는 사례로, 우리 상황(시뮬레이터는 이미 있고 프롬프트만 조정)과 같다.

---

## 2. 수치가 아니라 부호를 맞추는 근거

계량경제학의 **적률법 시뮬레이션(Simulated Method of Moments)** 과 **간접추론(Indirect
Inference)** 은 모형 파라미터를 "시뮬 적률이 실측 적률과 맞도록" 고른다. 우리는 적률 대신
**부호와 순위**를 쓴다.

왜 약한 기준을 쓰는가 — 두 가지 이유가 있고 둘 다 우리 상황에서 실질적이다.

**① 우리 구조가 못 내는 크기가 있다.** 하루 총액이 페르소나 앵커에 묶여 있어 실측의
총소비 −14% 같은 값은 구조적으로 표현되지 않는다. 크기를 목표로 주면 프롬프트가 그걸
억지로 짜내고, 홀드아웃에서 무너진다.

**② 부호는 작은 표본에서도 안정적이지만 크기는 아니다.** 같은 코드로 돌린 두 런에서
1분위 MPC 가 0.216 과 0.045 로 나온 적이 있다. 크기를 맞추려 들면 노이즈를 맞추게 된다.

부호 기준의 통계적 의미는 명확하다. 지표 k개 × 정책 6개 = 6k 개 부호 비교이고, 무작위면
일치율 50% 다. 80% 를 맞추면 이항검정으로 유의하다고 말할 수 있다.

---

## 3. 자동 프롬프트 최적화 — 왜 못 쓰는가

LLM 쪽에는 프롬프트를 자동으로 최적화하는 기법군이 있다.

| 기법 | 방식 |
|---|---|
| **OPRO** (Yang et al. 2023, DeepMind) | LLM 이 프롬프트를 제안 → 점수 보고 개선. 블랙박스 최적화 |
| **EvoPrompt** | 진화 알고리즘 — 프롬프트 집단을 변이·선택으로 진화 |
| **DSPy** (Stanford) | 파이프라인을 계산 그래프로 보고 컴파일러가 최적화 (BootstrapFewShot·MIPRO) |
| **TextGrad** (Yuksekgonul et al. 2024) | 텍스트 피드백을 gradient 처럼 역전파 |
| 서베이 | *A Systematic Survey of Automatic Prompt Optimization Techniques* (EMNLP 2025) |

**우리 조건에서는 전부 못 쓴다. 이유가 둘이다.**

**① 평가가 너무 비싸다.** 이 기법들은 후보 하나를 초 단위로 평가할 수 있다는 전제 위에
있다. 우리는 후보 하나를 정책 6개에서 각각 짧은 런(약 90분)으로 돌려야 하므로 평가
1회에 9시간이다. 탐색을 돌릴 수 있는 예산이 아니다.

**② 정답이 샘플 단위로 없다.** DSPy·TextGrad 는 입력 하나하나에 정답 레이블이 있어야
한다. 우리 정답은 **집계 지표**(업종별 증감, 상권별 차이)이고 개별 에이전트 응답에는
정답이 없다. 에이전트 한 명의 하루 계획이 "맞았다/틀렸다"를 판정할 근거가 없다.

→ **후보를 사람이 설계하고 소수를 평가한다.** 이건 자원 부족으로 인한 타협이 아니라
문제 구조상 맞는 선택이며, 오히려 실효 자유도를 낮춰(후보 6개 중 1개 선택) 과적합
여지를 줄인다. "왜 자동 최적화를 안 썼는가"에 대한 답이 여기 있다.

---

## 4. 선행연구에서 우리 위치

| 연구 | 한 일 | 한계 |
|---|---|---|
| Horton (2023) *LLMs as Simulated Economic Agents: Homo Silicus* (ACM EC 2024) | LLM 에 부존·정보·선호를 주고 경제 실험을 in silico 재현 | 실험실 실험 재현이지 **실제 정책 결과 대조가 아님** |
| **EconAgent** (ACL 2024) | LLM 에이전트로 거시경제 활동 시뮬 | 저자들이 명시한 한계 — **"미묘한 정책 변화에 대한 현실적 행동 반응이 부족"** |
| 생성 에이전트 검증 연구들 | 실험실 실험과 상관 r = 0.85~0.90 | 출력이 편향·변동성 부족·**프롬프트 민감**·하위집단 추론 불가 |

두 가지가 우리와 직결된다.

**EconAgent 가 스스로 인정한 한계가 정확히 우리가 겨냥하는 지점이다.** 우리는 실제 시행된
정책 6개의 실측 결과로 방향을 맞추고, 정답을 열지 않은 7번째 정책으로 검증한다.

**"프롬프트 민감"은 문제가 아니라 우리가 측정하려는 대상이다.** 프롬프트가 민감하다는
것은 한 정책에 맞춘 프롬프트가 다른 정책으로 옮겨가지 않는다는 뜻이고, 홀드아웃 시험이
재는 것이 정확히 그 전이 가능성이다.

---

## 5. 실행 절차 (POM 용어로)

1. **패턴 선정** — 정책별로 부호가 명확하고 우리 구조가 산출 가능한 지표만 고른다.
   못 내는 지표는 사전에 제외한다(`docs/POLICY_ANSWERKEY_MATRIX.md` §3).
2. **후보 모형 집합 고정** — 소비 프롬프트 후보 6개를 **먼저 적어 둔다.** 실효 자유도는
   글자 수가 아니라 시도한 후보 수다.
3. **필터링** — 훈련 정책 6개에서 각 후보를 짧은 런으로 평가, 부호 일치율로 거른다.
4. **홀드아웃 검증** — 살아남은 후보 하나로 8대 소비쿠폰을 **딱 한 번** 돌린다.
   돌려보고 고치면 홀드아웃이 아니다.
5. **보고** — 실패해도 그대로 보고한다. 정직한 실패가 한 정책에 맞춘 성공보다 낫다.

---

## 참고문헌

- Grimm, V. et al. (2005). Pattern-Oriented Modeling of Agent-Based Complex Systems:
  Lessons from Ecology. *Science*, 310(5750), 987-991.
- Topping, C. J. et al. (2012). Post-Hoc Pattern-Oriented Testing and Tuning of an
  Existing Large Model. *PLOS One*, 7(9), e45872.
- Piou, C. et al. (2009). Pattern-oriented modelling: a 'multi-scope' for predictive
  systems ecology. *Phil. Trans. R. Soc. B*.
- Horton, J. J. (2023). Large Language Models as Simulated Economic Agents: What Can
  We Learn from Homo Silicus? *ACM EC 2024*. arXiv:2301.07543
- Li, N. et al. (2024). EconAgent: Large Language Model-Empowered Agents for Simulating
  Macroeconomic Activities. *ACL 2024*. arXiv:2310.10436
- Yang, C. et al. (2023). Large Language Models as Optimizers (OPRO). arXiv:2309.03409
- Yuksekgonul, M. et al. (2024). TextGrad: Automatic "Differentiation" via Text.
  arXiv:2406.07496
- Ramnath, K. et al. (2025). A Systematic Survey of Automatic Prompt Optimization
  Techniques. *EMNLP 2025*. arXiv:2502.16923

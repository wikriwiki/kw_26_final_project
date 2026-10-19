# 문법 하나가 서버를 죽이고 30분치를 틀린 값으로 채웠다

**2026-09-21** · v20 라운드 · 두 번 반복 (13:21, 14:06)

---

## 무슨 일이 있었나

v20(120명)을 시작한 지 2분 만에 SGLang 이 죽었다. 다시 띄우고 다시 시작하자
3분 만에 똑같이 죽었다.

```
terminate called after throwing an instance of 'xgrammar::LogFatalError'
  what():  /project/cpp/earley_parser.cc:203: The element type is not supported! The type is: 5
Fatal Python error: Aborted
```

제약 디코딩 문법을 컴파일하다 C++ 단에서 abort 가 난다. 파이썬 예외가 아니라
프로세스 종료이므로 서버 전체가 내려간다.

**더 나쁜 것은 그 다음이다.** 계획기는 서버가 죽은 줄 모른다. 죽은 서버에 계속
요청을 보내고, 돌아오지 않는 요청을 **`eligible=False` 로 기록한다.**

```
completed 25/960 v25_c120 local_voucher eligible=False errors=['request_or_contract'] factual=None error=timed out
completed 26/960 v25_c120 distancing   eligible=False errors=['request_or_contract'] factual=None error=timed out
...
```

**빈 값이 아니라 틀린 값이 데이터에 들어간다.** 그대로 두면 "이 프롬프트는 자원
게이트를 자주 못 넘는다"로 읽힌다. 실제로는 서버가 없었을 뿐이다.

32칸이 기록됐고 25번 이후는 전부 이것이다. 라운드에서 제외하고
`/data/validation_v20_cohort120.stalled_20260921_1336/WHY_DISCARDED.txt` 에
이유를 적어 함께 보관했다.

## 무엇이 깨지는가 — 찾았다

문법은 칸마다 `attempts/{key}_grammar.json` 에 **호출 전에** 저장되므로 범인을
집어낼 수 있었다. fork 로 격리해 서로 다른 문법을 전부 컴파일해 봤다.

```
v20 문법 452개 중 실패 2개      둘 다 691,168자 · productions 1,103
728개 시도 중 3개(0.4%)가 이 문법을 쓴다
```

**크기 문제가 아니다.** v18 의 1,383,503자짜리 문법은 지금도 통과한다.

```
351,113자   ok        445,877자   ok
655,709자   ok      1,383,503자   ok
691,168자   실패
```

**구조 문제도 아니다.** 실패한 문법과 통과한 1.38MB 문법을 견주면 실패한 쪽이
오히려 더 작고 단순하다.

```
              규칙 수   본문 중앙   대안 최다   괄호 깊이   빈 대안   미정의 참조
실패 691KB     1,104      540자       77개        1          0         0
성공 1.38MB    1,191    1,029자      128개        1          0         0
```

**서명으로도 못 거른다.** productions=1103 인 문법이 7개인데 그중 2개만 깨진다.

> xgrammar 의 버그를 고치는 것은 상류의 일이다. **보내지 않는 것이 우리 일이다.**

## 무엇을 붙였나

### 1. `scripts/sim/preflight_grammars.py` — 보내기 전에 컴파일해 본다

각 칸의 EBNF 를 **fork 한 자식**에서 컴파일한다. C++ abort 는 in-process 로 잡을 수
없으므로 fork 여야 한다. 컴파일되지 않는 칸은 목록으로 남고,
`validate_action_planner.py --exclude` 가 그 칸을 빼고 돌린다.

자원 게이트가 불가능한 계획을 증거와 함께 빼 두는 것과 같은 모양이다.

**v20 에 걸어 본 결과:**

```
칸 960 · 서로 다른 문법 480 · 컴파일 불가 5 → 제외 10칸 (1.0%)
```

**제외된 10칸은 전부 on/off 짝이다.**

```
AGT_11590650_M_50대_002   cashback off·on · local_voucher off·on
AGT_11740640_F_50대_001   cashback off·on · grant off·on · distancing off·on
```

사람 둘 × 기전 다섯이고 양쪽 팔이 함께 빠지므로 **대조는 깨지지 않는다.**
잃는 것은 480쌍 중 5쌍이다.

### 2. `/data/watchdog.sh` — 멎으면 되살리지 말고 멈춰라

3분마다 짧은 생성을 던져 본다. 연속 3회(약 9분) 무응답이면 돌고 있는 검증을
**멈추고** `/data/SGLANG_STALLED` 에 시각과 마지막 서버 로그를 남긴다.

**자동으로 되살려 이어붙이지 않는다.** 이어붙이면 이미 오염된 칸이 그대로 남는다.

### 3. 곁가지 — v21 은 시작도 못 할 뻔했다

`validate_action_planner.py` 의 프롬프트 허용 목록이 v35 에서 끊겨 있었다.
v36·v37 을 더하지 않았으면 `Unregistered prompt module` 로 즉시 죽었다.

---

## 보고서에 반드시 적을 것

1. **v20 의 행렬은 950/960 이다.** 문법 사전검사로 10칸을 뺐고, 그 10칸은 on/off 짝이다
2. **13:19~14:06 의 두 시도는 데이터가 아니다.** 서버가 죽은 뒤의 timeout 이
   `eligible=False` 로 기록된 것이고 전량 폐기했다
3. **이 사고 이전 라운드(v18·v19)는 영향이 없다.** v19 계획은 13:18:36 에 끝났고
   서버는 13:21:34 에 멎었다

---

# 정정 (16:50) — 진단이 틀렸다. 문법 하나의 문제가 아니다

위에서 "컴파일되지 않는 문법 하나가 범인"이라고 적었다. **그 진단으로 막지 못했다.**
사전검사를 붙이고도 서버가 두 번 더 죽었다(15:33, 16:24).

## 사전검사가 서버와 다른 것을 컴파일하고 있었다

먼저 하나 고쳤다. 서버는 이렇게 컴파일한다.

```
우리:  TokenizerInfo.from_huggingface(tokenizer)
       GrammarCompiler(ti, max_threads=1)
       compile_grammar(Grammar.from_ebnf(eb))

서버:  TokenizerInfo.from_huggingface(tokenizer, vocab_size=153600, stop_token_ids=[53])
       GrammarCompiler(tokenizer_info=ti)        ← 기본 스레드 수
       compile_grammar(문자열 그대로)
```

맞추자 불량 문법이 2개에서 6개로 늘었다. **그래도 서버는 죽었다.**

## 재현 — 컴파일러 하나를 재사용하면 죽는다

격리하면 **전부 통과하는** 문법 233개만 골라, 서버처럼 **컴파일러 하나를 재사용**하며
차례로 컴파일했다.

```
공유 컴파일러 · 8스레드 동시   100개쯤에서 abort
공유 컴파일러 · 순차           100개쯤에서 abort
문법마다 새 컴파일러(fork)      233개 중 1개만 실패
```

**같은 문법이 혼자 컴파일하면 통과하고, 줄 세워 컴파일하면 죽는다.**
그러므로 "나쁜 문법을 골라내는" 방어는 원리상 통하지 않는다.

비결정적이기도 하다. 사전검사가 통과시킨 문법을 뒤이은 격리 검사가 떨어뜨렸다
(`86cb688d…`, 691,168자). 같은 설정인데 결과가 다르다.

## 그래서 무엇이 남았나

**확실한 것 하나는 이미 붙였다** — 계획기가 서버 사라짐을 감지하고 멈춘다.
16:24 에 실제로 작동했다.

```
{"aborted_at": "...T07:24:22Z", "consecutive_connection_failures": 12,
 "rows_written": 266, "jobs": 948,
 "reason": "연속 12회 연결 실패. 서버가 사라진 것이지 이 칸들의 결과가 아니다."}
```

**266칸에서 멈췄고 오염된 값은 하나도 없다.** 이전 두 번은 694칸·8칸이
`eligible=False` 로 기록됐었다. 그 차이가 이 방어의 값어치다.

남은 것은 **런을 끝까지 가게 하는 것**이고, 후보는 둘이다.

1. xgrammar 컴파일을 단일 스레드로 강제한다 (`max_threads=1`) — 생성 의미는
   안 바뀌므로 v18 과의 비교가 유지된다. **시험 중**
2. 안 되면 **이어달리기**: 크래시 → 서버 재시작 → 멈춘 지점부터 재개.
   지금 계획기는 `folder.mkdir(exist_ok=False)` 라 재개가 안 된다

## 보고서에 적을 것 (갱신)

- v20 은 **아직 한 번도 완주하지 못했다.** 13:19·14:04·15:13·16:04 네 번 모두
  서버 크래시로 중단됐다
- 마지막 시도만 **오염 없이** 멈췄다. 앞의 것들은 폐기했다
- **v18·v19 는 영향이 없다.** 같은 서버 인스턴스에서 480칸을 두 번 완주했고,
  그때는 서로 다른 문법이 240개였다. 120명 코호트는 480개다 — 이 배수가
  같은 버그를 깨운 것으로 보인다

---

# 해결 (17:01) — 스레드 하나로 묶으면 사라진다

`compile_grammar` 는 안에서 스스로 병렬화한다. 그래서 "순차"로 돌려도 여덟 스레드였다.
**스레드를 하나로 묶자 그대로 통과한다.**

```
공유 컴파일러 · 기본 스레드      233개 중 ~100개에서 abort
공유 컴파일러 · max_threads=1    233개 전부 통과
```

그리고 더 중요한 것이 있다. **진짜로 망가진 문법의 운명도 바뀐다.**

```
기본 스레드     C++ abort         → 서버 전체가 죽는다 (파이썬이 잡을 수 없다)
max_threads=1   RuntimeError      → dispatch_ebnf 가 잡아 InvalidGrammarObject 로 처리
```

SGLang 은 이미 `except RuntimeError` 를 갖고 있었다. 다만 abort 는 예외가 아니라
프로세스 종료라 그 손이 닿지 않았을 뿐이다. **한 칸이 나쁜 문법을 가져도 더는
라운드 전체를 죽이지 못한다.**

## 무엇을 바꿨나

`scripts/sim/patch_sglang_grammar_threads.py`

```
- self.grammar_compiler = GrammarCompiler(tokenizer_info=tokenizer_info)
+ self.grammar_compiler = GrammarCompiler(tokenizer_info=tokenizer_info,
+                                         max_threads=1)
```

**컴파일 병렬도만 바뀐다.** 샘플링도, 문법 자체도, 문법이 허용하는 토큰도 그대로다.
그러므로 v18 과의 비교가 유지된다. 원본은 `.py.orig` 로 보관했고, 스크립트를 다시
돌리면 이미 적용됐는지 알아본다.

사전검사도 같은 조건(`max_threads=1`)으로 맞췄다. 그래야 **결정적**이고, 서버가
보게 될 것과 같은 것을 본다. 스레드를 켠 채로 돌렸을 때는 같은 문법을 한 번은
통과시키고 한 번은 떨어뜨렸다.

## 방어가 세 겹이 됐다

| 겹 | 무엇을 막나 | 확인 |
|---|---|---|
| `max_threads=1` | 크래시 자체 | 233개 전부 통과 |
| 사전검사 | 진짜 나쁜 문법을 가진 칸을 미리 뺀다 | on/off 짝으로만 빠짐 |
| 회로 차단기 | 그래도 서버가 사라지면 멈춘다 | 16:24 에 266/948 에서 작동, 오염 0 |

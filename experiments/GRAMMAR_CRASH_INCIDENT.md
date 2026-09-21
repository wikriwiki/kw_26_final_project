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

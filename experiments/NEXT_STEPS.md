# 다음에 무엇을 하는가 — 명령까지 적는다

**2026-09-24 03:00 갱신** · 이 문서는 **진행 중인 계산이 끝났을 때** 할 일이다

---

## 지금 돌고 있는 것

```
본런   /data/run_p013_ruler.sh   arm A(v5·n=700·12일) → arm B(같은 것 한 번 더)
       arm A 09:30 무렵 · 두 팔 19시 무렵
복제   /data/run_case_trend_replicate.sh   후보 2 · 96 호출 · 03:40 무렵
```

## 끝난 것

```
후보 1 (범위의 산술)  탐침 48호출 → **기각**  양측 p=0.549 · 방향도 흐림
후보 2 파일럿         탐침 48호출 → 관문 불통과
                     소비성향 감소 11 : 증가 4 · 양측 p=0.118 · 단측 0.059
```

## ① 후보 2 **복제**가 끝나면

```bash
scp -i ~/.ssh/outofmemory.pem -P 10022 \
    outofmemory@123.37.28.167:/data/ct_replicate/responses.jsonl output/ct_replicate.jsonl
py scripts/sim/scope_fact_probe.py --frozen /dev/null --out output \
    --responses output/ct_replicate.jsonl
```

**등록한 판정은 단측이다** — 방향(소비성향 감소)을 파일럿에서 고정했다.
화면에 양측·단측이 나란히 찍히므로 **단측 칸을 본다.**

**파일럿의 15쌍을 합치지 않는다.** 합치면 유의해질 때까지 표본을 늘린 것이 된다.

```
소비성향 단측 p < 0.05   → 라운드 자격 있음. ②로
그 밖                   → 후보 2 도 기각. ③으로
```

기각이면 `experiments/case_trend/` 에 결과 문서를 쓰고 `SELECTED_PROMPT.md` 의
"v5 를 이기려고 한 것들" 표에 한 줄을 더한다.

## ② 후보 2 가 갈렸다면 — 라운드

**본런이 GPU 를 비운 뒤에** 건다. 세 팔이고 약 19시간이다.

```bash
ssh ... 'setsid nohup bash /data/run_case_trend_round.sh > /data/ct_round_nohup.log 2>&1 &'
```

런너가 탐침 판정을 **다시 확인**하고 안 갈렸으면 스스로 멈춘다.
다만 런너의 관문은 `/data/ct_probe`(파일럿)를 읽는다. **복제로 판정했다면
그 경로를 `/data/ct_replicate` 로 바꿔서 건다** — 안 그러면 파일럿(불통과)을
보고 스스로 멈춘다.
판정 출력이 **반증(시점 위약)부터** 찍는다 — 거기서 깨지면 거리두기에서
좋아졌더라도 채택하지 않는다.

## ③ 후보 2 도 기각이라면 — arm B 를 살린다

후보 라운드가 없으면 GPU 의 최선 용도는 **P013 런 간 이동**(arm B)이다.
`run_p013_ruler.sh` 가 arm A 뒤에 자동으로 arm B 를 돌리므로 **그냥 두면 된다.**

(반대로 ②로 간다면 arm A 채점 파일이 생긴 직후 아래로 arm B 를 건너뛴다.
arm A 자료는 이미 저장돼 있어 잃는 것이 없다.)

```bash
ssh ... 'pkill -f "[r]un_p013_ruler.sh"; pkill -f "run_simulation.py --start 2020-05-04"'
```

## ④ arm A 가 끝나면 — 결과를 표에 넣는다

```bash
scp ... :/data/p013_ruler/score_ruler_a.json output/rounds/
py scripts/report/score_to_block.py output/rounds/score_ruler_a.json \
   --key EMERGENCY_2020 --name result_ruler_a \
   --design "p013_ruler arm A · v5 · n=700 · 고친 적격 판정으로 EM-2 를 처음 제대로 잰 런" \
   --window "2020-05-07:2020-05-08 → 2020-05-14:2020-05-15"
```

찍힌 블록을 `data/experiments/scoring_table.json` 의 `EMERGENCY_2020` 안에
붙인다. **그 다음 판단이 하나 있다** — `sign_scoreboard.READINGS` 의
긴급재난 항목을 이 런으로 바꿀지. 규칙은 **가장 큰 표본**이므로 n=700 이
n=200 을 대체한다. 바꾸면 `SUSPECT` 의 EM-2 줄도 지운다(고친 자로 다시 쟀으므로).

그리고 등록한 예측(`p013_ruler/prediction.md`)에 대고 결과를 적는다.
**빗나갔으면 빗나갔다고 적는다** — 표본을 올렸는데도 구간이 0 을 지나면
"병목은 표본" 주장이 약해진다.

```bash
bash tools/refresh_reports.sh        # 보고 여섯을 같은 자료로 다시 만든다
```

마지막으로 `output/report/convergence.html` 을 아티팩트로 다시 올린다.

## ⑤ GPU 가 비면 — 아직 안 잰 정책 셋

부호 적중표에 "아직 안 쟀다" 로 남은 지표가 아홉이다. **잴 수 있는데 안 잰
것**이므로 GPU 가 비는 대로 순서대로 돌린다.

```
P016 농할      런너 준비 완료 · 13일 × 500명 ≈ 10시간
               bash /data/run_p016.sh
               **런 전 점검이 들어 있다** — 사용처 표시·판정 룰·표시문구가
               어긋나면 스스로 멈춘다

사적모임        창은 등록됐으나 **기준선 런이 함께 필요**하다. 12-29:30 이 연말
               주간이라 교란이 크고, 단일 런으로 채점하지 않기로 등록했다

P010          창은 등록됐고 정책파일도 붙였다. 2025년이라 거리두기 레짐이 없다
```

P016 을 먼저 하는 이유: 런너·사전등록·창 수정이 모두 끝나 있고, 기전
일반화(코드 수정 없이 새 정책이 붙는가)를 시험하는 라운드라 다른 것과 겹치지
않는다.

## 지키는 것

- 채점표에 붙이는 것도, READINGS 를 바꾸는 것도 **사람이 한다.** 어느 런을
  읽는지가 코드에 숨으면 안 된다
- 결과를 보고 사전등록한 지표·창·판정식을 고치지 않는다
- 후보가 v5 를 대체하려면 **두 정책 이상**에서 이기고, 이동이 런 간 이동보다
  크며, 방향 9/9 를 하나도 잃지 않아야 한다

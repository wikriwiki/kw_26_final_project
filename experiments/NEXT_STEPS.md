# 다음에 무엇을 하는가 — 명령까지 적는다

**2026-09-24 22:20 갱신**

---

## 먼저 읽을 것 — 오늘 네 번, 저장소 안에 이미 답이 있었다

```
plan_channel/v44_changes_the_plan.md   v44 가 "창으로는 크기를 못 키운다"를
                                       이미 보였는데 안 읽고 창 실험을 다시 걸었다
error_budget/wallet_audit.md           28일 P012 런이 **이미 있었고**, 소득이 꺼져
                                       67%가 파산해 −57.70%를 정책 효과로 읽을 뻔했다
error_budget/p012_2_why_unproducible.md 분류기가 맞는데 내 질의가 틀렸다
                                       (코드 주석에 이유가 적혀 있었다)
plan_channel/recompute_tool_validated.md 옛 문서가 **두 자를 섞어** 적어 둔 것을
                                       재현하다 찾았다
```

**규칙을 넓힌다**: "채점표를 먼저 읽어라" → **"돌리기 전에 그 주제로 앞서 돌린
것을 먼저 읽어라."** 앞선 라운드 결과 · 정책 파일 notes · 코드 주석 전부.

## 지금 돌고 있는 것 하나

```
/data/p012_28d/28d_v5   v5 · N=500 · 2021-10-01~10-28 · 회계 고침 **꺼짐**
                        EXP_DAILY_INCOME=anchor · EXP_BALANCE_DAYS=39  <- 파산 방지 확인됨
                        하루 약 46분 · 3/28일 (22:20 기준) · **9/25 16시 무렵 종료**
```

정책은 **10-15** 부터다(effective_from). 그래서 날짜가 이렇게 갈린다.

```
기준선  10-04~10-08 (월~금) · 10-11,10-12 (월·화)   정책 전이고 OFF 창 밖 — **평일만**
OFF     10-13, 10-14 (수·목)                        정책 전
ON      10-27, 10-28 (수·목)                        정책 후 · 요일종류 맞춤
```

**요일을 섞지 마라.** 앵커와 계획이 요일에 따라 **반대로** 움직인다(금 계획/앵커
0.500 · 토 1.568, 같은 439/500명이 양쪽으로 갈린다). 섞으면 클램프가 신호를 먹는다.

---

## ① 런이 끝나면 — **반드시 이 순서로**

### (1) 지갑 관문부터. 이걸 안 보고 지표를 읽으면 파산을 정책으로 읽는다

```bash
ssh -i outofmemory.pem -p 10022 outofmemory@123.37.28.167 \
  'python3 -c "
import json,statistics as st
rows=[json.loads(l) for l in open(\"/data/p012_28d/28d_v5/metrics/day_2021-10-28.jsonl\") if l.strip()]
ok=[r for r in rows if r.get(\"status\")==\"ok\"]
bal=[r.get(\"balance\") or 0 for r in ok]
print(\"잔고0 %.1f%%\" % (100*sum(1 for b in bal if b<=0)/len(ok)))
"'
```

**10%를 넘으면 지표를 읽지 않는다.** 넘으면 `EXP_BALANCE_DAYS` 를 올려 다시 건다.

### (2) 문턱 모양. 크기보다 **먼저** 본다

```bash
ssh ... 'cd /data/repo && export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j \
  NEO4J_PASSWORD=exp001pass && /data/venv/bin/python \
  scripts/report/threshold_response_shape.py /data/p012_28d/28d_v5/metrics \
  --off 2021-10-13:2021-10-14 --on 2021-10-27:2021-10-28'
```

**날짜는 `시작:끝` 콜론 표기다**(쉼표 아님). metrics_dir 은 위치 인자이고
`--metrics` 가 아니다. 그래프 자격증명이 필요하다 — 2026-09-24 예행에서 셋 다 틀려
있었다. **언저리 구간(85~100%)이 가장 커야** 문턱 제도의 모양이다.

이틀치 투영은 **도달 48% · 누적/문턱 중앙 0.975** 였다. 실제가 그 근처면 창이
살아 있는 것이고 P012-3·4·6 이 처음으로 값을 갖는다.

### (3) 회계 고침을 원장 위에서 다시 셈한다 — GPU 0시간

```bash
ssh ... '/data/venv/bin/python /data/repo/scripts/report/recompute_plan_channel.py \
  --metrics /data/p012_28d/28d_v5 \
  --baseline 2021-10-04,2021-10-05,2021-10-06,2021-10-07,2021-10-08,2021-10-11,2021-10-12 \
  --off 2021-10-13,2021-10-14 --on 2021-10-27,2021-10-28 \
  --trend-days 2021-10-04,2021-10-05,2021-10-06,2021-10-07,2021-10-08,2021-10-11,2021-10-12,2021-10-13,2021-10-14 \
  --json-out /data/p012_28d/recompute.json'
```

`--trend-days` 는 **예열(10-01~03) 뒤이고 정책(10-15~) 전**인 날만 준다. OFF 와
ON 사이가 14일이라 그 사이의 표류를 덜어야 하는데, **예열을 표류로 읽으면 효과가
지워진다** — P013 에서 그렇게 +8.99% 가 −0.36% 가 됐다. 도구가 모양을 찍고
세 가지로 갈라 말한다(표류 없음 / 예열 섞임 / 흩어짐). 셋 다 **안 덜고 넘어간다** —
덜었다면 그 줄이 찍힌다.

자 셋(incl_online · personal · instore)을 함께 찍는다. **`incl_online` 이 실측
"카드 소비" 에 맞는 자다.** 도구가 스스로 거부·경고하는 것들:

```
기준선이 창과 겹치면    거부 (순환)
클램프 40% 초과         요일종류 의심 → --baseline-mode daytype 로 다시
기준선 결손 10% 초과    날짜를 늘려라
```

검증됨: P013 에서 옛 읽기 −0.04% / +0.32% / +9.03% 를 재현한다
(`plan_channel/recompute_tool_validated.md`).

### (4) 채점표에 옮기고 오차 예산을 다시 낸다

```bash
py -3 scripts/report/error_budget.py
bash tools/refresh_reports.sh
```

**두 숫자를 병기한다** — 어느 자로 읽었는지 안 적은 수는 못 쓴다.

---

## ② 지금 열려 있는 것 / 닫힌 것

```
닫힘  프롬프트로 P012 크기 맞추기   탐침 5회 1,344호출 531쌍 전부 기각 + v44
      제외업종 몫을 프롬프트로       calib_02 — 27%가 논리적으로 불가능한 답
열림  회계 고침(EXP_PLAN_DRIVES_TOTAL)  P013 표본 밖 확인됨(+9.03% vs 실측 +7.30%)
      28일 창                        문턱을 **존재하게** 한다(크기 장치가 아니다 — v44)
막힘  P012-2                         판을 바꿔야 한다. 28일 런이 그래프를 쓰는 동안
                                     손대지 않는다. p012_2_why_unproducible.md
```

## ③ 아직 못 잰 것 — 빼지 말고 "못 잼" 으로 센다

```
P012-5 · P012-6 · DS-6(hub_type 자료가 그래프에 없다)
EM-4   다음 P013 런에 **업종군을 per-agent 출력에** 넣어야 COVID 교란과 갈린다
       (arm A 그래프가 덮여 지금은 못 가른다)
```

## ④ 함정

```
· 라운드 런너의 관문이 파일럿 경로를 읽는다. 복제로 판정했다면 경로를 바꿔 걸 것
· scp 뒤에는 반드시 sed -i "s/\r$//" — CRLF 가 export 를 조용히 죽인다
· 프로세스 확인은 자기 자신을 빼고: ps -eo pid,cmd --no-headers | grep run_simulation.py | grep -v grep
· /data/repo 의 미커밋 사본이 실제 런 코드다. 배포 전 대조할 것
· /data/out_P012_200x28 은 읽지 마라 (DO_NOT_READ.md) — 지갑이 말랐다
```

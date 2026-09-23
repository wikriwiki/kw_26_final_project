# 다음에 무엇을 하는가 — 명령까지 적는다

**2026-09-24 06:00 갱신** · 이 문서는 **진행 중인 계산이 끝났을 때** 할 일이다

---

## 판이 바뀌었다 — 먼저 이것부터 읽어라

```
experiments/error_budget/diagnosis_03.md   우리 에이전트에게는 제외업종이 없다
experiments/error_budget/diagnosis_04.md   한 상수가 두 일을 한다 — 가르는 수리안
experiments/split_anchor/prereg.md         사전등록 (결과 보기 전 커밋됨)
```

**오차의 77%가 프롬프트 밖에 있다.** 정답지가 재라는 제외업종(백화점·대형마트·
온라인)을 소비 모델이 측정 전에 통째로 지우고, 적립업종은 총액이 상수라 못 큰다.
그래서 프롬프트 후보를 더 만드는 대신 **회계 2층을 가른다.**

## 지금 돌고 있는 것

```
본런   p013_ruler arm A (v5·n=700·12일)   11:30 무렵 · 이어서 arm B
탐침   /data/online_calib                 배송 몫 교정 96호출 (사전등록 단계 가)
```

## ① 교정 탐침이 끝나면 — 중단 조건이 먼저다

```bash
ssh -i outofmemory.pem -p 10022 outofmemory@123.37.28.167 \
  'tail -20 /data/online_calib/probe.log; cat /data/online_calib/calibration.json'
```

```
응답률 < 90%      → **중단.** 안 오는 필드로는 회계를 못 돌린다.
                    v5online 의 블록 위치를 고쳐 다시 탐침 (본런 금지)
분산 ~ 0          → 레버가 죽었다. 수리를 접고 "이 모델로는 상생 기전을
                    표현할 수 없다"를 결론으로 보고한다 — 그것도 답이다
통과              → keep_mean 을 적어 두고 ② 로
```

## ② 항등 관문 — arm A/B 가 끝나 GPU 가 비면

`consumption.py` 를 올린 **뒤에** 돌린다(지금은 본런 중이라 안 올렸다).

```bash
scp -i outofmemory.pem -P 10022 scripts/sim/consumption.py \
    outofmemory@123.37.28.167:/data/validation_v3/repo/scripts/sim/
# v5(= online_share 없음) + SPLIT=1 로 하루만. 현행과 ±1% 안이어야 한다.
EXP_SPLIT_ANCHOR=1 SIM_PROMPT_VARIANT=v5 python scripts/sim/run_simulation.py \
    --start 2021-10-01 --days 1 --limit 200 --workers 48
```

벗어나면 **본런을 돌리지 않는다.** 수준을 건드린 것이므로 정책 효과와 섞인다.

## ③ 본런 — P012 · v5online + 가른 회계

```bash
EXP_SPLIT_ANCHOR=1 EXP_KEEP_MEAN=<탐침에서 잰 값> \
SIM_PROMPT_VARIANT=v5online python scripts/sim/run_simulation.py ...
```

끝나면 채점 → `score_to_block.py` 로 옮겨 적기 → `bash tools/refresh_reports.sh`.

**판정은 총 오차로 한다.** P012-1 만 좋아지고 P012-2 가 그만큼 나빠지면 기각이다
(사전등록에 적어 두었다).

## 끝난 것

```
후보 1 (범위의 산술)   탐침 48호출 → 기각  양측 p=0.549
후보 2 (기준 배수)     파일럿 단측 0.059 → **복제 단측 0.094 → 기각**
                       등록 기준은 "새 48쌍만으로 0.05". 합치지 않는다.
                       라운드 자격 없음 — 19시간 세 팔 라운드를 돌리지 않는다
눈금 이동              P012-2 를 st.online_spent 로. 옛 읽기는 SUSPECT
                       **총 오차 46.27 → 29.52%p 는 진전이 아니다**(ruler_move_01.md)
```

## 함정 — 두 번 당했다

```
· 라운드 러너의 관문이 파일럿 경로를 읽는다. 복제로 판정했다면 경로를 바꿔 건다
· 서버 /data/repo 와 /data/validation_v3/repo 는 다른 세대다. run_simulation.py 를
  레포에서 덮어쓰지 않는다 (tools 디렉터리는 validation 쪽에 없어서 새로 만들었다)
· 탐침을 둘 동시에 돌리면 둘 다 느려진다. 본런과는 워커 4로 나눠 쓴다
```

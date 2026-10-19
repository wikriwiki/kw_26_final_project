# 실험 v1 — 정책별 결과 비교

> 막대는 **0이 가운데**다. 왼쪽이 감소, 오른쪽이 증가.
> 값은 기록된 근거 파일에서 계산했다. 손으로 적은 숫자는 없다.

## 직전 실험 daily_v2 seed56001

- 칸 96 · 결제 행 96 · 전체 행렬 완전: **예**
- 범위: One-day matched conditional catalog-consumption probe. Same people and calendar on/off. No population significance, observed historical magnitude or total household consumption claim.

**정책 있음 − 정책 없음 (총소비, 1인 1일)**

```
          -1,917원                 0 +1,917원                
캐시백                            │██                        +167원
거리두기  ████████████████████████│                          -1,917원
지원금                            │██████████                +833원
지역화폐                          │████████████████          +1,250원
```

시뮬레이터 안에서 같은 사람·같은 날짜의 on/off 차이다. 실제 정책 효과나 모집단 유의성이 아니다.

**수준값 (원 / 1인 1일)**

| 기전 | 정책 없음 | 정책 있음 | 차이 |
|---|---:|---:|---:|
| 캐시백 | 5,250 | 5,417 | +167 |
| 거리두기 | 9,000 | 7,083 | -1,917 |
| 지원금 | 4,333 | 5,167 | +833 |
| 지역화폐 | 3,417 | 4,667 | +1,250 |

## 직전 실험 daily_v2 seed57001

- 칸 96 · 결제 행 96 · 전체 행렬 완전: **아니오**
- ⚠ **행렬이 불완전하다.** 아래 수치를 온전한 비교로 읽으면 안 된다.
- 범위: One-day matched conditional catalog-consumption probe. Same people and calendar on/off. No population significance, observed historical magnitude or total household consumption claim.

**정책 있음 − 정책 없음 (총소비, 1인 1일)**

```
          -3,417원                 0 +3,417원                
캐시백                            │████████                  +1,167원
지원금                            │███████████████           +2,083원
지역화폐                          │████████████████████████  +3,417원
```

시뮬레이터 안에서 같은 사람·같은 날짜의 on/off 차이다. 실제 정책 효과나 모집단 유의성이 아니다.

**거리두기 — 채점 거부**

```
  Failed response cannot be omitted or zero-filled
```

거부는 결과다. 0으로 그리지 않는다 — "효과 없음"이 아니라 "비교 불가"다.

**수준값 (원 / 1인 1일)**

| 기전 | 정책 없음 | 정책 있음 | 차이 |
|---|---:|---:|---:|
| 캐시백 | 5,583 | 6,750 | +1,167 |
| 지원금 | 4,917 | 7,000 | +2,083 |
| 지역화폐 | 2,000 | 5,417 | +3,417 |

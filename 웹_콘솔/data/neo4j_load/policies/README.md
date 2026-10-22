# policies/ — 자연어 정책 파일 (선택)

POC 단계에선 1~3개 정도면 충분.

## 파일 형식 예시 — `P001.txt`

```
강남구 소비쿠폰 10만원

대상지역: 강남구 전체 (15개 행정동)
혜택업종: 음식점, 카페
환급률: 30%
1인 한도: 100,000원
발표일: 2026-05-07
시행일: 2026-05-08
종료일: 2026-05-31
홍보채널: 뉴스, SNS
```

## 처리 흐름

```
policies/*.txt
  ↓ Watchdog 감지
LangChain LLM 추출 (vLLM Qwen3-32B)
  ↓
Pydantic 검증
  ↓
Cypher MERGE → :Policy + :applied_to + :targets
  ↓
Signal Sender → Redis L3 dirty/DEL → Celery 재요약
```

## POC 단축 경로
LangChain 파이프라인 가동 전엔 **수동 JSON으로 대체** 가능:

```json
{
  "id": "P001",
  "name": "강남구 소비쿠폰 10만원",
  "benefit_categories": ["식사", "카페"],
  "benefit_rate": 0.30,
  "cap_per_agent": 100000,
  "announce_date": "2026-05-07",
  "effective_from": "2026-05-08",
  "effective_until": "2026-05-31",
  "target_districts": ["강남구"]
}
```

수동 JSON은 `policies/P001.json`으로 저장하면 `08_policies.py`가 LangChain 우회로 처리.

## 부재 시
- 정책 노드 미생성 → Dawn ⑤ 활성 정책 쿼리가 빈 결과 → 정책 시뮬레이션 효과 측정 불가
- 일반 시뮬 (정책 없이 60일 베이스라인)은 정상 동작

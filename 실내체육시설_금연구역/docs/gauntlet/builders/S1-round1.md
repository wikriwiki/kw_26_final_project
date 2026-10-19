# S1 Builder 전달물 — round 1

역할: S1 전담 빌더
기준서: `docs/GAUNTLET_WEB_CONSOLE.md`
계약: `docs/gauntlet/contracts/S1.md` (`s1.0.0`)

## 변경 파일

- `web/fixtures/_build_fixtures.py` — 실제 산출물에서 JSONL·체크포인트·요약·로그를 스트리밍 집계하고 계약 응답을 생성한다. 이번 라운드에서 기존 재작업을 실제 데이터로 재생성했다.
- `web/fixtures/*.json` — BASE/FINAL/BASE7500 3종 run과 P008~P011 정책의 실제 응답 픽스처 36개.
- `tests/unit/test_s1_contract.py` — S1 계약 회귀 테스트 8개.
- `docs/gauntlet/contracts/S1.md` — 잠금된 계약과 재현 명령.

## 검증

```text
python web/fixtures/_build_fixtures.py
python -m unittest discover -s tests -p "test_*.py" -v
```

`_build_fixtures.py`: exit 0. 실제 소스 경로에서 JSON 36개를 생성했다.
정책 preflight 4건은 모두 `exit_code: 0`; `db_wiring_checked`는 실제 환경에
`NEO4J_URI`가 없어 `unknown: ["db_wiring"]`으로 남겼다. 이 경고를 통과로
위장하지 않았다.

계약 테스트: **8 passed**, exit 0.

## 실물 예시

- `run.BASE7500.day.2025-07-14.json`: 원본 `source_bytes` 19,599,953, 응답 7,009 bytes, `agents_ok` 4,533, 분위 미상 5명.
- `run.BASE7500.day.2025-07-14.failed.json`: 체크포인트가 없어 `total: null`, `unknown: ["failed_checkpoint"]`.
- `policy.P010.validate.json`: preflight 원문 `prompt_preview`에 들여쓴 `배경:` 줄 포함.

## 알려진 한계

실행 중인 run의 일자 집계는 완료 전 캐시하지 않아 매 요청 스트리밍 비용이
발생할 수 있다. S2는 첫 화면에서 `status_scan`만 사용하고 전체 집계는
워커 스레드에서 수행해야 한다(B4/B5).


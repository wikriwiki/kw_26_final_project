# 업종형소비쿠폰_P015 — 파일 출처

다시 만들려면 `python scripts/report/collect_policy_raw_data.py`.
**원본**이 `http` 로 시작하면 공공기관에서 직접 내려받은 것이고,
경로면 저장소 안의 파일을 복사한 것이다.

| 파일 | 원본 | 크기 | sha256 |
|---|---|---:|---|
| `시뮬정의_P015.json` | `data/neo4j_load/policies/P015.json` | 3.0KB | `bc490303612afce2fc2f31696a30447c2aedb06fd3413b31653d1c9fc176c472` |
| `정책원문_2020년_하반기_경제정책방향.pdf` | [www.korea.kr](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156393266) | 3.8MB | `1129fb80665394584ba39a4cc31e1f5796d05a5b6950630e2a559367dcae5ad0` |
| `정책원문_하반기경제정책방향_보도자료_20200601.hwp` | [www.korea.kr](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156393266) | 153.5KB | `768b8cc5ddf5d78600ff60b27f010298fe6dfabbc91842b9d91d4789279ee101` |

PDF 는 복사 전에 실제 내용이 있는지 확인했다 — `%PDF` 로 시작하고
`%%EOF` 로 끝나는지. G: 드라이브의 껍데기 파일을 거르기 위해서다.

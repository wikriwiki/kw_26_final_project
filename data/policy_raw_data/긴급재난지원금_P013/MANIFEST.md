# 긴급재난지원금_P013 — 파일 출처

다시 만들려면 `python scripts/report/collect_policy_raw_data.py`.
**원본**이 `http` 로 시작하면 공공기관에서 직접 내려받은 것이고,
경로면 저장소 안의 파일을 복사한 것이다.

| 파일 | 원본 | 크기 | sha256 |
|---|---|---:|---|
| `시뮬정의_P013.json` | `data/neo4j_load/policies/P013.json` | 2.3KB | `3f771660a939a8781a2fe0f2e92adf2cd08fa2a60a80b52a04de6cfd52a06661` |
| `정책원문_긴급재난지원금_신청및지급방안_행안부_20200429.hwp` | [www.mois.go.kr](https://www.mois.go.kr/frt/bbs/type010/commonSelectBoardArticle.do?bbsId=BBSMSTR_000000000008&nttId=76947) | 99.0KB | `2aeb7326c32a010f7acb6469ce8541526447285b9ba04af17508e90ec4d61967` |
| `정답지_KDI_FOCUS_1차긴급재난지원금_효과와시사점_2020.pdf` | [www.kdi.re.kr](https://www.kdi.re.kr/research/focusView?pub_no=16851) | 600.3KB | `e40157a5167bbcd658c9bbed90c7898538a0f087e9e322b78e6a174d55d95399` |

PDF 는 복사 전에 실제 내용이 있는지 확인했다 — `%PDF` 로 시작하고
`%%EOF` 로 끝나는지. G: 드라이브의 껍데기 파일을 거르기 위해서다.

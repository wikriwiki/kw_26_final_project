# 민생회복소비쿠폰_P010 — 파일 출처

다시 만들려면 `python scripts/report/collect_policy_raw_data.py`.
**원본**이 `http` 로 시작하면 공공기관에서 직접 내려받은 것이고,
경로면 저장소 안의 파일을 복사한 것이다.

| 파일 | 원본 | 크기 | sha256 |
|---|---|---:|---|
| `정답지_한국은행_이슈노트_2026-13.pdf` | `한국은행_이슈노트.pdf` | 2.0MB | `38217828d07a75be74d4d39db593de87ebb96cad8d4bc0b35fe44bf67f9d9c8d` |
| `대조표_BOK_이슈노트_전체지표.md` | `docs/BOK_실측지표_대조표.md` | 73.6KB | `4daa524310c51adf77c42f9f88f2f3db70247ce1bc98c13815958dd95153113b` |
| `시뮬정의_P010.json` | `data/neo4j_load/policies/P010.json` | 3.9KB | `82bf0f7645e8455befc9108cda18d1ba270a97006575ff4315e0524c71e93611` |
| `정책원문_민생회복소비쿠폰_지급시작_행안부_20250705.pdf` | [www.mois.go.kr](https://www.mois.go.kr/frt/bbs/type010/commonSelectBoardArticle.do?bbsId=BBSMSTR_000000000008&nttId=118705) | 2.5MB | `b44ee23767390c87f9ea25463f637869f0374667fa53a7e2ef11728d19cc061f` |
| `집행결과_민생회복소비쿠폰_최종집계_20251205.pdf` | [www.korea.kr](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156733265) | 325.7KB | `02004aaca1dd57c278af6a38d4c7db091cf0bf34042505b192b9098acb85f1a2` |

PDF 는 복사 전에 실제 내용이 있는지 확인했다 — `%PDF` 로 시작하고
`%%EOF` 로 끝나는지. G: 드라이브의 껍데기 파일을 거르기 위해서다.

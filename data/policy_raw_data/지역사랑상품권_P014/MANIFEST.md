# 지역사랑상품권_P014 — 파일 출처

다시 만들려면 `python scripts/report/collect_policy_raw_data.py`.
**원본**이 `http` 로 시작하면 공공기관에서 직접 내려받은 것이고,
경로면 저장소 안의 파일을 복사한 것이다.

| 파일 | 원본 | 크기 | sha256 |
|---|---|---:|---|
| `시뮬정의_P014.json` | `data/neo4j_load/policies/P014.json` | 2.6KB | `9dee642929e4f780fc42f1e40bd2a56ce47a54d367352b850c02dfa5d9264af0` |
| `정책원문_지역사랑상품권_발행지원사업_종합지침_20210122.pdf` | [www.mois.go.kr](https://www.mois.go.kr/frt/sub/a06/b07/localVoucher/screen.do) | 1.1MB | `0c92f803d80208e75f45bd946721ff5b039c702828ea94142d6a860c4cb79ccd` |
| `정답지_조세재정연구원_지역화폐가_지역경제에_미친_영향_2020.pdf` | [www.kipf.re.kr](https://www.kipf.re.kr/uloads/kiPublish/202012/FILE_202102020125031083.pdf) | 1.2MB | `6e3a744a33d7bf635b218b9da9430f9a8359e017da60b1e2aaab5758facf0950` |

PDF 는 복사 전에 실제 내용이 있는지 확인했다 — `%PDF` 로 시작하고
`%%EOF` 로 끝나는지. G: 드라이브의 껍데기 파일을 거르기 위해서다.

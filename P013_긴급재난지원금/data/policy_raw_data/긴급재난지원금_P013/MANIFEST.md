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

## 2026-10-03 추가 — 검증지표 원문

행정안전부 보도자료 HWP 3건은 이 폴더에 둔다. 큰 PDF 3건(KDI 연구 Ⅱ 9.6MB · 협동연구 종합편 27.5MB · 지역편 11.8MB)은
저장소에 넣지 않고 `E:\p013_raw_sources_20261003\` 에 둔다(같은 폴더의 `SHA256SUMS`).

| 파일 | 원본 | sha256 |
|---|---|---|
| `M1_행안부_20200605_3주만에64.hwp` | [www.mois.go.kr nttId=77678](https://www.mois.go.kr/frt/bbs/type010/commonSelectBoardArticle.do?bbsId=BBSMSTR_000000000008&nttId=77678) | `bc12df6260a2de8608e1f251965b76cb4e8901e71b104c332ca5b1a727d2d759` |
| `M2_행안부_20200611_음식점마트.hwp` | [www.mois.go.kr nttId=77785](https://www.mois.go.kr/frt/bbs/type010/commonSelectBoardArticle.do?bbsId=BBSMSTR_000000000008&nttId=77785) | `4c8c93378b5118ce4fd13622dc5a2b9a41297a22b3fafc7f31c12c7222bfc2c8` |
| `M4_행안부_20200923_지급완료.hwp` | [www.korea.kr newsId=156412201](https://www.korea.kr/briefing/pressReleaseView.do?newsId=156412201) | `5f1cb528f88620dd44ca70b67dcd028112b4f382eceb0f32367ba2f45a0c1020` |
| (E:) `K2_KDI_긴급재난지원금지급에관한연구II_2020.pdf` | [www.kdi.re.kr pub_no=16889](https://www.kdi.re.kr/research/reportView?pub_no=16889) | `966be0ab6ad4d90b53f5da36d8decc4658a363ac3f498f53cb1e07cd5681728e` |
| (E:) `C1_협동연구_종합편_2021.pdf` | [www.nkis.re.kr](https://www.nkis.re.kr/view/acpt/2021/07/OTP_202107070419506890.pdf) | `7c19fea796f374f606d06a3ad3d4e67853a121b9837d826952ea352123780044` |
| (E:) `C2_협동연구_지역편_2021.pdf` | [www.nkis.re.kr](https://www.nkis.re.kr/view/acpt/2021/07/OTP_202107070419509951.pdf) | `dfd42c4d4f89ede783f3c8af131360d10df7ff29c35b262ae2fc407934ba9cf4` |

M4 는 정책브리핑에서 `.pdf` 링크로 받았지만 실제로는 HWP(OLE) 파일이라 확장자를 고쳤다. HWP 본문은
olefile+zlib 로 문단 레코드를 풀어 읽었다(별도 변환 도구 없이).

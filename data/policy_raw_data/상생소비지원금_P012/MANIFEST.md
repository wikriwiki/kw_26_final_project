# 상생소비지원금_P012 — 파일 출처

다시 만들려면 `python scripts/report/collect_policy_raw_data.py`.
**원본**이 `http` 로 시작하면 공공기관에서 직접 내려받은 것이고,
경로면 저장소 안의 파일을 복사한 것이다.

| 파일 | 원본 | 크기 | sha256 |
|---|---|---:|---|
| `정책원문_상생소비지원금_시행방안_배포용.pdf` | `docs/references/1._상생소비지원금_시행방안(최종)_배포용.pdf` | 2.2MB | `a12163a2d205c6caaccf41939947ead629dac092b29fda9eacd2c0e372d578a9` |
| `정답지_KDI_상생소비지원금_효과분석_2022.pdf` | `docs/references/상생소비지원금효과분석.pdf` | 1.1MB | `92ae21e27092d183361fd1bce66a28a8fa9fd01e6b71ff238f95a0c4743521d0` |
| `정답지_KDI_효과분석_추출텍스트.txt` | `output/validation_reference/sources/kdi_cashback_2022.txt` | 156.8KB | `75393b0e4b09252bcc4be24627c68ca169a2c18cd4458cdb116cbec5f32349cb` |
| `시뮬정의_P012.json` | `data/neo4j_load/policies/P012.json` | 3.0KB | `5197090cc7d6441c96b423529643be0478f3604db12114bd9a1810344e8a593d` |

PDF 는 복사 전에 실제 내용이 있는지 확인했다 — `%PDF` 로 시작하고
`%%EOF` 로 끝나는지. G: 드라이브의 껍데기 파일을 거르기 위해서다.

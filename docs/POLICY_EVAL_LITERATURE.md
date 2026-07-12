# 정책평가 문헌·제도 조사 — 백테스트 설계의 학술·제도적 근거

> 작성일: 2026-07-07 · `POLICY_BACKTEST_RESEARCH.md`의 동반 문서 (검토용)
> 목적: ① 관련 실증 논문(국내·국제) ② 시뮬레이션 검증 방법론 계보 ③ 한국 제도권 정책평가 체계 → 백테스트 설계에 무엇을 차용할지 확정

---

## A. 한국 실증 논문 — 이전지출·쿠폰·지역화폐

| # | 연구 | 데이터·방법 | 핵심 결과 | 우리 백테스트에의 시사 |
|---|---|---|---|---|
| A1 | 김미루·오윤해, KDI 정책포럼 (2020) 「1차 긴급재난지원금 정책의 효과와 시사점」 | 카드매출, 업종 비교 DiD | 투입 대비 매출 증가 **26.2~36.1%**, 대면서비스 미미 | 전이율 대역의 하한 앵커 |
| A2 | [서울시민 카드 데이터 연구 (국토계획 게재)](https://kpaj.or.kr/_common/do.php?a=full&bidx=2564&aidx=29180) | **서울 카드 거래**, 지역별 | 지급 후 6주 내 **약 +29%** 소비 진작, **고소득 지역일수록 효과 낮음** | ⭐ T3(공간)·T4(분위)의 **서울 단위 정답이 학술적으로 이미 추정돼 있음** — 우리 공간 예측과 직접 대조 가능. 원문 정독 필수 |
| A3 | [「긴급재난지원금 현금수급가구의 소비 효과」 (KCI 2021)](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002705252) | 현금 수급(취약)가구 | 증가 소득의 **70~80%를 소비 지출** | 저소득층 MPC 상단 앵커 (시뮬 INCOME_PRIOR '하' 0.90과 대조) |
| A4 | [「코로나19와 1차 긴급재난지원금이 가구 소득과 지출에 미친 영향」 (KCI)](https://www.kci.go.kr/kciportal/landing/article.kci?arti_id=ART002758339) | 가구패널 | 가구 단위 소득·지출 반응 | 분위별(T4) 보조 정답 |
| A5 | [송경호·이환웅 (KIPF, 한국경제의 분석 2021) 「지역화폐의 경제적 효과」](https://www.kci.go.kr/kciportal/landing/article.kci?arti_id=ART002786657) | 통계청 빅데이터, 소상공인 매출 | **동네슈퍼마켓 업종만 유의한 매출 증대**, 타 업종 효과 없음 — 업종 편차 큼 | ⭐ "사용처 제한 정책의 효과는 업종에 집중된다"는 정설 → **T2(업종 이질성)가 옳은 1차 과녁**이라는 근거 |
| A6 | [김영철 외, 「지역화폐 도입의 지역경제 영향: 학술적 평가와 점검」 (한국경제연구)](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002765045) | 방법론 리뷰 | 지역화폐 효과 추정의 식별 논쟁 정리 | 식별 전략 함정 목록 (우리 Layer 1 설계 시 회피) |
| A7 | KDI (2025) 소비쿠폰 효과 2건 (연구 문서 §2 참조) | 사용처/비사용처 DiD | +4.93%(6주)·비사용처 무변화·전이율 42.5%·영세 집중 | **주 비교 앵커** |
| A8 | [상생소비지원금 (2021.10~11, 카드 캐시백)](http://policywiki.kr/%EC%83%81%EC%83%9D%EC%86%8C%EB%B9%84%EC%A7%80%EC%9B%90%EA%B8%88) | — | 월 카드사용액 2분기 평균 대비 3% 초과분의 10% 환급 | 예비 후보(한계 인센티브형 — 구조 복잡해 2순위), placebo 창 설계 시 겹침 주의 (10~11월) |

---

## B. 국제 실증 — 쿠폰·현금 지급의 MPC (전이율 과녁 대역)

| 정책 (유형) | 대표 연구 | MPC / 효과 |
|---|---|---|
| 일본 1999 지역진흥권 (쿠폰, 사용기한 6개월·지역 제한) | [Hsieh, Shimizutani & Hori, *J. Public Economics* (2010)](https://www.sciencedirect.com/science/article/abs/pii/S0047272710000241) | 준내구재만 소폭 증가, **전반 효과 미미** — 대체(현금 전용) 큼 |
| 대만 2009 소비권 (쿠폰, 전 국민) | Kan, Peng & Wang (2017) ([관련 분석](https://www.mdpi.com/2071-1050/12/12/4895)) | **MPC ≈ 0.24** — 약 3/4은 기존 소비 대체 |
| 미국 2001 세금환급 | Johnson, Parker & Souleles, *AER* (2006) | 비내구재 약 20~40% (분기) |
| 미국 2008 경기부양 지급 | Parker et al., *AER* (2013) | 12~30% (내구재 포함 시 상회), 저유동성 가구 ↑ |
| 미국 2020 CARES | Baker et al. (2020); Chetty et al. (Opportunity Insights) | 초기 수주 25~40%, 저소득·저유동성 집중, 실시간 카드 데이터 방법론 |
| 일본 2020 특별정액급부금 (현금 10만엔) | Kubota, Onishi & Toyama (2021, 은행계좌 데이터) | 초기 수주 내 소폭 (현금형은 저축 누출 큼) |
| 중국 디지털 쿠폰 (도시별, 할인 매칭형) | [NBER w27596](https://www.nber.org/system/files/working_papers/w27596/w27596.pdf) · [BFI 실험](https://bfi.uchicago.edu/wp-content/uploads/China-coupon-experiment-full-version-V4.pdf) · [최근 arXiv](https://arxiv.org/html/2507.01365) | 소액 할인쿠폰의 레버리지 효과 — 설계(매칭 비율)에 민감 |
| **한국 2020 재난지원금** (카드 포인트형) | A1 | **26.2~36.1%** |
| **한국 2025 소비쿠폰** (사용처·기한 제한) | A7 | **42.5%** |

**패턴 (V3 대역 설정 근거):** 순수 현금 < 카드 포인트 < **사용처·기한 제한 쿠폰** 순으로 전이율이 높다 (제한이 강제 소비를 유도). 한국 2025의 42.5%가 국제 대역 상단인 것이 이 패턴과 정합. → 시뮬 판정 대역 제안: **전이율 ∈ [0.25, 0.55]** (국제 문헌 스팬), 1차 목표는 KDI 점추정 42.5% ± 10%p.

---

## C. 시뮬레이션 기반 정책평가·검증 — 우리 방법의 학술 계보

| 연구 | 내용 | 우리가 차용할 것 |
|---|---|---|
| ⭐ [Poledna, Miess & Hommes, 「Economic Forecasting with an Agent-Based Model」 *European Economic Review* (2023)](https://www.sciencedirect.com/science/article/pii/S0014292122001891) | 오스트리아 전체 경제 ABM이 **out-of-sample 예측에서 VAR·DSGE와 동급~우위**(RMSE), COVID 록다운 효과 예측에 실사용 | **검증 프로토콜의 원형**: ① out-of-sample 기간 고정 ② 표준 오차지표(RMSE) ③ 기존 방법 대비 벤치마크. "ABM도 예측력이 입증될 수 있다"는 선례 |
| [Hommes & Poledna (2023)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4381261) | 유로지역 위기 예측 ABM | 위기(충격) 시기 예측 확장 |
| CANVAS — Bank of Canada 행동 ABM | 중앙은행이 예측·통화정책 분석에 ABM 채택 ([서베이](https://arxiv.org/html/2412.16591v1)) | "제도권이 시뮬레이션을 정책 도구로 수용"하는 흐름 |
| Windrum, Fagiolo & Moneta, *JASSS* (2007) | ABM 실증 검증 방법 분류 (input/output validation) | 용어·프레임: 우리는 input(페르소나 실측) + **output(정책 반응) 검증** |
| Johnson-Parker류 자연실험 + Chetty 실시간 카드 | 준실험 인과추정의 표준 | Layer 1(실측 효과 추정)의 방법 |
| EconAgent (ACL 2024) · Generative Agents (Park 2023) · **1,000명 개인 복제 (Park et al. 2024, 태도 재현 ~85%)** · AgentSociety (2025) | LLM 에이전트 사회 시뮬의 타당성 검증 흐름 | LLM-ABM 계보. 단 이들은 태도·정형사실 재현까지 — **"실제 정책 자연실험에 대한 out-of-sample 백테스트"는 문헌 공백 → 우리의 기여 주장 지점** |

---

## D. 한국 제도권 정책평가 체계 (조사 결과)

| 제도 | 근거 | 주체 | 시점 | 방법 |
|---|---|---|---|---|
| [정부업무평가](https://www.evaluation.go.kr/web/page.do?menu_id=23) | 정부업무평가기본법 (2006) | 국무조정실 (정부업무평가위) | 연례 | 자체평가 + 특정평가 (성과지표·만족도) |
| [예비타당성조사](https://pimac.kdi.re.kr/study/study_list.jsp?classcd=F1) | 국가재정법 §38 | **KDI 공공투자관리센터(PIMAC)** (+[KIPF 정부투자분석센터](https://gafsc.kipf.re.kr/gmac/OtherData_preliminary.do)) | **사전** | 경제성(B/C)·정책성·지역균형 → AHP. 총사업비 500억+국비 300억 이상 |
| 재정사업 자율평가·[심층평가](https://www.moef.go.kr/com/cmm/fms/FileDown.do?atchFileId=ATCH_OLD_00004011636&fileSn=427946) | 국가재정법 | 기재부·KDI | **사후** | 성과지표 / 심층: **계량 인과추정(DiD 등)** |
| [조세특례 예타·심층평가](https://pimac.kdi.re.kr/about/depth.jsp) | 조세특례제한법 | KIPF·KDI | 사전+사후 | 조세지출 효과 계량 |
| 고용영향평가 | 고용정책기본법 | 고용부·한국노동연구원 | 사전+사후 | 고용 효과 계량 |
| 규제영향분석 | 행정규제기본법 | 규제개혁위 | 사전 | 비용편익 |
| [NABO 사업평가](https://www.nabo.go.kr/Sub/01Report/01_01_Board.jsp) | 국회법 | 국회예산정책처 | 사후 (독립) | 결산·사업평가 보고서 |
| 감사원 성과감사 | 감사원법 | 감사원 | 사후 | 성과감사 |

### 제도 조사에서 나온 우리 프로젝트의 포지셔닝 (발표 스토리)

1. **한국 평가 체계는 사전(예타)–사후(심층평가) 이원 구조**인데, 소비쿠폰 같은 긴급 이전지출은 예타가 사실상 면제되어 **"사전 정량 평가의 공백"**이 존재한다 (13.5조가 사전 효과 추정 없이 집행 → 사후에야 KDI 분석).
2. 우리 시뮬레이션 = 그 공백을 메울 **사전(ex-ante) 평가 도구** 후보. 백테스트는 "이 도구를 사후(ex-post) 실측으로 역검증"하는 작업 — 제도권 용어로 바로 번역된다: *"심층평가(DiD)가 사후에 확인한 효과를, 시뮬레이션은 사전에 근사할 수 있는가."*
3. 방법 정합성: 제도권 심층평가가 쓰는 계량기법(DiD)이 우리 Layer 1과 동일 → "우리 잣대"가 아니라 **제도권의 잣대**로 채점받는 구조.

---

## E. 설계 반영 사항 (`POLICY_BACKTEST_RESEARCH.md` 업데이트 제안)

1. **V3(T1) 판정 대역 확정**: 전이율 [0.25, 0.55] (국제 스팬) / 1차 목표 42.5%±10%p (KDI 점추정) — B표 근거.
2. **T2 정당화 강화**: 사용처 제한 정책의 효과는 업종 집중이 정설(A5 지역화폐, A7 쿠폰) → 업종 이질성 부호가 1차 과녁.
3. **T3·T4 정답 추가**: A2(서울 카드, 지역·소득 이질성 +29%) 원문 확보 → 공간·분위 정답 벡터로 사용.
4. **검증 프로토콜 차용** (Poledna): out-of-sample 기간·지표(RMSE/부호정합/Spearman)·벤치마크(naive 예측: "효과 균등 분배" 널모형) 사전 고정.
5. **기여 주장 문구**: "LLM 기반 도시 시뮬레이션을 실제 재정정책 자연실험으로 out-of-sample 백테스트한 첫 사례(문헌 공백)" — C표 근거.
6. **후속 읽기 목록** (원문 정독 순서): A2 → A7 2건 → A5 → Poledna(2023) → A1.

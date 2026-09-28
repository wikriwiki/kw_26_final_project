"""Freeze outcome definitions reviewed on whole PDF pages before CPU scoring.

No catalog value is sent to a citizen prompt. Model simulations in the source
paper and control-variable estimates are identified separately from outcomes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/pdf_benchmark_expansion_20260928"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entry(id_, label, value, unit, page, printed, locator, *, kind="policy_effect",
          period="2011–2018", population="전국 시군구×업종 패널", denominator="지역화폐 판매량 / 2010년 GRDP",
          method="삼중차분, 시군구 추세 포함", requirement="실측과 동일한 업종·기간·정책 강도 정의 및 지역별 패널이 필요",
          formula="실측과 동일한 추정식으로 재추정", family=None, **extra):
    return {"id": id_, "label": label, "benchmark_kind": kind,
            "independent_outcome_family": family or id_,
            "count_as_new_indicator": kind not in {"specification_sensitivity", "conditional_purchase_level", "source_model_estimate"},
            "empirical": {"value": value, "unit": unit, "pdf_page": page,
                          "printed_page": printed, "locator": locator, "approximate": False,
                          "period": period, "population": population, "denominator": denominator,
                          "method": method, **extra},
            "simulation": {"feasibility": "requires_measurement_or_matching_scenario",
                           "proposed_formula": formula, "existing_data": "기존 시민·구매 원장",
                           "missing_requirements": requirement},
            "direct_gap_allowed": False}


def write(policy, source, pages, entries, note):
    obj = {"schema": "pdf_benchmark_catalog_v1", "policy": policy,
           "timing": "post-result exploratory", "source": {
               "path": source.relative_to(ROOT).as_posix(), "sha256": sha(source), "pages": pages},
           "review_note": note,
           "entries": entries}
    dest = OUT / f"{policy.lower()}_catalog.json"
    text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    if dest.exists() and dest.read_text(encoding="utf-8") != text:
        raise ValueError(f"catalog already frozen: {dest}")
    dest.write_text(text, encoding="utf-8")
    print(json.dumps({"policy": policy, "entries": len(entries), "sha256": sha(dest)}, ensure_ascii=False))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p014 = next((ROOT / "data/policy_raw_data/지역사랑상품권_P014").glob("정답지*.pdf"))
    rows = []
    sectors = [("47121", "슈퍼마켓", .141, .064, 79, 62),
               ("47129", "기타 음·식료품 위주 종합소매업", .082, .034, 79, 62),
               ("472", "음·식료품 및 담배 소매업", .134, .125, 79, 62),
               ("477", "연료 소매업", .004, .024, 79, 62),
               ("561", "음식점업", -.052, .033, 79, 62),
               ("476", "문화·오락 및 여가용품 소매업", .033, .057, 80, 63),
               ("478", "기타 상품 전문 소매업", .029, .020, 80, 63),
               ("961", "미용·욕탕 및 유사 서비스업", .024, .023, 80, 63)]
    for code, label, value, se, page, printed in sectors:
        rows.append(entry(f"P014_KSIC_{code}", f"{label} 매출 효과", value, "로그 회귀계수", page, printed,
                          f"표 VI-{6 if page == 79 else 7}, (3)열 l_ct×I(ind)×I(업종)",
                          standard_error=se, ksic=code,
                          requirement="2026 KSIC 매핑의 기존 전체 99% 관문 실패를 해소하고, 2011–2018 패널/정책판매 강도를 맞춰야 함. 현재 3일 ON/OFF는 같은 추정량 아님"))
    for weighted, page, printed, values in [(True, 75, 58, [-.019, -.023, .007, -.042]),
                                           (False, 77, 60, [.044, .025, .058, .035])]:
        for i, value in enumerate(values):
            outcome = "매출액" if i < 2 else "사업체 수"
            col = 4 + i % 2
            rows.append(entry(f"P014_TOTAL_{'W' if weighted else 'U'}_{i}",
                              f"전체 소상공인 {outcome}: {'가중' if weighted else '비가중'} 모형 {col}",
                              value, "로그 회귀계수", page, printed,
                              f"표 VI-{4 if weighted else 5}, {'A' if i < 2 else 'B'}패널 ({col})열",
                              kind="policy_effect" if weighted and col == 4 else "specification_sensitivity",
                              family=f"P014_TOTAL_{outcome}",
                              requirement="동일 패널과 처치 강도 및 가중 모형 필요. 사업체 수는 개·폐업을 관측하는 사업체 패널도 필요"))
    names = ["일반휴게음식", "유통업", "음료식품", "학원", "보건위생", "의원", "의류", "약국", "레저용품",
             "연료판매점", "레저업소", "회원제형태", "신변잡화", "자동차정비", "기타의료기관", "서적문구", "가구",
             "직물", "문화취미", "병원", "수리서비스", "건강식품", "서비스", "전기", "건축자재", "사무통신", "기타",
             "주방용구", "농업", "광학사진", "숙박", "자동차판매", "여행"]
    payment = [29.4,20.3,9.8,7.3,5.5,4,3.2,3.1,2.1,1.7,1.7,1.4,1.4,1.2,1,1,1,.8,.7,.7,.6,.4,.3,.3,.3,.2,.2,.2,.2,.1,.1,0,0]
    merchants = [25.6,4.1,6.7,7.8,9.5,2.2,3.3,.9,1.3,.4,3.7,2.3,2.1,3.5,.5,1.6,1,1,2.7,.1,1.5,.7,4.9,1.2,3.5,1.6,2.5,.6,.2,.4,1.1,.3,1.2]
    for i, (name, pay, count) in enumerate(zip(names, payment, merchants), 1):
        for dimension, value, denom in [("PAY_SHARE", pay, "지역화폐 총 결제액"), ("MERCHANT_SHARE", count, "지역화폐 전체 가맹점 수")]:
            rows.append(entry(f"P014_{dimension}_{i:02d}", f"{name} {'결제액' if dimension == 'PAY_SHARE' else '가맹점 수'} 비중",
                              value, "%", 84, 67, f"표 VII-1 {i}번 행",
                              kind="usage_distribution" if dimension == "PAY_SHARE" else "merchant_distribution",
                              period="2020년 상반기", population="경기도 카드형 지역화폐",
                              denominator=denom, method="가맹점 및 실제 카드형 지역화폐 결제 집계",
                              formula="동일 카드업종별 지역화폐 결제액 / 지역화폐 총 결제액 × 100" if dimension == "PAY_SHARE" else "동일 업종별 가맹점 수 / 총 가맹점 수 × 100",
                              requirement="P014의 유상 구매·액면 잔액·지역화폐 실제 결제 원장 및 경기도 2020 상반기 대상/33개 카드업종 매핑 필요. 현재 purchase amount를 지역화폐 결제액으로 대체하지 않음"))
    ticket_names = ["병원","유통업","연료판매점","약국","기타의료기관","의원","레저용품","음료식품","일반휴게음식","의류","가구","학원","직물","신변잡화","농업","서적문구","회원제형태","보건위생","건강식품","레저업소"]
    ticket = [22.01,12.89,11.74,8.88,5.83,4.73,4.22,3.78,2.96,2.45,2.44,2.40,1.97,1.67,1.67,1.66,1.56,1.50,1.43,1.20]
    for i, (name, value) in enumerate(zip(ticket_names, ticket), 1):
        rows.append(entry(f"P014_MERCHANT_MEAN_{i:02d}", f"{name} 가맹점당 지역화폐 결제액", value, "백만원/가맹점", 85, 68,
                          f"표 VII-2 {i}번 행", kind="merchant_payment_level", period="2020년 상반기", population="경기도 카드형 지역화폐",
                          denominator="업종별 가맹점 수", method="가맹점당 실제 지역화폐 결제 집계",
                          formula="업종별 실제 지역화폐 결제금액 / 해당 업종 모든 가맹점 수 / 1,000,000",
                          requirement="전체 가맹점 수와 카드형 지역화폐 결제 원장 필요. 구매가 있었던 POI만 분모로 쓰면 관측 편향"))
    for i, (label, pay, count) in enumerate(zip(["10억원 이상", "5~10억원", "3~5억원", "3억원 이하"], [21,21,16,41], [4,9,10,78]), 1):
        for dim, value in [("PAY", pay), ("MERCHANT", count)]:
            rows.append(entry(f"P014_SIZE_{dim}_{i}", f"연매출 {label} 가맹점 {'결제액' if dim == 'PAY' else '수'} 비중", value, "%", 88, 71,
                              f"표 VII-3 {'B' if dim == 'PAY' else 'A'}패널", kind="merchant_size_distribution",
                              period="2020년 상반기", population="경기도 카드형 지역화폐, 신규 가맹점 제외",
                              denominator="신규 제외 실제 결제액 또는 가맹점 수", method="행정 결제/가맹점 집계(원문 반올림)",
                              requirement="POI별 연매출과 신규 가맹점 여부, 실제 지역화폐 결제 및 가맹점 원장 필요"))
    write("P014", p014, 101, rows, "PDF 75/77/79/80/84/85/88 전체 페이지 시각 확인. 강건성 모형을 독립 성과로 중복 집계하지 않음. 경기도 결제 구성은 서울/전국 정책 효과와 구분.")

    p016 = next((ROOT / "data/policy_raw_data/농축산물할인쿠폰_P016").glob("정답지*.pdf"))
    rows = []
    for name, value, page, printed in [("과일",8.044,69,59),("채소",4.601,69,59),("축산",6.919,69,59),("곡물",3.465,69,59),("전체",6.957,70,60)]:
        rows.append(entry(f"P016_DID_{name}", f"2020년 {name} 농축산물 구매액 효과", value, "%", page, printed,
                          "표 3-2 DID 추정치(%)", period="2020년 할인사업 기간", population="카드 소비자료의 농축산물 구매액",
                          denominator="동일 품목 구매액 반사실", method="품목별 DID, 100×(exp(β−Var(β)/2)−1)",
                          requirement="영수증 상품행에 과일·채소·축산·곡물 품목/수량/가격을 기록하고 같은 기간/대상으로 추정. POI 청과만으로 과일과 채소를 분리하지 않음"))
    for id_, label, value, unit in [("MART_TOTAL","대형마트 전체 매출",4.6,"%"),("MART_AGRI","대형마트 농축산물 매출",11.6,"%"),("MART_SHARE","대형마트 농축산물 매출 비중 변화",1.9,"%p"),("ONLINE_TOTAL","온라인 전체 매출",23,"%"),("ONLINE_AGRI","온라인 농축산물 매출",44,"%"),("ONLINE_SHARE","온라인 농축산물 매출 비중 변화",2,"%p")]:
        rows.append(entry("P016_DESCRIPTIVE_"+id_, "2020년 "+label, value, unit, 20, 10, "2020년 연구 결과 요약, 할인사업 매출 비교",
                          kind="descriptive_sales_change", period="2020-07-30~2020-11-30", population="할인사업 참여 대형마트/온라인 채널",
                          denominator="동일 채널 정책전 비교기준", method="사업 기간 매출 기술 통계, 정책 인과효과와 다름",
                          requirement="4개월 기간, 참여채널, 동일 비교기준과 상품행 필요. 현재 3일 시민 ON/OFF는 기간·모집단·추정량이 다름"))
    for id_, label, value in [("SEX_F","여성 대 남성",.87868),("AGE_40_59","40~59세 대 40세 미만",-1.05678),("AGE_60_PLUS","60세 이상 대 40세 미만",-1.05546)]:
        rows.append(entry("P016_CONDITIONAL_"+id_, label+" 농축산물 구매액 조건부 차이", value, "로그 회귀계수", 70, 60,
                          "표 3-2 전체열 성별/연령 통제변수", kind="conditional_purchase_level", period="2020년 분석기간",
                          population="카드 소비자료 전체 농축산물 구매액", denominator="기준 집단 및 다른 통제변수 고정",
                          method="DID 회귀의 통제변수 계수, 할인정책 효과 아님",
                          requirement="연령 실측분포 및 실제 구매 상품행과 같은 회귀 통제 필요. 표본군별 비보정 평균과 구분"))
    for i, (name, per_budget, multiplier, demand) in enumerate(zip(["곡물","채소","과일","축산","계란"], [7.8,5.5,4.5,6.3,10.0], [.30,1.27,.39,.86,1.17], [.43,.94,.81,.77,5.75]), 1):
        rows.append(entry(f"P016_2021_BUDGET_SALES_{i}", f"2021년 {name} 해당 품목 예산 1억원당 판매액 변화", per_budget, "%/해당 품목 예산 1억원", 82, 72,
                          "표 3-5 A행", period="2021년", population="분석 대상 대형마트", denominator="해당 품목에 투입된 할인 예산 1억원",
                          method="고정효과 패널회귀, 원문 β×100 근사", requirement="2021년 별도 정책 배경·품목 예산 및 판매액 시계열 필요. 2020년 3일 런 값을 재사용하지 않음"))
        rows.append(entry(f"P016_2021_BUDGET_MULTIPLIER_{i}", f"2021년 {name} 전체 사업 예산 1원당 판매액 증대", multiplier, "원/전체 사업 예산 1원", 82, 72,
                          "표 3-5 E행", kind="source_model_estimate", period="2021년", population="대형마트 추정모형",
                          denominator="사업 전체 예산 1원", method="실증 추정치×품목예산 배정비율, 파생 추정량", family=f"P016_2021_BUDGET_SALES_{i}",
                          requirement="실측 원장 자체가 아닌 논문 파생 추정치. 가격/상품/예산배분 원장 갖춘 2021 시나리오가 필요"))
        rows.append(entry(f"P016_2021_ANNUAL_DEMAND_{i}", f"2021년 {name} 연간 수요 증대 추정", demand, "%", 83, 73,
                          "표 3-6 연간 수요 증대율 행", kind="source_model_estimate", period="2021년", population="전국 산업연관표 기반 모형",
                          denominator="2019 산업연관 민간수요에 물가를 반영한 2021 수요", method="표 3-5 계수로 산출한 모형 추정치, 직접 관측값 아님",
                          family=f"P016_2021_BUDGET_SALES_{i}", requirement="관측 지표와 별도 참고. 연간 수요/가격/공급 대응 및 예산배분 모형 필요"))
    write("P016", p016, 97, rows, "PDF 20/69/70/71/82/83/86 전체 페이지 시각 확인. 품목별 소비 효과를 확인했으나 현 원장에는 상품행이 없음. 논문 균형가격/수량 모형 결과는 실측 검증지표로 수록하지 않음. 2021년 별도 정책기간과 파생 모형값을 분리.")


if __name__ == "__main__":
    main()

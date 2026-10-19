"""Post-result PDF benchmark catalog and read-only v53 ledger decompositions.

prepare writes definitions transcribed from visually checked whole PDF pages.
score reads the frozen catalogs; it never calls a model or a graph.  None of
these same-calendar ON/OFF proxies is the PDF's 2019/2020 YoY estimator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "experiments/pdf_benchmark_expansion_20260928"
OUT = ROOT / "output/pdf_benchmark_expansion_20260928"
PDFS = {
    "P013": "data/policy_raw_data/긴급재난지원금_P013/정답지_KDI_FOCUS_1차긴급재난지원금_효과와시사점_2020.pdf",
    "DISTANCING_2020": "data/policy_raw_data/사회적거리두기_DISTANCING2020/정답지_서울연구원_코로나19_서울_경제적손실_2021.pdf",
}
SOURCE_CSV = Path(r"C:\Users\Administrator\Documents\kw26_a100_recovery_20260926\multipolicy_v53_20260928\p014\source_recovery\seoul202603_recovered_7374e0ed.csv")
CSV_SHA = "7374e0edda19beb6f89c64918c7eb3e7a89007121bddbf2fc0f78aca310d2506"
MAPPING = ROOT / "data/neo4j_load/mapping/mapping_upjong_to_sub.json"
P013_ARCHIVE = ROOT / "output/recovery_20260927/p013_v53_pilot_20260927/all_artifacts.tar.gz"
DIST_BASE = ROOT / "output/recovery_20260928/multipolicy_v53/distancing"
P013_DAYS = ["2020-05-11", "2020-05-12", "2020-05-13"]
DIST_DAYS = ["2020-11-24", "2020-11-25", "2020-11-26"]
COMMON_SCOPE = ("기존 동결 v53의 비가중 소규모 시민 표본, 같은 달력 날짜의 정책 ON/OFF 비교이다. "
                "성별·연령·행정동·소득의 실측 인구분포를 맞춘 대표 표본이 아니다. "
                "2019년 기준 원장과 장기 관찰이 없으므로 원문의 전년동기 증감률/회귀효과와 직접 오차를 계산하지 않는다.")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(path: Path) -> dict:
    try:
        label = path.relative_to(ROOT).as_posix()
    except ValueError:
        label = str(path)
    return {"path": label, "sha256": sha(path)}


def write(path: Path, obj: dict, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and path.exists():
        raise ValueError(f"frozen catalog already exists: {path}")
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def emp(value, unit, page, locator, *, population, period, method,
        denominator, printed=None, **extra) -> dict:
    return {"value": value, "unit": unit, "denominator": denominator,
            "population": population, "period": period, "method": method,
            "pdf_page": page, "printed_page": page if printed is None else printed,
            "locator": locator, "approximate": False, "verified_visually": True,
            "visual_evidence": f"tmp/pdfs/pdf_benchmark_expansion_20260928/{'P013' if printed is None else 'DISTANCING_2020'}/page-{page if printed is None else str(page).zfill(2)}.png",
            **extra}


def sim(mapping=None, *, feasibility="supported_existing_ledger", missing=(),
        formula="100 * (ON sector purchases - OFF sector purchases) / OFF sector purchases",
        note="") -> dict:
    return {"feasibility": feasibility, "proposed_formula": formula,
            "poi_mapping": mapping or {}, "existing_data": "completed frozen v53 canonical metrics / receipts or sector ledger",
            "missing_requirements": list(missing), "scope_note": note}


def entry(id_, label, empirical, simulation, *, kind="descriptive_sales_change",
          existing=None, new=True, family=None, **extra) -> dict:
    row = {"id": id_, "benchmark_kind": kind, "label": label,
           "empirical": empirical, "simulation": simulation,
           "direct_gap_allowed": False, "count_as_new_indicator": new,
           "independent_outcome_family": family or id_, **extra}
    if existing:
        row["existing_indicator_id"] = existing
    return row


def base(policy: str, pages: int, entries: list[dict]) -> dict:
    pdf = ROOT / PDFS[policy]
    return {"schema": "pdf_benchmark_catalog_v1", "policy": policy,
            "source": {"path": PDFS[policy], "sha256": sha(pdf), "pages": pages},
            "timing": "post-result exploratory", "definition_frozen_before_cpu_scoring": True,
            "sampling_caveat": COMMON_SCOPE,
            "counting_rule": "각 업종/집단의 독립 결과를 구분하되 같은 통계의 단위변환·증감액/증감률·시계열 점들은 별도 독립지표 수로 부풀리지 않는다.",
            "entries": entries}


def prepare_p013() -> dict:
    pop = "전국 8개 카드사(BC·신한·국민·NH·롯데·삼성·현대·하나) 합산 카드매출; 현금소비 제외"
    causal = dict(population=pop, period="2020년 지원금 지급 후 19~33주, 지급 전 추세로 합성대조군 설정",
                  method="합성대조군 구성 후 이중차분(DID), 매출 증감률의 추가 증가분",
                  denominator="사용가능 업종의 카드매출 증감률; 현금소비·전체 GDP가 아님")
    entries = []
    for id_, label, value, mapping, old in [
        ("P013-PDF-SEMI", "(준)내구재 카드매출 추가 증가", 10.8,
         {"by_sub": ["가구", "문구", "안경", "의류"], "missing": ["서점", "잡화 세부분류"]}, "EM-4"),
        ("P013-PDF-ESSENTIALS", "필수재 카드매출 추가 증가", 8.0,
         {"by_sub": ["슈퍼마켓", "식료품", "청과", "정육", "수산", "편의점"], "mapping_kind": "식품 소매 POI 대리; 생협/마트 상세 분류 미완비"}, None),
        ("P013-PDF-FACE", "대면서비스 카드매출 추가 증가", 3.6,
         {"by_l1": ["여가", "미용"], "by_sub": ["욕탕·신체관리"], "mapping_kind": "기존 EM-4의 넓은 장소 대리"}, "EM-4"),
        ("P013-PDF-FOODSERVICE", "음식업 카드매출 추가 증가", 3.0,
         {"by_l1": ["식사", "카페", "디저트"], "mapping_kind": "음식점·커피·제과·간이음식의 장소 대리"}, None),
    ]:
        entries.append(entry(id_, label, emp(value, "%p", 4, "그림 3 및 본문; 업종 정의 각주", **causal),
                             sim(mapping, note="원문 DID %p를 같은 기간 ON/OFF %와 동일시하지 않는다."),
                             kind="policy_effect", existing=old, new=old is None))
    descriptive = [
        ("CLOTHING", "의류/잡화", -17.8, 11.2, ["의류"], "잡화 포함 범위가 현 POI 의류보다 넓다."),
        ("FURNITURE", "가구", -3.5, 19.9, ["가구"], "가구 POI 장소 매핑"),
        ("TRAVEL", "여행", -61.1, -55.6, ["여행사"], "여행 상품 카드매출 전체가 아닌 여행사 POI 구매 대리"),
        ("BATH", "사우나/찜질방/목욕탕", -26.3, -20.9, ["욕탕·신체관리"], "마사지 등이 섞일 수 있어 정확한 업종 crosswalk가 필요하다."),
        ("GROCERY", "식료품", 2.5, 12.3, ["식료품"], "원문 카드업종 식료품과 POI 식료품 소매 분류의 일치 확인 필요"),
        ("CONVENIENCE", "편의점", 0.8, 5.6, ["편의점"], "편의점 POI 장소 매핑"),
    ]
    for code, label, before, after, names, note in descriptive:
        entries.append(entry("P013-PDF-YOY-" + code, label + " 카드매출 전후 전년동기 증감률",
                             emp({"pre": before, "post": after}, "% (전년동기)", 3, "본문의 16~18주 대 20~25주 비교",
                                 population=pop, period={"pre": "2020년 16~18주", "post": "2020년 20~25주"},
                                 method="각 기간의 평균 전년동기 카드매출 증감률, 정책 순효과 아님",
                                 denominator="각 업종의 2019년 같은 주 카드매출"), sim({"by_sub": names}, note=note)))
    entries.append(entry("P013-PDF-SALES-RATIO", "사용가능 업종 증분 카드매출 / 영향 가능 지원액",
                         emp([26.2, 36.1], "%", 4, "본문 첫 번째 강조 문단",
                             population=pop, period="2020년 지급 후 19~33주",
                             method="추정 증분 카드매출 4.0조원을 지원액 분모 11.1~15.3조원으로 나눈 범위",
                             denominator="중앙·지방 지원 중 카드매출에 영향을 줄 것으로 추정한 예산 11.1~15.3조원",
                             raw_components={"incremental_sales_trillion_won": 4.0, "eligible_budget_trillion_won": [11.1, 15.3]}),
                         sim({"ledger": "eligible_offline_spent / grant_received_cumulative"},
                             formula="100 * sum(ON eligible purchases - OFF eligible purchases) / sum(ON grants received)",
                             note="분모는 실험 시민에게 지급한 모델 지원금이며 전국 카드매출 영향 가능 예산과 다르다."),
                         kind="policy_effect", new=True))
    weeks = ["5월1주", "5월2주", "5월3주", "5월4주", "6월1주", "6월2주", "6월3주", "6월4주", "7월1주", "7월2주", "7월3주", "7월4주", "7월5주", "8월1주", "8월2주"]
    values = [-0.6, 6.4, 16.8, 15.3, 6.4, -0.4, 0.3, 0.7, 1.2, 0.6, 1.1, 3.0, -1.4, -2.0, -6.9]
    entries.append(entry("P013-PDF-WEEKLY-EFFECT", "지원금 지급 후 주별 증분 카드매출 시계열",
                         emp(dict(zip(weeks, values)), "천억원", 5, "그림 4의 인쇄된 막대 수치",
                             population=pop, period="2020년 5월1주~8월2주", method="시기별 증분 매출 효과 추정",
                             denominator="전국 사용가능 업종 총 카드매출, 표본 시민당 금액이 아님"),
                         sim(feasibility="needs_more_days_and_population_scaling",
                             missing=["각 주 전체 7일", "5~8월의 지연·소진·재유행 경로", "전국 인구·카드결제 확장 가중치", "합성대조군/2019 카드매출"],
                             formula="주별 sum(ON cardlike purchases - OFF cardlike purchases), 같은 결과의 시계열로 보고"),
                         kind="policy_effect_time_series", family="P013-PDF-WEEKLY-EFFECT"))
    incomes = dict(population="지원금 신한카드 수령 가구 중 전국 1인가구 6.85만 표본; 취약계층 현금 수령 가구 제외",
                   method="KCB 전 카드소비와 신한 수령 가구 자료 결합, 연소득 추정치 5분위",
                   denominator="1인가구당 전 카드소비 금액; 시민 개인·지원금 MPC 아님")
    thresholds = [27000000, 34000000, 45000000, 66000000, 150000000]
    for q, june, difference, upper in zip(range(1, 6), [30.3, 18.8, 16.9, 11.2, 22.2], [35.9, 20.9, 22.9, 26.9, 21.6], thresholds):
        mapping = {"income_quintile": q, "annual_income_upper_won": upper, "household_size": 1}
        missing = ["1인가구 식별·가구단위 집계", "KCB 추정 연소득 기준 5분위 crosswalk", "2019년 같은 월 카드소비", "2020년 4~6월 완주", "현금 수령 취약계층 제외 기준"]
        entries.append(entry(f"P013-PDF-INCOME-Q{q}-JUNE", f"전국 1인가구 소득 {q}분위 6월 전년동기 소비 증가액",
                             emp(june, "만원/1인가구", 6, "6월 소득분위별 소비 증가액 본문; 표본/소득경계 그림 5 각주(p7)",
                                 period="2020년 6월 대 2019년 6월", **incomes),
                             sim(mapping, feasibility="needs_household_income_and_yoy_baseline", missing=missing,
                                 formula="해당 1인가구 소득분위의 2020년6월 카드소비 - 2019년6월 카드소비 평균")))
        entries.append(entry(f"P013-PDF-INCOME-Q{q}-MAYAPR", f"전국 1인가구 소득 {q}분위 지급 전후 전년동기 소비 변화",
                             emp(difference, "만원/1인가구", 6, "마지막 본문: 5월 전년동월 증가액 - 4월 전년동월 증가액",
                                 period="2020년 5월 YoY 증가액 - 2020년 4월 YoY 증가액", **incomes),
                             sim(mapping, feasibility="needs_household_income_and_yoy_baseline", missing=missing,
                                 formula="mean[(2020May-2019May)-(2020Apr-2019Apr)] by original household income quintile")))
    for code, label, a, b, printed_diff in [
        ("DELINQUENT", "지급 전 2~4월 연체 경험 가구", -92.9, -49.3, 44.6),
        ("NOTDELINQUENT", "지급 전 2~4월 연체 무경험 가구", -20.4, 13.7, 34.1),
    ]:
        caveat = [] if code != "DELINQUENT" else ["원문 표1과 본문은 차이44.6을 인쇄하지만 (-49.3)-(-92.9)=43.6이다. 원문 내부 산술 불일치; 임의 수정하지 않으며 차이44.6을 정답으로 채점하지 않는다."]
        entries.append(entry("P013-PDF-" + code, label + "의 전후 평균 전년동기 소비 증가액",
                             emp({"pre": a, "post": b, "difference_as_printed": printed_diff}, "만원/가구", 7, "표 1",
                                 population="신한 지원금 수령 총203.8만 가구; 연체 경험26.3만. 2019.1~2020.1 연체 없는 가구로 한정",
                                 period={"pre": "2020년3~4월 평균 YoY", "post": "2020년5~8월 평균 YoY"},
                                 method="가구원의 카드/대출 1일 이상 연체 경험에 따른 기술적 소비 비교",
                                 denominator="가구당 전 카드소비", source_consistency_notes=caveat),
                             sim(feasibility="needs_new_measurement_and_household_panel", missing=["카드/대출 연체 상태", "가구원 결합", "2019·2020 월별 카드소비", *caveat],
                                 formula="원문 연체 정의별 pre/post YoY 소비 평균"), kind="heterogeneous_descriptive_outcome"))
    result = base("P013", 8, entries)
    result["source_limitations"] = ["기존 EM-2의11.1%p·EM-3의7.3%는 별도KDI보도자료 수치이고 이PDF에 없는 숫자이다.",
                                    "본 원문은 소득 변화 자료가 없어서 소득분위별 MPC 비교 불가라고 명시(p6). 소득별 소비 증가액을 MPC로 바꾸지 않는다.",
                                    "성별·연령별 정책 결과 수치 없음. 대표성 개선용 실측 분포와 결과 검증지표를 구분한다."]
    return result


def prepare_dist() -> dict:
    pop = "서울 신한카드 내외국인 업종별 매출(63업종); 현금·타 카드 매출 및 시민 거주지 지출 아님"
    sector_data = [
        ("한식", -14.1, -15264, 13, ["한식"], "DS-1"),
        ("기타요식", -10.7, -6152, 13, None, None),
        ("의복/의류", -19.6, -3824, 13, ["의류"], None),
        ("학원", -12.6, -3749, 13, ["학원"], None),
        ("백화점", -5.8, -2954, 13, None, None),
        ("양식", -19.5, -2616, 13, ["양식"], None),
        ("호텔/콘도", -32.5, -2300, 13, None, None),
        ("유흥주점", -49.2, -2248, 13, None, None),
        ("주유소", -7.5, -2015, 13, ["주유소"], None),
        ("면세점", -82.8, -1773, 13, None, None),
        ("자동차판매", 210.6, 6546, 13, None, None),
        ("약국", 20.6, 2819, 13, ["약국"], None),
        ("일반병원", 5.5, 2474, 13, ["병원"], None),
        ("할인점/슈퍼마켓", 2.2, 1929, 13, ["슈퍼마켓"], None),
        ("자동차서비스", 20.8, 1834, 13, ["차량정비"], None),
        ("정육점", 21.0, 1341, 13, ["정육"], None),
        ("기타의료", 20.9, 1291, 13, None, None),
        ("농수산물", 16.8, 1123, 13, ["청과", "수산"], None),
        ("기타음/식료품", 6.1, 646, 13, ["식료품", "음료소매"], None),
        ("편의점", 1.7, 529, 13, ["편의점"], None),
        ("여행사", -60.9, -696, 14, ["여행사"], None),
        ("종합레저시설", -52.4, -79, 14, None, None),
        ("기타유흥업소", -48.6, -1468, 14, None, None),
        ("유아교육", -47.4, -361, 14, None, None),
        ("스포츠시설", -42.9, -883, 14, ["스포츠", "헬스장"], None),
        ("노래방", -40.7, -1262, 14, ["노래방"], None),
        ("영화/공연", -39.1, -827, 14, None, None),
        ("모텔/여관/기타숙박", -32.8, -978, 14, None, None),
        ("실외골프/스키", 95.9, 162, 14, None, None),
        ("가구", 23.3, 501, 14, ["가구"], None),
    ]
    entries = []
    for index, (label, rate, won, page, names, old) in enumerate(sector_data, 1):
        note = "원문 카드사 업종코드 crosswalk 부재. 의미가 유사한 고정 POI subclass만 사용하는 탐색 대리이며 결과를 맞추기 위해 매핑을 확대·축소하지 않는다."
        if label == "할인점/슈퍼마켓":
            note += " 현 매핑은 슈퍼마켓만 포함하므로 원문 할인점 부분을 누락한다."
        if label == "자동차판매":
            note += " 원문은 양재동 본사 귀속 자동차매출을 서울 점포의 직접 매출로 보기 어렵다고 경고한다(p13)."
        entries.append(entry(f"DS-PDF-SECTOR-{index:02d}", label + "의 서울 카드매출 전년동기 변화",
                             emp(rate, "% (전년동기)", page, "표7" if page == 13 else "표8",
                                 printed=page-1, population=pop, period="2020년1~40주 대 2019년 같은 주",
                                 method="업종별 전년동기 신한카드 매출 증감; 코로나·규제·지원금·명절 등 혼합",
                                 denominator=f"2019년 같은 주 서울 {label} 신한카드 매출", amount_change_100million_won=won),
                             sim({"by_sub": names} if names else {},
                                 feasibility="supported_existing_ledger" if names else "needs_merchant_code_crosswalk",
                                 missing=[] if names else ["카드사 업종과 POI 업종의 정확한 crosswalk 또는 더 세부적인 거래 분류"], note=note),
                             existing=old, new=old is None))
    regional = [
        ("마포구", "서교동", -19.0, -2241), ("서대문구", "신촌동", -27.6, -2195),
        ("중구", "명동", -26.2, -2075), ("강남구", "삼성1동", -16.8, -1876),
        ("강남구", "역삼1동", -8.0, -1786), ("송파구", "잠실3동", -28.7, -1605),
        ("용산구", "한강로동", -20.2, -1566), ("중구", "소공동", -17.0, -1533),
        ("종로구", "종로1.2.3.4가동", -11.7, -1415), ("구로구", "구로5동", -34.2, -1300),
        ("중구", "광희동", -31.2, -1281), ("강남구", "대치4동", -16.8, -1189),
        ("금천구", "가산동", -12.5, -1047), ("강서구", "방화2동", -25.2, -1024),
        ("용산구", "이태원1동", -40.7, -952), ("서초구", "양재2동", 78.7, 4623),
        ("영등포구", "여의동", 7.2, 974), ("강서구", "발산1동", 22.5, 617),
        ("노원구", "월계3동", 34.6, 469), ("강남구", "도곡2동", 15.9, 415),
        ("강동구", "고덕2동", 154.4, 297), ("은평구", "진관동", 9.9, 256),
        ("동대문구", "용신동", 6.7, 239), ("송파구", "가락1동", 9.2, 230),
        ("양천구", "목2동", 19.4, 167), ("양천구", "신월3동", 36.2, 160),
        ("강서구", "화곡본동", 14.2, 159), ("영등포구", "당산2동", 5.4, 154),
        ("송파구", "위례동", 24.2, 144), ("강서구", "가양1동", 2.5, 143),
    ]
    for index, (district, dong, rate, amount) in enumerate(regional, 1):
        entries.append(entry(f"DS-PDF-DONG-{index:02d}", f"{district} {dong} 점포 매출 전년동기 변화",
                             emp(rate, "% (전년동기)", 15, "표9 매출액 증감 상위15개 행정동", printed=14,
                                 population=f"서울 {district} {dong} 소재 신한카드 가맹점, 거주민 소비 아님",
                                 period="2020년1~40주 대 2019년 같은 주", method="점포 소재 행정동별 매출 합계의 전년동기 변화",
                                 denominator="해당 행정동의2019년 동기간 신한카드 매출", amount_change_100million_won=amount),
                             sim({"poi_merchant_district": district, "poi_merchant_dong": dong},
                                 feasibility="supported_existing_receipts_with_source_join",
                                 missing=["2020년 행정동 경계·명칭 crosswalk: 현 POI 자료는2026년3월"],
                                 note="시민 거주동으로 대신 집계하지 않는다. 영수증 POI를 보존된2026 업소자료 소재 행정동에 결합한 탐색 대리이다.")))
    reg_pop = "서울397행정동 × 2019·2020 80주 balanced panel(31,760관측), 동·월 고정효과; 업종별 신한카드 매출"
    for code, label, value, page, mapping, unit, denominator in [
        ("FOOD-DISTANCING", "음식점 매출의 2단계 이상 거리두기 추가 효과", -0.1401, 17, {"by_l1": ["식사", "카페", "디저트"]}, "log point", "음식점 매출 로그; 음식점은 한식뿐 아니라 제과·커피·간이음식 포함"),
        ("RETAIL-DISTANCING", "소매점 매출의 2단계 이상 거리두기 추가 효과", 0.05395, 19, {"by_l1": ["마트", "편의점"]}, "log point", "소매 매출 로그; 원문 백화점·대형마트·슈퍼마켓·편의점의 혼합"),
        ("FOOD-MOBILITY", "음식점 매출의 유동인구 탄력성", 0.34081, 17, {}, "탄력성 (log/log)", "첨두 유동인구1% 변화에 대한 음식점 매출% 반응"),
        ("RETAIL-MOBILITY", "소매점 매출의 유동인구 탄력성", 0.29599, 19, {}, "탄력성 (log/log)", "첨두 유동인구1% 변화에 대한 소매 매출% 반응"),
        ("FOOD-YEAR", "음식점 2020년 더미 회귀계수", -0.11685, 17, {}, "log point", "2019년 대비2020년 효과, 다른 설명변수 통제"),
        ("RETAIL-YEAR", "소매점 2020년 더미 회귀계수", 0.04199, 19, {}, "log point", "2019년 대비2020년 효과, 다른 설명변수 통제"),
        ("FOOD-GRANT", "음식점 국가재난지원금 기간 더미 계수", 0.08049, 17, {}, "log point", "지원금 지급기간 여부, 유동인구·거리두기·시간 고정효과 통제"),
        ("RETAIL-GRANT", "소매 국가재난지원금 기간 더미 계수", 0.01365, 19, {}, "log point", "지원금 지급기간 여부, 유동인구·거리두기·시간 고정효과 통제"),
    ]:
        available = bool(mapping)
        entries.append(entry("DS-PDF-REG-" + code, label,
                             emp(value, unit, page, "표10" if page == 17 else "표12", printed=page-1,
                                 population=reg_pop, period="2019~2020주간 패널(각40주)",
                                 method="로그 매출 종속변수, 행정동·월 고정효과 패널 회귀", denominator=denominator),
                             sim(mapping, feasibility="supported_existing_ledger" if available else "needs_panel_and_new_measurement",
                                 missing=[] if available else ["2019·2020 동일 업종 주간 패널", "동별첨두 생활인구", "각 주 거리두기·지급기간·명절과 월 변수"],
                                 formula="ln(sum ON purchases / sum OFF purchases)" if available else "원문 설명변수와 동일한 고정효과 패널회귀 계수",
                                 note="음식점/소매 장소 대리이며 log point를 원문 설명의14%/5.4%와 중복 독립 지표로 세지 않는다."),
                             kind="policy_effect" if available or "GRANT" in code else "regression_association"))
    for code, label, value, page in [
        ("ALL", "서울 상권 첨두 유동인구", -7.6, 9),
        ("TOURIST", "관광특구 첨두 유동인구", -25.5, 10),
        ("DEVELOPED", "발달상권 첨두 유동인구", -13.0, 10),
        ("ALLEY", "골목상권 첨두 유동인구", -1.5, 10),
        ("MARKET", "전통시장 첨두 유동인구", -7.4, 10),
    ]:
        entries.append(entry("DS-PDF-MOBILITY-" + code, label + " 전년 대비 변화",
                             emp(value, "% (전년대비)", page, "본문 및 그림1/3", printed=page-1,
                                 population="서울 상권의11~14시·18~24시 생활인구, 면적 중첩 비례배분; 방문 구매건수 아님",
                                 period="2020년 대2019년 비교(상권유형별 본문 평균)",
                                 method="생활인구를 상권면적에 배분한 평균 첨두 유동인구 변화", denominator="2019년 동일 상권의 첨두 생활인구"),
                             sim(feasibility="needs_new_measurement_and_population_scaling", missing=["비구매자를 포함한 시간대별 체류 인원/이동경로", "상권별경계 중첩면적 배분", "2019 동일 시간대 기초 생활인구", "대표 표본 및 외삽가중치"],
                                 formula="각 상권의 첨두시간 생활인구 합/평균의 동일년 전후 변화"), kind="mobility_outcome"))
    entries.append(entry("DS-PDF-ALL-SALES", "서울 점포 전체40주 카드매출 변화",
                         emp(-6.2, "% (전년동기)", 11, "매출액 변화 첫 본문", printed=10,
                             population=pop, period="2020년1~40주 대2019년 같은주", method="전체업종 신한 매출합계 전년동기비",
                             denominator="2019년1~40주 서울점포 신한매출", amount_change_100million_won=-48162),
                         sim({"ledger": "offline_spent"}, note="현40명3일 오프라인 소비합계 대리; 서울 전체 매출로 확대하지 않는다.")))
    closure = [("FOOD", "음식점", 0.21, 0.19), ("RETAIL", "소매점", 0.15, 0.13)]
    for code, label, previous, current in closure:
        entries.append(entry("DS-PDF-CLOSURE-" + code, label + " 평균 폐업률",
                             emp({"2019": previous, "2020": current}, "%", 16, "본문 및 그림6; 폐업률=폐업/(영업+폐업)", printed=15,
                                 population="지방행정인허가DB의서울음식점/대규모점포·담배소매업 사업자", period="2019·2020 연평균",
                                 method="인허가DB상 폐업/(영업+폐업), 미신고·영업자 지위승계 한계", denominator="영업+폐업 사업자 수"),
                             sim(feasibility="needs_business_supply_model", missing=["사업자 영업·폐업 상태", "매출·비용·임대료·신고/지위승계 과정", "연간 추적"],
                                 formula="폐업 사업자수 / (영업+폐업 사업자수)"), kind="business_supply_outcome"))
    result = base("DISTANCING_2020", 22, entries)
    result["source_limitations"] = ["현DIST동결기간2020-11-24~26은원문매출결과2020년1~40주범위밖이다.",
                                    "표7과표8에반복된업종은한번만목록화했고금액·증감률을독립지표둘로세지않는다.",
                                    "표11·13의합성 요약(A+거리두기 등)은같은회귀계수합으로중복독립지표가아니다.",
                                    "R²·표준오차·t값은추정품질정보이며정책행동결과검증지표로세지않는다.",
                                    "원문 성별·연령·소득별 결과 수치 없음. 표본 분포 개선 요구와 결과 benchmark를 구분한다."]
    return result


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def pair_matrix(rows_by_arm: dict[str, list[dict]], days: list[str], n: int) -> list[str]:
    roster = None
    for arm, rows in rows_by_arm.items():
        keys = {(r["aid"], r["day"]) for r in rows}
        aids = sorted({r["aid"] for r in rows})
        if len(rows) != n * len(days) or len(keys) != len(rows) or keys != {(a, d) for a in aids for d in days}:
            raise ValueError(f"{arm} citizen-day matrix incomplete/duplicated")
        if roster is not None and roster != aids:
            raise ValueError("paired rosters differ")
        roster = aids
    return roster or []


def boot(on: dict[str, int], off: dict[str, int], *, log=False) -> dict:
    aids = sorted(on)
    a, b = sum(on.values()), sum(off.values())
    def calc(x, y):
        if y <= 0 or (log and x <= 0):
            return None
        return math.log(x/y) if log else 100*(x-y)/y
    value = calc(a, b)
    rng = random.Random(20260928)
    vals = []
    for _ in range(2000):
        sample = [rng.choice(aids) for _ in aids]
        v = calc(sum(on[k] for k in sample), sum(off[k] for k in sample))
        if v is not None:
            vals.append(v)
    vals.sort()
    ci = [vals[int((len(vals)-1)*q)] for q in (0.025, 0.975)] if vals else None
    return {"value": value, "unit": "log point" if log else "% (같은 기간 ON/OFF)",
            "on_won": a, "off_won": b, "difference_won": a-b,
            "citizen_bootstrap_95_interval": ci, "bootstrap_valid_draws": len(vals),
            "reason": "OFF분모0:비율정의불가" if b == 0 else ("ON분자0:로그정의불가" if log and a == 0 else None)}


def load_p013() -> tuple[dict[str, list[dict]], list[dict]]:
    result = {}
    sources = [evidence(P013_ARCHIVE)]
    manifests = {}
    with tarfile.open(P013_ARCHIVE) as tf:
        for arm in ("on", "off"):
            rows = []
            ledger = ROOT / f"output/p013_v53_pilot_20260927/{arm}.ledger.jsonl"
            manifest = ledger.with_name(ledger.name + ".manifest.json")
            m = json.loads(manifest.read_text(encoding="utf-8"))
            if m["output_sha256"] != sha(ledger) or m["prompt_variant"] != "v53":
                raise ValueError("P013 frozen ledger SHA/prompt gate")
            manifests[arm] = m
            daily = {(r["aid"], r["day"]): r for r in read_jsonl(ledger)}
            sources += [evidence(ledger), evidence(manifest)]
            event_ids = set()
            for day in P013_DAYS:
                name = f"p013_v53_pilot_20260927/{arm}/metrics/day_{day}.jsonl"
                raw = tf.extractfile(name).read()
                records = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x]
                for r in records:
                    if r.get("status") != "ok" or r.get("experience_day") != day:
                        raise ValueError("P013 noncanonical status/date")
                    receipts = r["execution_receipts"]
                    amount = 0
                    for receipt in receipts:
                        if receipt.get("kind") == "purchase_receipt" and receipt.get("amount", 0) > 0:
                            if receipt["event_id"] in event_ids:
                                raise ValueError("P013 duplicate positive purchase event_id")
                            event_ids.add(receipt["event_id"])
                            amount += receipt["amount"]
                    if amount != daily[(r["aid"], day)]["offline_spent"]:
                        raise ValueError("P013 canonical receipt/daily ledger amount differs")
                    rows.append({"aid": r["aid"], "day": day, "receipts": r["execution_receipts"]})
            result[arm] = rows
    for field in ("roster_sha256", "execution_fingerprint", "baseline_income_map_sha256",
                  "prompt_variant", "system_prompt_sha256", "stage2_system_prompt_sha256"):
        if manifests["on"][field] != manifests["off"][field]:
            raise ValueError(f"P013 frozen paired {field} differs")
    pair_matrix(result, P013_DAYS, 80)
    return result, sources


def source_join(receipt_sets: dict[str, list[dict]]) -> tuple[dict, list[dict], dict]:
    projection_path = OUT / "poi_projection.json"
    if projection_path.exists():
        projection = json.loads(projection_path.read_text(encoding="utf-8"))
        if (projection.get("schema") != "pdf_benchmark_poi_projection_v1"
                or projection.get("read_only") is not True
                or projection.get("graph_arm") != "P014_OFF_completed"
                or projection.get("policy_nodes") != []
                or len(projection["pois"]) != projection["poi_count"]):
            raise ValueError("native POI projection schema/source state gate")
        ids = {r["poi_id"] for rows in receipt_sets.values() for row in rows for r in row["receipts"]
               if r.get("kind") == "purchase_receipt" and r.get("amount", 0) > 0}
        all_pois = {id_: {"sub": r.get("sub"), "l1": r.get("l1"),
                          "district": r.get("district_name"), "dong": r.get("dong_name")}
                    for id_, r in projection["pois"].items()}
        bad_tax = sorted(id_ for id_ in ids
                         if id_ in all_pois and not (all_pois[id_]["sub"] and all_pois[id_]["l1"]))
        missing = sorted(ids - all_pois.keys())
        sub_counts = {}
        l1_counts = {}
        for meta in all_pois.values():
            if meta["sub"]:
                sub_counts[meta["sub"]] = sub_counts.get(meta["sub"], 0) + 1
            if meta["l1"]:
                l1_counts[meta["l1"]] = l1_counts.get(meta["l1"], 0) + 1
        # All selected current DIST transactions must reproduce the preserved
        # native category ledger exactly. This prevents quietly using a later
        # different POI taxonomy as if it were the completed arm's taxonomy.
        for arm, rows in receipt_sets.items():
            for row in rows:
                if "by_sub" not in row:
                    continue
                sums_sub = {}
                sums_l1 = {}
                for receipt in row["receipts"]:
                    if receipt.get("kind") != "purchase_receipt" or receipt.get("amount", 0) <= 0:
                        continue
                    meta = all_pois.get(receipt["poi_id"])
                    if not meta or not (meta["sub"] and meta["l1"]):
                        raise ValueError("DIST native projection selected taxonomy unresolved")
                    sums_sub[meta["sub"]] = sums_sub.get(meta["sub"], 0) + receipt["amount"]
                    sums_l1[meta["l1"]] = sums_l1.get(meta["l1"], 0) + receipt["amount"]
                if sums_sub != row["by_sub"] or sums_l1 != row["by_l1"]:
                    raise ValueError("DIST current POI projection differs from preserved per-citizen taxonomy")
        return all_pois, [evidence(projection_path)], {
            "source": "existing_native_P014_OFF_POI_projection",
            "selected_positive_purchase_poi_ids": len(ids), "matched_poi_ids": len(ids)-len(missing),
            "unmatched_poi_ids": missing, "ambiguous_selected_taxonomy_poi_ids": bad_tax,
            "all_graph_pois": len(all_pois), "source_year": "native model POI taxonomy, historically2020 unverified",
            "historical_2020_crosswalk": False, "catalog_poi_count_by_sub": sub_counts,
            "catalog_poi_count_by_l1": l1_counts,
            "per_citizen_DIST_receipt_projection_matches_frozen_sector_ledger": policy_gate(receipt_sets),
            "note": "사후읽기전용현모델POI projection. DIST양팔per-citizen동결원장의subcategory/parent금액과완전대조. 원문2020카드가맹점census아님."}
    if sha(SOURCE_CSV) != CSV_SHA:
        raise ValueError("recovered source CSV SHA changed")
    ids = {r["poi_id"] for rows in receipt_sets.values() for row in rows for r in row["receipts"]
           if r.get("kind") == "purchase_receipt" and r.get("amount", 0) > 0}
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    out = {}
    count = 0
    with SOURCE_CSV.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            count += 1
            id_ = "C_" + row["상가업소번호"]
            if id_ in ids:
                if id_ in out:
                    raise ValueError("source has duplicated selected POI")
                m = mapping.get(row["상권업종소분류코드"], {})
                out[id_] = {"sub": m.get("sub"), "l1": m.get("cat"),
                            "district": row["시군구명"], "dong": row["행정동명"]}
    missing = sorted(ids - out.keys())
    return out, [evidence(SOURCE_CSV), evidence(MAPPING)], {
        "selected_positive_purchase_poi_ids": len(ids), "matched_poi_ids": len(out),
        "unmatched_poi_ids": missing, "source_rows_scanned": count,
        "source_year": "2026-03", "historical_2020_crosswalk": False,
        "note": "원본복구 CSV는 2026년 POI의 업종/소재동 식별용이며2020 실측표본 복원 데이터가 아니다."}


def policy_gate(receipt_sets: dict[str, list[dict]]) -> bool | None:
    return True if all("by_sub" in row for rows in receipt_sets.values() for row in rows) else None


def dong_name(value: str) -> str:
    """Lexical official name variant only, never a spatial nearest-neighbor."""
    return re.sub(r"제(?=\d+동$)", "", value.replace("·", ".").replace(" ", ""))


def score(policy: str, catalog_path: Path) -> dict:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("schema") != "pdf_benchmark_catalog_v1" or catalog["policy"] != policy:
        raise ValueError("catalog mismatch")
    if sha(ROOT / catalog["source"]["path"]) != catalog["source"]["sha256"]:
        raise ValueError("source PDF changed")
    if policy == "P013":
        rows_by_arm, sources = load_p013()
        days, n = P013_DAYS, 80
    else:
        sys.path.insert(0, str(ROOT))
        from scripts.report.score_multi_policy_proxies import read_pair
        gated_on, gated_off, _, gated_days, gated_sources = read_pair(
            DIST_BASE / "on/on/sector.ledger.jsonl",
            DIST_BASE / "off/off/sector.ledger.jsonl", policy="DISTANCING_2020",
            effect_start=DIST_DAYS[0], effect_end=DIST_DAYS[-1])
        if gated_days != DIST_DAYS:
            raise ValueError("DIST primary paired gate selected another date window")
        rows_by_arm = {}
        sources = [evidence(p) for p in gated_sources]
        manifests = {}
        days, n = DIST_DAYS, 40
        for arm in ("on", "off"):
            base_ = DIST_BASE / arm / arm
            ledger = base_ / "sector.ledger.jsonl"
            manifest = ledger.with_name(ledger.name + ".manifest.json")
            m = json.loads(manifest.read_text(encoding="utf-8"))
            manifests[arm] = m
            if m["output_sha256"] != sha(ledger) or m["prompt_provenance"]["prompt_variant"] != "v53":
                raise ValueError("DIST frozen ledger SHA/prompt gate")
            rows_by_arm[arm] = read_jsonl(ledger)
            sources.extend([evidence(ledger), evidence(manifest)])
            for row in rows_by_arm[arm]:
                metric = base_ / "metrics" / f"day_{row['day']}.jsonl"
                if "receipts" not in row:
                    row["receipts"] = []
            metrics = {}
            for day in days:
                metric = base_ / "metrics" / f"day_{day}.jsonl"
                if sha(metric) != m["metrics_sha256"][day]:
                    raise ValueError("DIST canonical metric SHA differs from sector manifest")
                sources.append(evidence(metric))
                records = read_jsonl(metric)
                if len(records) != n or len({r["aid"] for r in records}) != n or any(r.get("status") != "ok" for r in records):
                    raise ValueError("DIST canonical metrics incomplete")
                metrics.update({(r["aid"], day): r["execution_receipts"] for r in records})
            for row in rows_by_arm[arm]:
                row["receipts"] = metrics[(row["aid"], row["day"])]
                amount = sum(r.get("amount", 0) for r in row["receipts"]
                             if r.get("kind") == "purchase_receipt" and r.get("amount", 0) > 0)
                if amount != row["offline_spent"]:
                    raise ValueError("DIST canonical receipt/sector ledger amount differs")
        for field in ("baseline_income_map_sha256", "prompt_variant",
                      "system_prompt_sha256", "stage2_system_prompt_sha256", "requested_model_id"):
            if manifests["on"]["prompt_provenance"][field] != manifests["off"]["prompt_provenance"][field]:
                raise ValueError(f"DIST frozen paired {field} differs")
    roster = pair_matrix(rows_by_arm, days, n)
    poi, joined_sources, join_audit = source_join(rows_by_arm)
    sources += joined_sources
    tax = json.loads(MAPPING.read_text(encoding="utf-8"))
    sub_to_l1 = {}
    l1_to_sub = {}
    for meta in tax.values():
        sub_to_l1.setdefault(meta.get("sub"), set()).add(meta.get("cat"))
        l1_to_sub.setdefault(meta.get("cat"), set()).add(meta.get("sub"))
    output = []
    for e in catalog["entries"]:
        if e.get("existing_indicator_id"):
            # Preserve the independently frozen original14 computations.
            # Existing source constituents are references, not fresh outcomes.
            continue
        spec = e["simulation"]
        if not spec["feasibility"].startswith("supported_existing"):
            continue
        mapping = spec["poi_mapping"]
        if mapping.get("ledger") == "eligible_offline_spent / grant_received_cumulative":
            pair_path = ROOT / "output/p013_v53_pilot_20260927/paired_effect.json"
            paired = json.loads(pair_path.read_text(encoding="utf-8"))
            for arm in ("on", "off"):
                ledger = ROOT / f"output/p013_v53_pilot_20260927/{arm}.ledger.jsonl"
                if paired["provenance"]["arms"][arm]["ledger_sha256"] != sha(ledger):
                    raise ValueError("P013 paired grant result uses another ledger")
            sources.append(evidence(pair_path))
            output.append({"id": e["id"], "label": e["label"],
                           "simulation": 100 * paired["eligible_offline_effect_per_grant_won"],
                           "simulation_unit": "% (증분 적격 오프라인 지출/모델 지급액)",
                           "formula": spec["proposed_formula"],
                           "raw_components": {"on_won": paired["eligible_offline_spend_on_won"],
                                              "off_won": paired["eligible_offline_spend_off_won"],
                                              "difference_won": paired["eligible_offline_difference_won"],
                                              "grant_issued_won": paired["grant_issued_won"]},
                           "sample_citizens": n, "simulation_days": days,
                           "scope_note": COMMON_SCOPE + " " + spec["scope_note"],
                           "quality_notes": ["이미 보존된 동일 v53 양팔 eligible offline/grant 산식을 새 원문 범위26.2~36.1% 옆에 사후 표시한다.",
                                             "정책의 전체중앙+지방 지원 예산 대비 전국 증분카드매출이 아니다. 표본22.4백만원 지급 후3일만 관측했다."],
                           "citizen_bootstrap_95_interval": [100*v for v in paired["eligible_offline_citizen_bootstrap_95_interval"]],
                           "direct_gap_allowed": False})
            continue
        amounts = {arm: dict.fromkeys(roster, 0) for arm in rows_by_arm}
        unknown_won = {arm: 0 for arm in rows_by_arm}
        regional = "poi_merchant_district" in mapping
        native_dist = policy == "DISTANCING_2020" and not regional
        native_projection = join_audit.get("source") == "existing_native_P014_OFF_POI_projection"
        classification_basis = ("frozen_sector_ledger_by_sub/by_l1" if native_dist else
                                "canonical_receipt_native_category" if mapping.get("by_l1") else
                                "native_model_POI_projection" if native_projection else "recovered2026_poi_source_join")
        for arm, rows in rows_by_arm.items():
            for row in rows:
                if native_dist:
                    if mapping.get("ledger") == "offline_spent":
                        amount = row["offline_spent"]
                    elif mapping.get("by_l1"):
                        amount = sum(row["by_l1"].get(k, 0) for k in mapping["by_l1"])
                    else:
                        amount = sum(row["by_sub"].get(k, 0) for k in mapping.get("by_sub", []))
                    amounts[arm][row["aid"]] += amount
                    continue
                for r in row["receipts"]:
                    if r.get("kind") != "purchase_receipt" or r.get("amount", 0) <= 0:
                        continue
                    meta = poi.get(r["poi_id"])
                    # Native receipt L1 is complete and does not require the
                    # recovered catalog; it is not a guessed finer subclass.
                    if mapping.get("by_l1"):
                        if r.get("category") in mapping["by_l1"]:
                            amounts[arm][row["aid"]] += r["amount"]
                        continue
                    if meta is None or not (meta.get("sub") and meta.get("l1")):
                        targets = set(mapping.get("by_sub", []))
                        possible = set().union(*(sub_to_l1.get(k, set()) for k in targets)) if targets else set()
                        native = r.get("category")
                        # If a native L1 has exactly one subclass, the recorded
                        # category proves that subclass without a source join.
                        if not regional and l1_to_sub.get(native, set()) <= targets and l1_to_sub.get(native):
                            amounts[arm][row["aid"]] += r["amount"]
                        elif regional or native in possible or native is None:
                            unknown_won[arm] += r["amount"]
                        continue
                    qualifies = False
                    if mapping.get("ledger") == "offline_spent":
                        qualifies = True
                    elif "poi_merchant_district" in mapping:
                        if not meta.get("district") or not meta.get("dong"):
                            unknown_won[arm] += r["amount"]
                            continue
                        qualifies = (meta["district"] == mapping["poi_merchant_district"]
                                     and dong_name(meta["dong"]) == dong_name(mapping["poi_merchant_dong"]))
                    else:
                        qualifies = (meta["sub"] in mapping.get("by_sub", []) or meta["l1"] in mapping.get("by_l1", []))
                    if qualifies:
                        amounts[arm][row["aid"]] += r["amount"]
        log = spec["proposed_formula"].startswith("ln(")
        value = boot(amounts["on"], amounts["off"], log=log)
        if any(unknown_won.values()):
            value["value"] = None
            value["citizen_bootstrap_95_interval"] = None
            value["reason"] = "일부양수구매POI미결합:정확한업종/지역분모보장불가"
        catalog_support = None
        if native_projection and regional:
            catalog_support = sum(meta.get("district") == mapping["poi_merchant_district"]
                                  and meta.get("dong") is not None
                                  and dong_name(meta["dong"]) == dong_name(mapping["poi_merchant_dong"])
                                  for meta in poi.values())
        elif native_projection:
            if mapping.get("by_l1"):
                catalog_support = sum(join_audit["catalog_poi_count_by_l1"].get(k, 0) for k in mapping["by_l1"])
            elif mapping.get("by_sub"):
                catalog_support = sum(join_audit["catalog_poi_count_by_sub"].get(k, 0) for k in mapping["by_sub"])
        value["catalog_poi_support_count"] = catalog_support
        support_note = ""
        if value["value"] is None and catalog_support is not None:
            if catalog_support == 0 and regional:
                support_note = "현 모델 POI 투영에는 원문과 같은 명칭의 행정동이 없다. 동 통합·분리 또는 자료 연도 차이일 수 있으며, 표본 확대만으로 해결되지 않는다. 먼저 공식 과거 행정동 대응표를 검증해야 한다."
            elif catalog_support == 0:
                support_note = "정의한 세부업종이 현재 모델 POI 목록에 없다. 시민 수를 늘리는 것만으로 누락 업종이 복구되지는 않으며, 먼저 업종 분류와 장소 지원을 보완해야 한다."
            else:
                support_note = (f"현재 모델 목록에는 해당 범위 POI가{catalog_support}개 있지만 동결 소규모 표본의OFF구매 분모가0이다. "
                                "실측이 없다는 뜻도, 효과가0이라는 증거도 아니다. 실측 분포를 반영한 새 동결 표본과 더 긴 기간으로 구매 노출을 늘려볼 수 있다. "
                                "동시에 원문 연도·업종·기준 통계량을 맞춰야 하며, 대규모 실험만으로 외부 검증이 보장되지는 않는다.")
        output.append({"id": e["id"], "label": e["label"],
                       "simulation": value["value"], "simulation_unit": value["unit"],
                       "formula": spec["proposed_formula"],
                       "raw_components": {k: v for k, v in value.items() if k not in ("value", "unit")},
                       "sample_citizens": n, "simulation_days": days,
                       "classification_basis": classification_basis,
                       "dong_name_matching_rule": "동명 공백·중점 표기 정규화 및제N동=N동;경계재배치/근접지역합산없음" if regional else None,
                       "scope_note": COMMON_SCOPE + " " + spec.get("scope_note", ""),
                       "quality_notes": ["사후 탐색 분해; 사전등록된14지표 점수·최적프롬프트 판정을 변경하지 않는다.",
                                         ("업종은 보존된 엔진 POI taxonomy를 원장에서 직접 집계; 사후CSV 미결합금액으로 완전 원장을 대체하지 않는다." if native_dist else
                                          "L1은canonical영수증을 직접 집계; subclass/소재동은동일모델POI읽기전용projection을사용한다." if native_projection else
                                          "L1은canonical영수증을 직접 집계; 세부업종/소재동은2026년3월 상가자료join의 누락 관문을 적용한다."),
                                         "DIST 기간은 원문40주 범위밖인2020-11-24~26이다." if policy == "DISTANCING_2020" else "P013 정책발효3일만집계: 월·주 전체 및2019년 기준없음.",
                                         support_note,
                                         value["reason"] or "OFF분모가 양수여서 수학적으로비율정의가능; 외적타당성확보는별개."],
                       "unmatched_positive_purchase_won": unknown_won,
                       "citizen_bootstrap_95_interval": value["citizen_bootstrap_95_interval"],
                       "direct_gap_allowed": False})
    return {"schema": "pdf_benchmark_simulation_v1", "policy": policy,
            "timing": "post-result exploratory", "catalog_path": catalog_path.relative_to(ROOT).as_posix(),
            "catalog_sha256": sha(catalog_path), "source_evidence": sources,
            "generator": evidence(Path(__file__)),
            "join_audit": join_audit, "rows": output,
            "unsupported_catalog_ids": [e["id"] for e in catalog["entries"] if e["id"] not in {r["id"] for r in output}],
            "no_model_or_graph_calls": True, "sampling_caveat": COMMON_SCOPE}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "score"))
    args = parser.parse_args()
    if args.mode == "prepare":
        for filename, data in (("p013_catalog.json", prepare_p013()), ("distancing_catalog.json", prepare_dist())):
            path = CAT / filename
            write(path, data, refuse_existing=True)
            print(f"{path.relative_to(ROOT)}: {len(data['entries'])} entries SHA256={sha(path)}")
    else:
        for policy, name in (("P013", "p013"), ("DISTANCING_2020", "distancing")):
            catalog_path = CAT / f"{name}_catalog.json"
            result = score(policy, catalog_path)
            path = OUT / f"{name}_simulation.json"
            write(path, result)
            print(f"{path.relative_to(ROOT)}: {len(result['rows'])} rows SHA256={sha(path)}")


if __name__ == "__main__":
    main()

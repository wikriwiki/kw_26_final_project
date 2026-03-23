"""프로젝트 경로 및 설정"""
from pathlib import Path

# ── 경로 ──
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════
# 데이터 모드: True → 합성 데이터, False → 실제 데이터
# ══════════════════════════════════════
USE_SYNTHETIC = True

_DATA = SYNTHETIC_DIR if USE_SYNTHETIC else DATA_DIR

# ── 데이터 파일 매핑 ──
DATA_FILES = {
    # B079 카드소비 (행정동별)
    "b079_gender_age": _DATA / "7.서울시 내국인 성별 연령대별(행정동별).csv",
    "b079_inflow": _DATA / "8.서울시 내국인의 개인카드 기준 유입지별(행정동별).csv",
    "b079_time": _DATA / "2.서울시민의 일별 시간대별(행정동).csv",

    # B063 카드소비패턴 (블록/집계구)
    "b063_block_gender_age": _DATA / "블록별 성별연령대별 카드소비패턴.csv",
    "b063_census_gender_age": _DATA / "내국인(집계구) 성별연령대별.csv",
    "b063_census_inflow": _DATA / "내국인(집계구) 유입지별.csv",
    "b063_block_time": _DATA / "내국인(블록) 일자별시간대별.csv",

    # 업종 코드 매핑
    "industry_code_ss": _DATA / "카드소비 업종코드.csv",
    "industry_code_sb": _DATA / "신한카드 내국인 63업종 코드.csv",

    # B009 KT 유동인구
    "kt_time_age": _DATA / "KT 월별 시간대별 성연령대별 유동인구.csv",
    "kt_residence": _DATA / "KT 월별 성연령대별 거주지별 유동인구.csv",

    # B069 상권발달지수
    "district_index": _DATA / "행정동별 상권발달 개별지수.csv",

    # B078 생활이동
    "purpose_od": DATA_DIR / "PURPOSE_250M_202403.csv",

    # B068 임대시세
    "rent_monthly": _DATA / "월세임대 예측시세.csv",
    "rent_deposit": _DATA / "전세임대 예측시세.csv",
}

# ── 인코딩 ──
ENCODING = "cp949"

# ── 시뮬레이션 기본값 ──
DEFAULT_AGENT_COUNT = 300
DEFAULT_ROUNDS = 24
ROUND_UNIT = "week"

"""
scripts/persona/_common.py
==========================
페르소나 강화 두 방식(rank-coupling / conditional-graft) 공통 모듈.

책임:
  1. 입력 로딩 — 우리 BDC 통계(agent_profiles/allocation/decile_boundaries) + NVIDIA 서울 풀
  2. 정규화 — NVIDIA sex/age/district → 우리 키(F/M, 연령대, 자치구)
  3. SES_proxy — NVIDIA education/occupation/housing → 사회경제지위 점수 [0,1]
  4. 통계 샘플링 — 셀(adm8,성별,연령) 프로필 → 소비분위·금액·업종·행태 정량 생성
  5. 출력 스키마 — 기존 agents_final.json 포맷 + NVIDIA 정성/확보 필드

LLM 호출 없음. 모두 결정적 통계 샘플링(seed 고정 가능).

설계 출처: 페르소나 강화 연구 보고서 v2 (조건부 입히기 / 줄세우기 결합).
"""
from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATS_DIR = PROJECT_ROOT / "output" / "stats"
PERSONA_DIR = PROJECT_ROOT / "data" / "personas"


# ---------------------------------------------------------------------------
# 1. 입력 로딩
# ---------------------------------------------------------------------------
def load_bdc_stats(stats_dir: Path | None = None) -> dict:
    """우리 BDC 통계 로드. agent_profiles / allocation / decile_boundaries."""
    d = stats_dir or STATS_DIR
    return {
        "profiles": json.loads((d / "agent_profiles.json").read_text(encoding="utf-8")),
        "allocation": json.loads((d / "agent_allocation.json").read_text(encoding="utf-8")),
        "deciles": json.loads((d / "decile_boundaries.json").read_text(encoding="utf-8")),
    }


def _read_persona_records(path: Path) -> list[dict]:
    """`.json`(배열) 또는 `.jsonl`(라인당 1레코드) 모두 지원."""
    if path.suffix == ".jsonl":
        out: list[dict] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_nvidia_path(persona_dir: Path | None = None) -> Path:
    """NVIDIA 입력 경로 결정.

    우선순위:
      1. env `NVIDIA_PERSONA_PATH` (실데이터 경로 직접 지정)
      2. nvidia_seoul_full.jsonl  (prepare_nvidia.py --jsonl 산출)
      3. nvidia_seoul_full.json   (prepare_nvidia.py 산출, 실데이터)
      4. nvidia_seoul_sample.json (레포 동봉 120건 fixture, fallback)
    """
    env = os.getenv("NVIDIA_PERSONA_PATH")
    if env:
        return Path(env)
    d = persona_dir or PERSONA_DIR
    for name in ("nvidia_seoul_full.jsonl", "nvidia_seoul_full.json",
                 "nvidia_seoul_sample.json"):
        p = d / name
        if p.exists():
            return p
    return d / "nvidia_seoul_sample.json"


def load_nvidia_seoul(persona_dir: Path | None = None) -> list[dict]:
    """NVIDIA 서울 풀 로드. 실데이터(full) 우선, 없으면 sample fallback.

    `NVIDIA_PERSONA_PATH` env 로 경로 직접 지정 가능. `.json`/`.jsonl` 모두 지원.
    """
    path = resolve_nvidia_path(persona_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"NVIDIA 페르소나 파일 없음: {path}. "
            f"실데이터는 `python scripts/persona/prepare_nvidia.py` 로 받으세요."
        )
    return _read_persona_records(path)


def write_personas(personas: list[dict], out: Path, jsonl: bool = False) -> Path:
    """페르소나 저장. jsonl=True 면 라인당 1건(대용량 메모리 절약)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        with out.open("w", encoding="utf-8") as f:
            for p in personas:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    else:
        out.write_text(json.dumps(personas, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# 2. 정규화 — NVIDIA → 우리 키
# ---------------------------------------------------------------------------
_SEX_MAP = {"여자": "F", "남자": "M", "여": "F", "남": "M"}


def nvidia_sex(rec: dict) -> str:
    return _SEX_MAP.get((rec.get("sex") or "").strip(), "")


def age_to_group(age: int) -> str:
    """NVIDIA age(int) → 우리 연령대 라벨. 우리 통계는 20대~70대이상."""
    if age < 30:
        return "20대"   # 19도 20대로 흡수 (우리 통계 최소 단위)
    if age < 40:
        return "30대"
    if age < 50:
        return "40대"
    if age < 60:
        return "50대"
    if age < 70:
        return "60대"
    return "70대이상"


def nvidia_gu(rec: dict) -> str:
    """'서울-서초구' → '서초구'."""
    d = (rec.get("district") or "").strip()
    if "-" in d:
        return d.split("-", 1)[1]
    return d


def nvidia_cell(rec: dict) -> tuple[str, str, str]:
    """NVIDIA 레코드 → (자치구, 성별, 연령대) 매칭 키."""
    return (nvidia_gu(rec), nvidia_sex(rec), age_to_group(int(rec.get("age") or 0)))


# ---------------------------------------------------------------------------
# 3. SES_proxy — 사회경제지위 점수 [0,1]
# ---------------------------------------------------------------------------
# 학력 순위 (NVIDIA education_level 7값)
_EDU_RANK = {
    "초등학교": 0.0, "중학교": 0.17, "고등학교": 0.33,
    "2~3년제 전문대학": 0.5, "4년제 대학교": 0.75, "대학원": 1.0,
}
# 주거 순위 (housing_type)
_HOUSING_RANK = {
    "비거주용건물내주택": 0.0, "주택이외의거처": 0.1, "다세대주택": 0.4,
    "연립주택": 0.5, "단독주택": 0.6, "아파트": 0.85, "오피스텔": 0.7,
}
# 직업 skill tier — 키워드 기반 (KSCO 대분류 근사)
_OCC_HIGH = ("의사", "변호사", "교수", "연구", "약사", "회계사", "변리사", "임원",
             "경영", "전문", "엔지니어", "개발자", "건축사", "감정평가", "세무사", "기자")
_OCC_MID = ("사무원", "교사", "간호사", "디자이너", "영업", "관리", "공무원",
            "기사", "상담", "은행", "보험", "회계 사무")
_OCC_LOW = ("단순", "노무", "배달", "청소", "경비", "판매원", "서비스", "알바",
            "아르바이트", "운전", "미화", "주방")


def _occupation_tier(occ: str) -> float:
    o = occ or ""
    if any(k in o for k in _OCC_HIGH):
        return 1.0
    if any(k in o for k in _OCC_LOW):
        return 0.0
    if any(k in o for k in _OCC_MID):
        return 0.5
    return 0.5   # 미상은 중간


def ses_proxy(rec: dict, w_edu: float = 0.35, w_occ: float = 0.4,
              w_house: float = 0.25) -> float:
    """education + occupation + housing → SES 점수 [0,1].

    가중치 합 1.0. 초기 동일 계열, occupation 비중을 약간 높게 (소득 직결).
    값은 절대 의미보다 *셀 내 순위* 용 (rank-coupling).
    """
    edu = _EDU_RANK.get((rec.get("education_level") or "").strip(), 0.4)
    occ = _occupation_tier(rec.get("occupation") or "")
    house = _HOUSING_RANK.get((rec.get("housing_type") or "").strip(), 0.4)
    return w_edu * edu + w_occ * occ + w_house * house


# ---------------------------------------------------------------------------
# 4. 통계 샘플링 — 셀 프로필 → 정량
# ---------------------------------------------------------------------------
def parse_cell_key(key: str) -> tuple[str, str, str]:
    """'11110515_F_20대' → (adm8, 성별, 연령대)."""
    parts = key.split("_")
    return parts[0], parts[1], "_".join(parts[2:])


def decile_amount(deciles: dict, kind: str, level: int, rng: random.Random) -> int:
    """소비분위(1~10) → 실제 일일 소비액(원). decile_boundaries 의 [min,max] 내 샘플.

    1분위는 하한 outlier 무시(상단 60~95%), 10분위는 상한 무시(하단 5~40%),
    중간 분위는 중앙 근처 — generate_agents.py 의 long-tail 규칙 계승.
    """
    key = f"{kind}_spending_level"
    bounds = deciles.get(key, {}).get("boundaries", [])
    b = next((x for x in bounds if x["decile"] == level), None)
    if not b:
        return 10000
    lo, hi = float(b["min"]), float(b["max"])
    if level == 1:
        frac = rng.uniform(0.60, 0.95)
    elif level >= 10:
        frac = rng.uniform(0.05, 0.40)
    else:
        frac = rng.uniform(0.35, 0.65)
    return int(lo + (hi - lo) * frac)


# 우리 industry_ratio 업종명 → 시뮬 12 L1 카테고리 매핑
_IND_TO_L1 = {
    "한식": "식사", "양식": "식사", "일식": "식사", "중식": "식사", "기타요식": "식사",
    "커피전문점": "카페", "제과점": "디저트",
    "편의점": "편의점", "슈퍼마켓 일반형": "마트", "농수산물": "마트", "기타식품": "마트",
    "미용실": "미용", "화장품": "미용",
    "의복/의류": "쇼핑", "패션잡화": "쇼핑", "시계/귀금속": "쇼핑", "문화용품": "쇼핑",
    "생활잡화/수입상품점": "쇼핑", "수제용품점": "쇼핑", "인테리어/건축자재/주방기구": "쇼핑",
    "컴퓨터/소프트웨어": "쇼핑",
    "영화/공연": "여가", "서점": "여가", "독서실": "교육", "학원/학습지": "교육",
    "약국": "건강", "일반병원": "건강", "애완동물": "기타", "화원": "기타",
    "주류판매": "주점", "주차장": "기타", "사무기기/문구용품": "쇼핑",
}


def industry_to_l1_ratio(industry_ratio: dict) -> dict[str, float]:
    """우리 세분 업종비율 → 시뮬 L1 카테고리 비율 (합 1.0 재정규화)."""
    agg: dict[str, float] = {}
    for ind, r in (industry_ratio or {}).items():
        if ind.startswith("ZZ"):
            continue
        l1 = _IND_TO_L1.get(ind)
        if not l1:
            continue
        agg[l1] = agg.get(l1, 0.0) + float(r)
    total = sum(agg.values())
    if total <= 0:
        return {}
    return {k: round(v / total, 4) for k, v in sorted(agg.items(), key=lambda x: -x[1])}


def top_categories(l1_ratio: dict[str, float], k: int = 5) -> dict[str, float]:
    """상위 k L1 카테고리."""
    items = sorted(l1_ratio.items(), key=lambda x: -x[1])[:k]
    return {kk: vv for kk, vv in items}


def spending_tendency_from(level_wd: int, level_we: int) -> str:
    """소비분위 → 절약형/보통/소비형."""
    avg = (level_wd + level_we) / 2
    if avg <= 3.5:
        return "절약형"
    if avg >= 7.5:
        return "소비형"
    return "보통"


# ---------------------------------------------------------------------------
# 5. 출력 스키마 빌더
# ---------------------------------------------------------------------------
# LLM 프롬프트에 넣을 NVIDIA 필드 (행동·소비 직접 영향)
NVIDIA_LLM_FIELDS = [
    "persona", "hobbies_and_interests_list", "cultural_background",
    "marital_status", "housing_type", "family_type", "education_level",
]
# 저장만 (LLM 미입력) — 직업관·목표·6종서사·skills
NVIDIA_RESERVED_FIELDS = [
    "professional_persona", "career_goals_and_ambitions",
    "sports_persona", "arts_persona", "travel_persona",
    "culinary_persona", "family_persona",
    "skills_and_expertise", "skills_and_expertise_list",
    "occupation", "bachelors_field", "military_status", "uuid",
]


def _parse_list_field(v) -> list:
    """NVIDIA *_list 필드가 "['a','b']" 문자열일 수 있어 안전 파싱."""
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.strip().startswith("["):
        try:
            import ast
            return ast.literal_eval(v)
        except Exception:
            return [v]
    return [v] if v else []


@dataclass
class PersonaRecord:
    """두 방식 공통 출력 페르소나. agents_final.json 호환 + NVIDIA 레이어."""
    agent_id: str
    residence: dict
    personal: dict
    workplace: dict
    spending: dict
    behavior: dict
    personality: dict
    nvidia_persona: dict = field(default_factory=dict)     # LLM 입력
    nvidia_reserved: dict = field(default_factory=dict)    # 저장만
    match_meta: dict = field(default_factory=dict)         # 매칭 디버그

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "residence": self.residence,
            "personal": self.personal,
            "workplace": self.workplace,
            "spending": self.spending,
            "behavior": self.behavior,
            "personality": self.personality,
            "nvidia_persona": self.nvidia_persona,
            "nvidia_reserved": self.nvidia_reserved,
            "_match": self.match_meta,
        }


def split_nvidia_fields(rec: dict) -> tuple[dict, dict]:
    """NVIDIA 레코드 → (LLM 입력 dict, 저장-only dict)."""
    llm = {}
    for f in NVIDIA_LLM_FIELDS:
        if f == "hobbies_and_interests_list":
            llm["hobbies"] = _parse_list_field(rec.get(f))
        elif f == "persona":
            llm["summary"] = rec.get(f)
        else:
            llm[f] = rec.get(f)
    reserved = {f: rec.get(f) for f in NVIDIA_RESERVED_FIELDS}
    return llm, reserved


def build_quant_from_cell(
    profile: dict, deciles: dict, rng: random.Random,
    spending_level_override: tuple[int, int] | None = None,
) -> dict:
    """셀 프로필 → 정량 dict (spending/behavior). 두 방식 공통.

    spending_level_override: (wd, we) 직접 지정 (conditional 의 SES 힌트용).
    None 이면 프로필의 셀 대표 분위 사용.
    """
    cons = profile.get("consumption", {})
    mob = profile.get("mobility", {})
    tel = profile.get("telecom", {})

    if spending_level_override:
        lv_wd, lv_we = spending_level_override
    else:
        lv_wd = int(cons.get("weekday_spending_level") or 5)
        lv_we = int(cons.get("weekend_spending_level") or 5)

    daily_wd = decile_amount(deciles, "weekday", lv_wd, rng)
    daily_we = decile_amount(deciles, "weekend", lv_we, rng)
    we_wd_ratio = round(daily_we / daily_wd, 3) if daily_wd else 1.0

    l1 = industry_to_l1_ratio(cons.get("industry_ratio", {}))
    spending = {
        "weekday_spending_level": lv_wd,
        "weekend_spending_level": lv_we,
        "daily_spending_weekday": daily_wd,
        "daily_spending_weekend": daily_we,
        "weekend_weekday_spending_ratio": we_wd_ratio,
        "weekday_top_categories": top_categories(l1),
        "weekend_top_categories": top_categories(l1),
    }
    mob_lv = int(mob.get("mobility_level") or 5)
    # 재택시간: telecom raw(tel_home_*_time) 은 단위 불명이라 직접 변환 불가
    # (예: 955794 → /3600%24 = 1.5h 비현실값). 대신 이동성 분위로 추정:
    # 이동 많을수록(mob_lv↑) 재택 적음. 주말은 평일보다 +1.5h.
    home_wd = round(max(8.0, min(20.0, 16.0 - mob_lv * 0.6 + rng.uniform(-1, 1))), 1)
    home_we = round(max(8.0, min(22.0, home_wd + 1.5 + rng.uniform(-1, 1))), 1)
    behavior = {
        "delivery_days": round(float(tel.get("tel_delivery_days") or 0) + rng.uniform(-3, 3), 1),
        "shopping_days": rng.randint(4, 18),
        "weekday_move_km": round(float(tel.get("tel_wd_move_dist") or 0) / 1000.0, 2),
        "weekend_move_km": round(float(tel.get("tel_we_move_dist") or 0) / 1000.0, 2),
        "home_hours_weekday": home_wd,
        "home_hours_weekend": home_we,
        "mobility_level": mob_lv,
    }
    return {"spending": spending, "behavior": behavior,
            "tendency": spending_tendency_from(lv_wd, lv_we)}

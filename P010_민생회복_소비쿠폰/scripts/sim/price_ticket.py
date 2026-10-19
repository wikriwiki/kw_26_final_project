# -*- coding: utf-8 -*-
"""에이전트가 고른 세부 업종의 실측 결제 1건당 금액 한 줄 (2026-10-07).

EXP_PRICE_MODE
  ''          기존 동작(경험 상수 '동네 평균단가' 와 가격대 배율) — 옛 런 재현용
  ticket      고른 세부 업종의 실측 결제 1건당(이 동네 값 + 서울 하위10%~상위10%)만 보여 준다
  knowledge   금액 숫자를 보여 주지 않는다(모델이 아는 메뉴·물건 가격). 실측은 검증에만 쓴다
두 모드 모두 2단계가 결제마다 menu · unit_price(1인분·1개) · pay_count(내가 계산하는 인분·개수)를 답하고
결제 금액 = unit_price × pay_count 로 정한다. 가격대 배율(경험 상수)과 사후 비례 축소는 쓰지 않는다.
연결되는 실측 업종이 없는 세부 업종은 숫자를 만들지 않는다(None).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

MODE = os.environ.get("EXP_PRICE_MODE", "").strip().lower()
if MODE not in ("", "ticket", "knowledge"):
    raise ValueError(f"EXP_PRICE_MODE 는 '', ticket, knowledge 중 하나: {MODE!r}")
ON = MODE in ("ticket", "knowledge")

_PATH = Path(__file__).resolve().parents[2] / "output" / "stats" / "ticket_price.json"

# 시뮬 세부 업종 → 상권분석 서비스 업종(원자료 이름 그대로). 없으면 숫자를 보이지 않는다.
SUB_TO_SVC = {
    "한식": "한식음식점", "중식": "중식음식점", "일식": "일식음식점", "양식": "양식음식점",
    "분식": "분식전문점", "치킨": "치킨전문점", "패스트푸드": "패스트푸드점",
    "카페": "커피-음료", "베이커리": "제과점", "제과": "제과점",
    "편의점": "편의점", "슈퍼마켓": "슈퍼마켓", "식료품": "슈퍼마켓",
    "의원": "일반의원", "치과": "치과의원", "한의원": "한의원", "약국": "의약품",
    "미용실": "미용실", "피부관리": "피부관리실", "네일": "네일숍", "화장품": "화장품",
    "의류": "일반의류", "신발": "신발", "가방": "가방", "안경": "안경", "서적": "서적", "문구": "문구",
    "가전·통신": "가전제품", "통신": "핸드폰", "가구": "가구", "식물·꽃": "화초", "반려동물": "애완동물",
    "노래방": "노래방", "노래연습장": "노래방", "PC방": "PC방", "당구": "당구장",
    "스포츠": "스포츠클럽", "학원": "일반교습학원", "일반주점": "호프-간이주점", "주점": "호프-간이주점",
    "세탁": "세탁소",
}


@lru_cache(maxsize=1)
def _table() -> dict:
    if not _PATH.is_file():
        raise FileNotFoundError(f"{_PATH} 가 없다 — scripts/prep/build_ticket_price.py 를 먼저 돌린다")
    return json.loads(_PATH.read_text(encoding="utf-8"))


def info(dong_code: str | None, sub: str | None) -> dict | None:
    svc = SUB_TO_SVC.get((sub or "").strip())
    if not svc:
        return None
    t = _table()
    stats = t["svc"].get(svc)
    if not stats:
        return None
    here = (t["dong"].get(str(dong_code or "")) or {}).get(svc)
    return {"svc": svc, "dong_won": here, "p10": stats["p10"], "p50": stats["p50"], "p90": stats["p90"]}


def label(i: dict | None) -> str:
    if not i:
        return ""
    head = (f"이 동네 {i['svc']} 카드 결제 1건당 ~{i['dong_won']:,}원" if i.get("dong_won")
            else f"서울 {i['svc']} 카드 결제 1건당 중앙 ~{i['p50']:,}원")
    return f" | {head} (서울 동별 하위10%~상위10% {i['p10']:,}~{i['p90']:,}원 · 2025 실측 · 여럿이 함께 낸 결제 포함)"


def fallback_won(i: dict | None) -> int | None:
    """모델이 금액을 빠뜨렸을 때만 쓰는 실측값(이 동네 값, 없으면 서울 중앙)."""
    if not i:
        return None
    return int(i.get("dong_won") or i["p50"])

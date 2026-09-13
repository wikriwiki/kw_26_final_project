"""정책 기전 레지스트리.

정책마다 코드를 새로 쓰면 소비행동 프롬프트를 일반화해도 **배관에서 1:1 결합이
되살아난다.** 새 정책은 코드가 아니라 JSON 파라미터로 붙어야 한다.

기전별로 달라지는 것은 셋뿐이다.

  LABEL       에이전트에게 보일 중립 라벨 (정책 이름을 쓰지 않는다 — 아래 참조)
  HAS_WALLET  지금 쓸 수 있는 별도 지갑이 생기는가
  PRINCIPLE   판단 원칙 한 줄
  facts/status  기전 고유 사실·개인 상태 (신규 기전만; 기존 grant·cashback 은
                dawn_context 의 검증된 분기를 그대로 둔다)

## 정책 이름을 쓰지 않는 이유

Sarkar & Vafa, *Lookahead Bias in Pretrained Language Models* (ICML 2025):
사전학습 데이터의 미래 정보가 과거만 써야 할 분석에 샌다. **과거 시점 경계를
지키라고 명시적으로 지시해도 막히지 않는다.**

우리는 2020~2021 정책을 2026년 모델로 백테스트하는데, 프롬프트가 "상생소비지원금"
이라고 이름을 대면 모델이 그 정책의 알려진 결과를 불러올 수 있다. 이름을 빼고
기전만 주면("이번 달 카드 사용액이 기준을 넘으면 초과분의 10%를 다음 달에
돌려받는다") 그 경로가 줄어든다. 완전한 차단은 아니며, 플라세보 정책 검정으로
남은 정도를 측정한다(docs/GENERALIZATION_METHOD.md §6·§7).

EXP_POLICY_ANONYMOUS=1 일 때만 익명 라벨을 쓴다. 기본 0 은 기존 동작이다.
"""
from __future__ import annotations

import os
from types import ModuleType

ANONYMOUS = os.environ.get("EXP_POLICY_ANONYMOUS", "0") == "1"

# 기존 표기(dawn_context._POLICY_TYPE_LABEL) — 익명 모드가 아닐 때 그대로 쓴다.
# 이 값을 바꾸면 P010 렌더가 달라진다.
_LEGACY_LABEL: dict[str, str] = {
    "voucher": "바우처", "discount": "할인", "subsidy": "환급/쿠폰",
    "grant": "지원금", "cashback": "캐시백",
}

# 기전 라벨 — 정책 이름이 아니라 '무엇을 하는 제도인가'. 익명 모드 전용.
_LABEL: dict[str, str] = {
    "grant": "정책지갑 지급",
    "subsidy": "환급",
    "voucher": "바우처",
    "cashback": "카드 실적 캐시백",
    "hours_limit": "영업시간 제한",
    "gathering_limit": "사적모임 인원 제한",
    "price_discount": "할인 구매 상품권",
    "sector_voucher": "업종 한정 할인권",
}

# 지금 당장 쓸 수 있는 별도 지갑이 생기는 기전
_WALLET_TYPES = frozenset({"grant", "subsidy", "voucher"})

# 판단 원칙 — 기전별로 다르다. 앞의 두 문장은 공통이고 마지막 한 문장만 갈린다.
_COMMON = ("소비 필요·시점·총액·POI는 본인의 평소 습관, 자산, 일정에 따라 판단한다. ")
_PRINCIPLE: dict[str, str] = {
    "wallet": _COMMON + (
        "정책 사용처에서 정책지갑으로 낼지 늘 쓰던 카드로 낼지는 결제 건마다 "
        "본인이 정한다. 정책이 있다는 것이 소비 자체를 새로 만들라는 뜻은 아니다."),
    "cashback": _COMMON + (
        "캐시백은 지금 쓸 수 있는 돈이 아니라 다음 달에 돌려받는 것이므로, "
        "이번 달 소비 예산을 늘려주지 않는다."),
    "hours_limit": _COMMON + (
        "정해진 시각 이후에는 그 업종 매장을 이용할 수 없다. 시간을 옮길지, "
        "다른 곳으로 갈지, 그만둘지는 본인이 정한다."),
    "gathering_limit": _COMMON + (
        "정해진 인원을 넘는 사적모임은 할 수 없다. 혼자 가거나 인원을 줄이거나 "
        "미루는 것 중 무엇을 할지는 본인이 정한다."),
    "price_discount": _COMMON + (
        "상품권은 본인 돈으로 미리 싸게 사 둔 것이라 새로 생긴 돈이 아니다. "
        "정해진 곳에서만 쓸 수 있다."),
    "sector_voucher": _COMMON + (
        "할인은 정해진 업종에서만, 정해진 조건을 채웠을 때만 적용된다. "
        "수량이 한정되어 있어 늘 받을 수 있는 것은 아니다."),
}

_MODULES: dict[str, ModuleType] = {}


def register(ptype: str, mod: ModuleType) -> None:
    _MODULES[ptype] = mod


def get(ptype: str | None) -> ModuleType | None:
    return _MODULES.get((ptype or "").strip())


def label(ptype: str | None, name: str | None = None) -> str:
    """프롬프트에 넣을 표시. 익명 모드면 정책 이름을 쓰지 않는다."""
    t = (ptype or "").strip()
    if ANONYMOUS:
        return f"[{_LABEL.get(t, t or '기타')}]"
    lab = _LEGACY_LABEL.get(t, t or "기타")
    nm = (name or "").strip()
    return f"[{lab}] {nm}" if nm else f"[{lab}]"


def has_wallet(ptypes) -> bool:
    return bool(set(ptypes) & _WALLET_TYPES)


def principle(ptypes) -> str:
    """활성 기전들에 맞는 판단 원칙. 지갑이 하나라도 있으면 지갑 원칙이 우선한다."""
    ts = {t for t in ptypes if t}
    if has_wallet(ts):
        return _PRINCIPLE["wallet"]
    for t in ("cashback", "sector_voucher", "price_discount",
              "hours_limit", "gathering_limit"):
        if t in ts:
            return _PRINCIPLE[t]
    return _PRINCIPLE["wallet"]


def known_types() -> tuple[str, ...]:
    return tuple(sorted(_LABEL))


# 신규 기전 모듈은 여기서 등록한다. grant·cashback 은 dawn_context 의 검증된
# 분기를 그대로 두고 라벨·원칙만 이 레지스트리를 거친다.
def _autoload() -> None:
    import importlib
    for name in ("hours_limit", "gathering_limit", "price_discount", "sector_voucher"):
        try:
            mod = importlib.import_module(f"{__name__}.{name}")
        except ImportError:
            continue
        register(getattr(mod, "PTYPE", name), mod)


_autoload()

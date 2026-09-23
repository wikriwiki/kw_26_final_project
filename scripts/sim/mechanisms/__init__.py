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
    # 아래 넷이 빠져 있어 한글 문장 한가운데에 영문 코드가 그대로 나갔다 —
    # 모델은 "[price_discount] 서울사랑상품권" 을 읽고 있었다. 라벨은 _LABEL 에
    # 이미 있던 것을 그대로 가져온다(제도가 무엇인지만 말하고 방향은 말하지 않는다).
    # grant·cashback 은 손대지 않는다 — P010 은 동결이고 그 렌더가 바뀌면 안 된다.
    "price_discount": "할인 구매 상품권",
    "sector_voucher": "업종 한정 할인권",
    "hours_limit": "영업시간 제한",
    "gathering_limit": "사적모임 인원 제한",
    "facility": "시설",
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
        "혜택은 정해진 업종에서 정해진 조건을 채웠을 때 적용된다. 그 업종의 "
        "일을 앞당길지, 평소대로 할지는 본인이 정한다."),
}

# dawn_context 가 예전부터 직접 그리는 기전들. 여기에 범용 모듈을 덧붙이면
# 같은 사실이 두 번 들어가고, P010 처럼 동결된 정책의 렌더가 달라진다.
_LEGACY_HANDLED = frozenset({"grant", "cashback", "subsidy", "voucher", "discount"})

_MODULES: dict[str, ModuleType] = {}


def register(ptype: str, mod: ModuleType) -> None:
    _MODULES[ptype] = mod


def get(ptype: str | None) -> ModuleType | None:
    """기전 모듈. **등록되지 않은 기전은 범용 모듈로 떨어진다.**

    None 을 돌려주면 그 정책은 사실·개인 상태 줄을 하나도 못 받는다. 새 정책이
    올 때마다 모듈을 요구하는 구조가 그 자리다 — 범용 모듈이 선언값만 읽어 옮긴다.
    """
    t = (ptype or "").strip()
    mod = _MODULES.get(t)
    if mod is not None:
        return mod
    if t in _LEGACY_HANDLED:
        # dawn_context 의 검증된 분기가 이미 이 기전을 그린다. 범용 모듈을 얹으면
        # 같은 사실이 두 번 들어간다(P012 에서 "1인 누적 한도 10만원" 이 본문과
        # 사실 줄에 겹쳤다). 범용은 **대체**지 추가가 아니다.
        return None
    from . import generic
    return generic


# 등록되지 않은 기전의 대체 표시. 영문 식별자를 한글 문장에 그대로 흘리지 않는다 —
# 모르는 기전이 들어와도 모델이 읽을 수 있는 말이어야 하고, 동시에 **무엇을 하라는
# 말이 아니어야** 한다. 조건은 정책 본문(description)이 이미 말한다.
FALLBACK_LABEL = "지원 제도"


def label(ptype: str | None, name: str | None = None) -> str:
    """프롬프트에 넣을 표시. 익명 모드면 정책 이름을 쓰지 않는다.

    등록되지 않은 기전이 들어오면 타입 문자열을 그대로 내보내던 자리가 있었다.
    그러면 "[interest_subsidy] 소상공인 이자지원" 처럼 한글 문장 한가운데에
    영문 식별자가 박힌다. 새 정책을 붙일 때마다 그 구멍이 다시 열리므로
    **모르는 기전은 중립 한글로 떨어뜨린다.**
    """
    t = (ptype or "").strip()
    if ANONYMOUS:
        return f"[{_LABEL.get(t) or FALLBACK_LABEL}]"
    lab = _LEGACY_LABEL.get(t) or _LABEL.get(t) or FALLBACK_LABEL
    nm = (name or "").strip()
    return f"[{lab}] {nm}" if nm else f"[{lab}]"


def has_wallet(ptypes) -> bool:
    return bool(set(ptypes) & _WALLET_TYPES)


def poi_restriction(policies, balances=None):
    """오늘 사용처가 제한되는 정책과, 그 판정 룰·표시 문구.

    **켜지는 조건이 기전마다 다르다.**

        지갑형(grant·subsidy·voucher)   잔액이 있어야 쓸 수 있다 → 잔액 > 0 일 때만
        결제시점형(그 밖의 전부)        지갑이 없다 → 정책이 발효 중이면 그 자리에서 걸린다

    이 구분이 빠져 있으면 **지갑 없는 정책은 사용처 표시가 영영 꺼진다.** 조건이
    "poi_restricted 이고 지갑 잔액 > 0" 하나였을 때 `sector_voucher`·
    `price_discount` 는 잔액이 영원히 0 이라 통째로 빠졌다. 에이전트에게는 자격
    있는 가게가 **하나도 없는 셈**이고, 그러면 그 업종을 피할 이유가 된다 —
    위약 런에서 대상 업종이 기대와 반대로 −10.5% 로 줄었다(`fc91872`).

    판정 룰과 표시 문구는 **정책이 선언한 것**을 쓴다. 여기에 정책별 분기를
    더하면 배관에서 1:1 결합이 되살아난다. 그래프에서 읽은 행은 선언이
    `mech_params`(JSON 문자열) 안에 들어 있으므로 양쪽을 다 본다 —
    이 한 줄이 없으면 DB 경로에서만 조용히 None 이 된다.

    여러 정책이 동시에 걸리면 **먼저 선언한 것**을 쓴다. 지금까지의 모든 런은
    사용처 제한 정책이 한 번에 하나였다.

    돌려주는 것
        ids     오늘 걸리는 정책 id 집합
        spec    적격 판정 규칙. 없으면 None — 호출부가 기존 쿠폰 룰로 떨어진다
        marker  후보 옆에 붙일 표시. 없으면 None — 호출부 기본값
    """
    import json as _json
    bal = balances or {}
    ids: set[str] = set()
    spec = None
    marker = None
    for p in (policies or []):
        if not p.get("poi_restricted"):
            continue
        pid = str(p.get("id") or "")
        if has_wallet([p.get("type")]):
            try:
                if int(bal.get(pid, 0) or 0) <= 0:
                    continue
            except (TypeError, ValueError):
                continue
        ids.add(pid)
        if spec is None:
            # 선언은 JSON 그대로 올 수도, mech_params 안에 실려 올 수도 있다.
            got = p.get("eligibility")
            if got is None:
                raw = p.get("mech_params")
                try:
                    mp = _json.loads(raw) if isinstance(raw, str) else (raw or {})
                except (TypeError, ValueError):
                    mp = {}
                got = mp.get("eligibility")
            if got:
                spec = got
        if marker is None and p.get("eligible_marker"):
            marker = p["eligible_marker"]
    return ids, spec, marker


def principle(ptypes) -> str:
    """활성 기전들에 맞는 판단 원칙. 지갑이 하나라도 있으면 지갑 원칙이 우선한다.

    **모르는 기전에 지갑 원칙을 떨어뜨리면 안 된다.** 예전에는 마지막 줄이
    지갑 원칙이어서, 지갑이 없는 정책에게 "정책지갑으로 낼지 늘 쓰던 카드로
    낼지" 라고 **있지도 않은 지갑을 사실처럼** 말했다. 라벨이 영문으로 새는 것은
    어색할 뿐이지만 이쪽은 거짓을 주입한다.
    """
    ts = {t for t in ptypes if t}
    if has_wallet(ts):
        return _PRINCIPLE["wallet"]
    for t in ("cashback", "sector_voucher", "price_discount",
              "hours_limit", "gathering_limit"):
        if t in ts:
            return _PRINCIPLE[t]
    from . import generic
    return generic.PRINCIPLE


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

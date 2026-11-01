"""실험 대상자 인터뷰 — 봉인된 실행 기록만 근거로, 근거 번호를 달아 답한다.

실험 브랜치(`scripts/sim/interview_evidence.py`)의 원칙을 따른다.

* 근거는 그 사람의 봉인된 metrics 행(영수증·하루 상태·대상 시설 결제)과, 받아 둔 경우 그날의 저널
  (모델이 밝힌 공개 이유·실행된 이벤트)이다. 고른 날짜(through_day) 뒤의 일은 넣지 않는다.
* 모델이 밝힌 이유는 **검증되지 않은 진술**로 따로 표시한다. 실행된 사실과 섞지 않는다.
* 건너뛴 날은 "기록 없음"이다. 아무 일도 없었던 날로 보지 않는다.
* 답변의 근거 번호는 실제로 보여 준 항목이어야 하고, 답변의 숫자는 근거에 있어야 한다. 어긋나면 경고로 돌려준다.

이 인터뷰는 **콘솔 탐색용**이다. 실험의 사후 인터뷰(봉인 기록)에 들어가지 않고 시뮬레이션에 되먹이지 않는다.
대화는 ``<미러>/<실험>/_console/interviews.jsonl`` 에만 남긴다.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.sim.evidence_integrity import EvidenceError, canonical, verify

from . import interview
from .experiments import AGENT_ID, CONSOLE_DIR, DAY, Arm, Experiment, ExperimentStore
from .store import StoreError

MAX_PROMPT_CHARS = 16000
FOCUS_RECENT_DAYS = 3
# 0.4 에서는 사람이 달라도 답이 거의 같았다(같은 첫마디 63~71%, '잘 모르겠어요' 88%).
# 0.7 에서 겹침이 약 25% 줄고 엉킨 문장은 0.4 의 옛 프롬프트와 같은 수준(약 4%)이었다 — 11/01 p017 8명×3회×2문항
TEMPERATURE = 0.7
MAX_RETRIES = 2
TARGET_WORDS = ("당구", "골프", "스크린", "금연", "흡연", "담배", "실내체육")
SMOKING_LABEL = {"smoker": "흡연함", "non_smoker": "흡연 안 함", "unknown": "흡연 여부 미상"}
PURCHASE_STATUS = {"not_purchased": "계획했지만 결제 안 함"}
FACILITY_LABEL = {"billiard": "당구장", "indoor_golf": "실내골프장"}
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
CITE = re.compile(r"\[E(\d{1,4})\]")
# 시뮬레이션 자체 프롬프트의 근거 번호(E0033 등)가 진술에 섞여 있다. 콘솔의 [E3] 과 헷갈리지 않게 바꿔 둔다.
SOURCE_REF = re.compile(r"\[?\bE(\d{3,})\b\]?")


def _plain(text: str) -> str:
    return SOURCE_REF.sub(lambda match: f"(입력 사실 {int(match.group(1))})", text)

SYSTEM = """당신은 서울에 사는 아래 [나]의 인물입니다. 인터뷰어의 질문에 그 사람으로서 대답하세요.

말하는 법:
- 실제 사람이 인터뷰에서 말하듯 자연스러운 구어체 존댓말로 2~5문장. 질문이 반말이어도 존댓말로 답하세요.
- 내 얘기를 하는 것입니다. '저는·제가'로 말하고, 내 행동에 '-셨-'을 붙이거나 나를 '본인'이라고 부르지 마세요.
- [나]에 적힌 것과 다른 사람처럼 말하지 마세요. 예를 들어 '흡연 안 함'이면 담배를 피운 적이 있는 것처럼 말하지 마세요.
- 질문을 되풀이하지 말고 첫 문장에서 바로 내 대답(그렇다·아니다·반반 중 어느 쪽인지)을 말하세요. '글쎄요'·'잘 모르겠어요' 같은 말로 시작하지 마세요.
- [요즘 상황]이 있으면 그 일을 겪고 있는 사람으로 답하세요. 거기 없는 규제·혜택은 지어내지 마세요.
- 사람마다 처지가 다릅니다. '조금 더 여유가 생겼다'처럼 누구나 할 법한 말 대신, 내 나이·직업·소득·소비 성향·생활 방식·평소 자주 쓰는 곳과 '내 요약'의 숫자(시행 전후 하루 평균, 받은 지원금, 많이 쓴 곳)에서 나온 나만의 이유를 말하세요.
- 의견이나 느낌을 물으면 그 처지에서 솔직하게, 입장을 분명히 답하세요. 절약형이면 아끼는 사람답게, 소비형이면 쓰는 사람답게 말하세요.
- 최근에 한 일은 [내 기록]에 있는 것만 말하세요. 기록에 없는 장소·금액·일은 지어내지 마세요. 기록에 없는 구체적인 일을 물었을 때만 "그건 잘 기억이 안 나요"라고 하세요.
- 금액은 꼭 필요할 때만, 기록의 값 그대로 쓰세요. 기분·피로는 숫자나 '수치'로 말하지 말고 느낌으로 말하세요. 날짜 나열, '기록'·'근거'·'시뮬레이션' 같은 말은 하지 마세요.
- '내역을 보면'처럼 기록을 읽어 주는 투로 말하지 말고, 겪은 일을 기억해서 말하는 사람처럼 답하세요.
- 지원금으로 낸 것은 '그중 정책 지원금'이 적힌 결제뿐입니다. '내 돈으로 냄'이나 '시행 전'인 날의 결제를 지원금으로 말하지 마세요.
- '그때 든 생각'과 그때 스스로 밝힌 이유는 확인된 사실이 아니니 "그땐 …싶었어요"처럼 말하세요. 가게 이름·메뉴·만족했는지는 내가 겪은 일이니 자연스럽게 말해도 됩니다.
- '들은 이야기'는 남에게 들은 것입니다. 내가 겪은 일처럼 말하지 마세요.
- 기록이 비어 있는 날은 무엇을 했는지 모르는 날입니다. 아무 일도 없었다고 말하지 마세요.
- 오늘은 {today}입니다. 그 뒤의 일은 모릅니다.

기록 번호(E3 같은 것)는 문장 안에 쓰지 마세요. 날짜는 '4월 23일'처럼 말하세요.
맨 마지막 줄에만 답에 쓴 기록 번호를 [E3][E7]처럼 적으세요. 이 줄은 화면에 보이지 않습니다.
"""


# 답 끝에 남은 번호·괄호 꼬리: "E24", "[답안]", "근거: E3, E7" 등
TRAILING_TAGS = re.compile(r"(?:\s*(?:근거\s*[:：]?|\[[^\]\n]{0,12}\]|\bE\d{1,4}\b|[,·]))+\s*$")


# 영어 단어 셋 이상이 이어진 조각. 'GS25', 'CU' 같은 상호는 한두 단어라 남는다
STRAY_LATIN = re.compile(r"\s*(?:\b[A-Za-z]+\b[ ,.'!?]*){3,}")
# 한글에 붙어 나온 소문자 영어 조각: "influence of support기보다는", "coexisting]".
# 한글도 단어 문자라 \b 가 안 걸린다. 상호는 대개 대문자(GS25·CU)라 소문자 낱말만 지운다
LOWER_LATIN = re.compile(r"\s*(?<![A-Za-z0-9])[a-z]{3,}(?:\s+[a-z]+)*(?![A-Za-z0-9])\]?")


# 근거 번호를 명사처럼 쓴 자리: "[E12]에서는", "E9부터", "E14 기록에", "E12일에"
# 한글도 단어 문자라 \b 로는 "E9부터" 의 경계를 못 잡는다. 앞뒤를 직접 막는다
NOUN_REF = re.compile(r"\[?(?<![A-Za-z0-9])E(\d{1,4})(?!\d)\]?(일)?(에서)?(?=\s*[가-힣])")
# 한국어 답에 섞여 나오는 다른 문자(그리스·키릴·한자·가나 등)
FOREIGN = re.compile(r"[\u0370-\u03FF\u0400-\u04FF\u0590-\u06FF\u0900-\u0DFF\u0E00-\u0E7F\u3040-\u30FF\u4E00-\u9FFF]+")
# 악센트 붙은 라틴 글자가 섞인 낱말 통째로: "ũng"
ACCENTED_WORD = re.compile(r"[A-Za-z]*[\u00C0-\u024F\u1E00-\u1EFF][A-Za-z\u00C0-\u024F\u1E00-\u1EFF]*")
# 문장 끝 뒤에 붙어 나온 영문 한두 글자: "생각해요.t"
TAIL_LATIN = re.compile(r"(?<=[.!?])[A-Za-z]{1,3}(?=\s|$)")
# 날짜를 괄호로 붙인 번호 "E13(11월 28일)" → 날짜만, 그 밖에 남은 번호 조각 "E7", "[E1" 은 지운다
REF_WITH_DAY = re.compile(r"\[?(?<![A-Za-z0-9])E\d{1,4}(?!\d)\]?\s*\(([^()\n]{1,20})\)")
LEFTOVER_REF = re.compile(r"\[?(?<![A-Za-z0-9])E\d{1,4}(?!\d)\]?|\[E?\d{0,4}(?=\s|$)")
# 말이 끝나는 꼴: "…요." "…다." "…죠?" 등. 이 꼴이 아닌 끝 조각("그래도 전체.")은 끊긴 문장이다
SENTENCE_END = re.compile(r"[요다죠까네지음함군걸게]\s*[.!?…~]*[\"'’”)]?$")


def _day_words(day: str | None) -> str:
    return f"{int(day[5:7])}월 {int(day[8:10])}일" if day and DAY.match(day) else ""


def spoken(answer: str, exposed: list[Item] | None = None) -> str:
    """화면에 보일 답 — 근거 번호를 떼고, 번호만 남은 마지막 줄은 지운다.

    번호를 명사처럼 쓴 자리("[E12]에서는")는 떼면 구멍이 나므로 그 근거의 날짜로 바꾼다.
    """
    lines = answer.strip().splitlines()
    tail = lines.pop() if len(lines) > 1 and not CITE.sub("", lines[-1]).strip(" .,·-") else ""

    def as_day(match: re.Match[str]) -> str:
        number = int(match.group(1))
        item = exposed[number - 1] if exposed and 1 <= number <= len(exposed) else None
        words = _day_words(item.day if item else None)
        return words + ("에" if words and match.group(3) else "")

    text = NOUN_REF.sub(as_day, "\n".join(lines))
    text = REF_WITH_DAY.sub(r"\1", text)
    # 떼어 낸 자리는 한 칸으로 둔다 — "많아졌어요[E5]7,500원" 이 "많아졌어요7,500원" 으로 붙지 않게
    text = CITE.sub(" ", text + ("\n" + tail if tail else ""))
    text = FOREIGN.sub("", text)
    text = ACCENTED_WORD.sub("", text).replace("()", "")
    text = LEFTOVER_REF.sub("", text)
    text = TAIL_LATIN.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([.,!?])", r"\1", text)  # 번호를 뗀 자리에 남은 공백
    text = STRAY_LATIN.sub("", text)  # 한국어 답 사이에 섞여 나온 영어 문장 조각
    text = LOWER_LATIN.sub("", text)
    text = TRAILING_TAGS.sub("", text.strip())
    lines = [line.rstrip() for line in text.strip().splitlines()]
    while lines and not lines[-1].strip(" .,·-"):
        lines.pop()
    if lines:
        # 마지막 줄 끝의 끊긴 조각을 걷어 낸다. 말이 끝난 문장이 하나는 남아야 한다
        sentences = re.split(r"(?<=[.!?])\s+", lines[-1])
        while len(sentences) > 1 and not SENTENCE_END.search(sentences[-1].strip()):
            sentences.pop()
        lines[-1] = " ".join(sentences)
    return "\n".join(lines).strip()


# 원문의 흠: 한국어 답에 섞인 다른 문자·영어 낱말, 꾸미는 말에서 끊긴 문장("평소처럼 필요한.", "그런 적.")
BROKEN_TAIL = re.compile(
    r"[가-힣](?:한|는|던|할|운|적)\.(?=\s|$|\[)"      # "평소처럼 필요한."
    r"|(?<=\s)적\.(?=\s|$|\[)"                       # "그런 적."
    r"|[는은던할] [가-힣]\.(?=\s|$|\[)"               # "준비를 하는 제."
    r"|(?:^|(?<=[.!?]\s))[가-힣]{1,3}, [가-힣]{0,3}[가-힣](?<![요다죠까네지음함군걸게])\.(?=\s|$|\[)"  # "네, 최근."
    r"|(?<![A-Za-z0-9])E\d{1,4}(?:\s*,\s*E\d{1,4}){2,}"             # 본문에 늘어놓은 기록 번호
    r"|\(기록\s*[:：]")


def answer_issues(answer: str) -> int:
    body = re.sub(r"\[[^\]\n]{0,12}\]", "", CITE.sub("", answer or ""))
    count = len(FOREIGN.findall(body)) + len(ACCENTED_WORD.findall(body))
    count += len(re.findall(r"(?<![A-Za-z0-9])[a-z]{3,}(?![A-Za-z0-9])", body))
    count += len(BROKEN_TAIL.findall(body))
    return count


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def won(value: Any) -> str:
    try:
        return f"{int(round(float(value))):,}원"
    except (TypeError, ValueError):
        return "금액 기록 없음"


@dataclass
class Item:
    evidence_id: str
    day: str | None
    kind: str
    text: str
    value: Any


DOSSIER_DIR = "dossier"
_DOSSIER_CACHE: dict[Path, tuple[tuple[int, int], dict[str, Any] | None]] = {}


def load_dossier(experiment: Experiment, agent_id: str) -> dict[str, Any] | None:
    """시뮬레이션이 이 사람에게 남긴 경험 기록(web/tools/extract_dossier.py 가 뽑아 둔 것). 없으면 None."""
    path = experiment.local_dir / CONSOLE_DIR / DOSSIER_DIR / f"{agent_id}.json.gz"
    try:
        stat = path.stat()
    except OSError:
        return None
    stamp = (stat.st_size, stat.st_mtime_ns)
    cached = _DOSSIER_CACHE.get(path)
    if cached and cached[0] == stamp:
        return cached[1]
    try:
        value = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    except (OSError, ValueError, EOFError):
        value = None
    if not isinstance(value, dict) or value.get("aid") != agent_id:
        value = None
    if len(_DOSSIER_CACHE) > 256:
        _DOSSIER_CACHE.clear()
    _DOSSIER_CACHE[path] = (stamp, value)
    return value


def _person(agent: dict[str, Any], dossier: dict[str, Any] | None) -> dict[str, Any]:
    """인터뷰가 말할 사람. 경험 기록이 있으면 시뮬레이션이 실제로 쓴 프로필이 이긴다.

    콘솔 번들(personas.json)은 다른 세대 파일이라 소득이 70~80% 다르고 생활 방식 문구는 전부 다르다(11/01 확인).
    그 파일에서만 오는 평소 지출·배달 횟수도 그때는 버린다.
    """
    profile = (dossier or {}).get("profile") or {}
    if not profile:
        return agent
    person = {key: agent.get(key) for key in ("id", "gu", "dong", "smoking_status", "district_code")}
    mapping = {"age": "age", "age_group": "age_group", "gender": "sex", "job": "job", "life_stage": "life_stage",
               "income_level": "income_level", "spending_tendency": "spending_tendency", "lifestyle": "lifestyle",
               "residence_dong": "dong", "gu": "gu", "workplace_dong": "work_dong", "commute_min": "commute_min",
               "s_daily_wd": "daily_spending_weekday"}
    for source, target in mapping.items():
        if profile.get(source) not in (None, ""):
            person[target] = profile[source]
    if isinstance(person.get("lifestyle"), str):
        person["lifestyle"] = " ".join(person["lifestyle"].split())
    top = profile.get("spending_top_wd_json")
    if isinstance(top, str):
        try:
            top = json.loads(top)
        except ValueError:
            top = None
    if isinstance(top, dict):
        person["top_categories"] = [name for name, _ in sorted(top.items(), key=lambda pair: -float(pair[1] or 0))[:3]]
    person["from_dossier"] = True
    return person


def _feeling(satisfaction: Any) -> str:
    try:
        value = float(satisfaction)
    except (TypeError, ValueError):
        return ""
    return "만족스러웠음" if value >= 0.8 else "그럭저럭이었음" if value >= 0.6 else "아쉬웠음"


def _visit_text(visit: dict[str, Any]) -> str:
    place = " ".join(str(visit[key]) for key in ("store", "menu") if visit.get(key) and str(visit.get(key)) != "None")
    bits = [place] if place else []
    if _feeling(visit.get("satisfaction")):
        bits.append(_feeling(visit.get("satisfaction")))
    why = visit.get("why") or visit.get("pick_why")
    if why:
        bits.append(f"그때 든 생각(확인되지 않은 진술): {why}")
    return (" · " + " · ".join(bits)) if bits else ""


def _profile_text(agent: dict[str, Any]) -> str:
    parts = []
    if agent.get("age") is not None:
        parts.append(f"{agent['age']}세")
    elif agent.get("age_group"):
        parts.append(str(agent["age_group"]))
    if agent.get("sex"):
        parts.append({"F": "여성", "M": "남성"}.get(str(agent["sex"]), str(agent["sex"])))
    place = " ".join(str(value) for value in (agent.get("gu"), agent.get("dong")) if value)
    if place:
        parts.append(f"{place} 거주")
    if agent.get("job"):
        parts.append(f"직업 {agent['job']}")
    if agent.get("life_stage"):
        parts.append(str(agent["life_stage"]))
    if agent.get("income_level"):
        parts.append(f"소득 {agent['income_level']}")
    if agent.get("work_dong"):
        commute = f", 통근 {agent['commute_min']}분" if agent.get("commute_min") else ""
        parts.append(f"직장 {agent['work_dong']}{commute}")
    if agent.get("spending_tendency"):
        parts.append(f"소비 성향 {agent['spending_tendency']}")
    if agent.get("lifestyle"):
        parts.append(f"생활 {agent['lifestyle']}")
    if agent.get("daily_spending_weekday"):
        weekend = f", 주말 {won(agent['daily_spending_weekend'])}" if agent.get("daily_spending_weekend") else ""
        parts.append(f"평소 하루 지출 평일 {won(agent['daily_spending_weekday'])}{weekend}")
    if agent.get("top_categories"):
        parts.append("평소 자주 쓰는 곳 " + "·".join(agent["top_categories"]))
    if agent.get("delivery_days") is not None:
        parts.append(f"한 달에 배달 시키는 날 약 {agent['delivery_days']}일")
    if agent.get("smoking_status"):
        parts.append(SMOKING_LABEL.get(str(agent["smoking_status"]), str(agent["smoking_status"])))
    return " · ".join(parts) or "프로필 기록 없음"


def _summary_text(rows: list[tuple[str, dict[str, Any]]], effective: str | None, policy_paired: bool,
                  visits: list[dict[str, Any]] | None = None) -> str | None:
    """시행 전후 하루 평균 지출과 지원금 합계 — 사람마다 다른 숫자라서 답이 그 사람 이야기가 된다."""
    def average(selected: list[dict[str, Any]]) -> float | None:
        values = [float(r["cm_today_total"]) for r in selected if r.get("cm_today_total") is not None]
        return sum(values) / len(values) if values else None

    bits: list[str] = []
    if effective and policy_paired:
        before = [r for d, r in rows if d < effective]
        after = [r for d, r in rows if d >= effective]
        avg_before, avg_after = average(before), average(after)
        if avg_before is not None:
            bits.append(f"시행 전 {len(before)}일 동안 가게에서 쓴 돈 하루 평균 {won(avg_before)}")
        if avg_after is not None:
            bits.append(f"시행 후 {len(after)}일 동안 하루 평균 {won(avg_after)}")
        if avg_before and avg_after is not None:
            change = (avg_after - avg_before) / avg_before * 100
            bits.append(f"시행 전보다 {'늘었음' if change >= 5 else '줄었음' if change <= -5 else '비슷함'}")
    else:
        avg = average([r for _, r in rows])
        if avg is not None:
            bits.append(f"{len(rows)}일 동안 가게에서 쓴 돈 하루 평균 {won(avg)}")
    received = sum(float(r.get("grant_applied_today") or 0) for _, r in rows)
    used = sum(float(r.get("policy_spend_today") or 0) for _, r in rows)
    rebate = sum(float(r.get("policy_rebate_today") or 0) for _, r in rows)
    if received > 0:
        bits.append(f"받은 정책 지원금 합계 {won(received)}, 그중 쓴 돈 {won(used)}")
    elif used > 0:
        bits.append(f"정책 지원금으로 결제한 합계 {won(used)}")
    if rebate > 0:
        bits.append(f"캐시백 적립 합계 {won(rebate)}")
    categories: dict[str, float] = {}
    for _, row in rows:
        for receipt in row.get("execution_receipts") or []:
            if isinstance(receipt, dict) and receipt.get("purchase_status") == "purchased" and receipt.get("category"):
                categories[str(receipt["category"])] = categories.get(str(receipt["category"]), 0.0) + float(receipt.get("amount") or 0)
    if categories:
        top = sorted(categories.items(), key=lambda pair: -pair[1])[:3]
        bits.append("가장 많이 쓴 곳 " + ", ".join(f"{name} {won(value)}" for name, value in top))
    stores: dict[str, int] = {}
    for visit in visits or []:
        if visit.get("store"):
            stores[str(visit["store"])] = stores.get(str(visit["store"]), 0) + 1
    regular = [(name, count) for name, count in sorted(stores.items(), key=lambda pair: -pair[1]) if count >= 2][:3]
    if regular:
        bits.append("자주 간 가게 " + ", ".join(f"{name} {count}번" for name, count in regular))
    return "내 요약: " + " · ".join(bits) if bits else None


def _journal_items(store: ExperimentStore, experiment: Experiment, arm: Arm, day: str, row: dict[str, Any]) -> tuple[list[Item], str]:
    """그날 저널이 이 PC 에 있으면 공개 이유·실행 이벤트를 꺼낸다. (항목들, 상태)"""
    reference = row.get("interview_evidence") or {}
    relative = reference.get("path")
    if not isinstance(relative, str):
        return [], "no_reference"
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not relative.startswith("evidence/"):
        return [], "invalid_reference"
    path = experiment.arm_dir(arm) / relative
    if not path.is_file():
        return [], "not_fetched"
    record = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if reference.get("record_id") and f'"record_id":"{reference["record_id"]}"' not in line:
                continue
            value = verify(json.loads(line))
            if value.get("record_id") == reference.get("record_id"):
                record = value
                break
    except (OSError, ValueError, EvidenceError):
        return [], "invalid_journal"
    if (
        record is None
        or record.get("integrity_sha256") != reference.get("integrity_sha256")
        or record.get("kind") != "agent_day"
        or record.get("day") != day
    ):
        return [], "journal_mismatch"
    items: list[Item] = []
    for index, event in enumerate(record.get("executed_events") or []):
        if not isinstance(event, dict):
            continue
        facility = FACILITY_LABEL.get(str(event.get("facility_type")), event.get("facility_type"))
        bits = [str(event.get("time") or ""), str(event.get("category") or "")]
        if facility:
            bits.append(f"({facility}{', 정책 대상' if event.get('policy_target') else ''})")
        if event.get("actual_spent") is not None:
            bits.append(won(event.get("actual_spent")))
        if event.get("actual_satisfaction") is not None:
            bits.append(f"만족도 {event['actual_satisfaction']}")
        items.append(Item(f"{arm.arm_id}:{day}:event:{index}", day, "executed_event", "한 일: " + " ".join(b for b in bits if b), event))
    for index, rationale in enumerate(record.get("stated_rationales") or []):
        statement = (rationale or {}).get("public_statement") or {}
        words = [str(statement.get(key)) for key in ("intent", "reasoning", "pick_reason", "trigger") if statement.get(key)]
        if not words:
            continue
        items.append(
            Item(
                f"{arm.arm_id}:{day}:rationale:{index}",
                day,
                "stated_rationale",
                "내가 밝힌 이유(확인되지 않은 진술): " + _plain(" / ".join(words))[:400],
                statement,
            )
        )
    return items, "used"


def build_packet(store: ExperimentStore, experiment: Experiment, arm_id: str, agent_id: str,
                 through_day: str | None = None) -> dict[str, Any]:
    if not AGENT_ID.match(agent_id or ""):
        raise StoreError(400, "잘못된 대상자 id 입니다")
    arm = experiment.arm(arm_id)
    roster = {item["id"]: item for item in store.roster(experiment.experiment_id)["items"]}
    if roster and agent_id not in roster:
        raise StoreError(404, f"이 실험의 대상자가 아닙니다: {agent_id}")
    lineage = experiment.lineage(arm)
    available = [
        (step, day)
        for step in lineage
        for day in step.day_list()
        if (experiment.arm_dir(step) / "metrics" / f"day_{day}.jsonl").is_file()
    ]
    if not available:
        raise StoreError(409, "이 팔의 실행 기록을 아직 받지 않았습니다")
    if through_day is not None and not DAY.match(through_day):
        raise StoreError(400, "날짜 형식은 YYYY-MM-DD 입니다")
    last_day = available[-1][1]
    through = through_day or last_day
    if through > last_day:
        raise StoreError(409, f"{through} 기록은 아직 없습니다 (마지막 {last_day})")

    dossier = load_dossier(experiment, agent_id) if arm.role == "treatment" else None
    person = _person(roster.get(agent_id) or {"id": agent_id}, dossier)
    items: list[Item] = [Item("profile", None, "profile", _profile_text(person), person)]
    # 영수증과 같은 날·같은 가게의 기억을 짝짓는다
    visits: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for visit in (dossier or {}).get("visits") or []:
        if visit.get("day") and visit.get("poi_id"):
            visits.setdefault((visit["day"], visit["poi_id"]), []).append(visit)
    heard_by_day: dict[str, list[dict[str, Any]]] = {}
    for heard in (dossier or {}).get("heard") or []:
        if heard.get("day") and heard.get("text"):
            heard_by_day.setdefault(heard["day"], []).append(heard)
    plans_by_day = {plan["day"]: plan["items"] for plan in (dossier or {}).get("plans") or [] if plan.get("day")}
    days, missing, skipped, journals = [], [], [], {}
    seen_rows: list[tuple[str, dict[str, Any]]] = []
    effective = (experiment.window or {}).get("effective_date")
    for step, day in available:
        if day > through:
            break
        days.append(day)
        row = store.agent_row(experiment, step, day, agent_id)
        if row is None:
            missing.append(day)
            continue
        if row.get("status") == "skipped":
            skipped.append(day)
            items.append(Item(f"{step.arm_id}:{day}:skipped", day, "skipped_day", "이날은 기록 없음(시뮬레이션이 이 사람의 하루를 만들지 못하고 건너뜀)", {"status": "skipped"}))
            continue
        seen_rows.append((day, row))
        state_bits = []
        # 숫자로 주면 '기분 점수'처럼 말한다. 느낌 말로 준다
        if row.get("mood") is not None:
            mood = float(row["mood"])
            state_bits.append("기분 좋음" if mood >= 0.8 else "기분 무난" if mood >= 0.6 else "기분 조금 가라앉음" if mood >= 0.4 else "기분 가라앉음")
        if row.get("fatigue") is not None and float(row["fatigue"]) >= 0.35:
            state_bits.append("많이 피곤" if float(row["fatigue"]) >= 0.6 else "조금 피곤")
        if row.get("cm_today_total") is not None:
            state_bits.append(f"이날 가게에서 쓴 돈 {won(row['cm_today_total'])}")
        if row.get("balance") is not None:
            state_bits.append(f"잔액 {won(row['balance'])}")
        # 정책 지원금 — 받은 날·쓴 돈·적립·소멸. 0 이면 적지 않는다
        for key, label in (
            ("grant_applied_today", "정책 지원금 받음"),
            ("policy_spend_today", "정책 지원금으로 결제"),
            ("policy_rebate_today", "캐시백 적립"),
            ("instant_discount_today", "즉시 할인"),
            ("grant_expired_today", "정책 지원금 소멸"),
        ):
            if (row.get(key) or 0) > 0:
                state_bits.append(f"{label} {won(row[key])}")
        if effective and experiment.kind == "policy_paired":
            name = re.sub(r"\s*\(.*?\)", "", experiment.title).strip() or "정책"
            state_bits.insert(0, f"{name} 시행 전" if day < effective else f"{name} 시행 후")
        items.append(Item(f"{step.arm_id}:{day}:state", day, "day_state", "하루 상태: " + " · ".join(state_bits), {k: row.get(k) for k in ("mood", "fatigue", "avg_sat", "cm_today_total", "balance", "grant_applied_today", "policy_spend_today", "policy_rebate_today", "instant_discount_today", "grant_expired_today")}))
        for index, receipt in enumerate(row.get("execution_receipts") or []):
            if not isinstance(receipt, dict) or receipt.get("kind") != "purchase_receipt":
                continue
            status = receipt.get("purchase_status")
            text = f"영수증: {receipt.get('scheduled_time') or ''} {receipt.get('category') or ''} {won(receipt.get('amount'))}"
            if status != "purchased":
                text += f" ({PURCHASE_STATUS.get(str(status), str(status))})"
            facts = receipt.get("policy_facts") if isinstance(receipt.get("policy_facts"), dict) else {}
            paid = sum(f.get("paid") or 0 for f in facts.values() if isinstance(f, dict))
            if status == "purchased" and paid > 0:
                own = receipt.get("own_paid")
                text += f" · 그중 정책 지원금 {won(paid)}" + (f", 내 돈 {won(own)}" if own is not None else "")
            elif status == "purchased" and experiment.kind == "policy_paired":
                text += " · 내 돈으로 냄"
            matched = visits.get((day, str(receipt.get("poi_id") or "")))
            visit = matched.pop(0) if matched and status == "purchased" else None
            if visit:
                text += _visit_text(visit)
            elif status != "purchased" and receipt.get("poi_id"):
                place = next((p.get("place") for p in plans_by_day.get(day, []) if p.get("time") == str(receipt.get("scheduled_time") or "")[:5]), None)
                if place:
                    text += f" · {place}"
            value = {**receipt, "memory": visit} if visit else receipt
            items.append(Item(receipt.get("event_id") or f"{step.arm_id}:{day}:receipt:{index}", day, "executed_receipt", text.strip(), value))
        for index, heard in enumerate(heard_by_day.get(day, [])):
            items.append(Item(f"{step.arm_id}:{day}:heard:{index}", day, "heard", "들은 이야기: " + heard["text"], heard))
        if plans_by_day.get(day):
            outing = [f"{p.get('time')} {p.get('place') or ''}({p.get('intent') or p.get('category') or ''})" for p in sorted(plans_by_day[day], key=lambda p: p.get("time") or "")]
            items.append(Item(f"{step.arm_id}:{day}:plan", day, "day_plan", "그날 나간 곳·할 일: " + ", ".join(outing)[:400], plans_by_day[day]))
        no_smoking = row.get("no_smoking") if isinstance(row.get("no_smoking"), dict) else {}
        aliases = (experiment.analysis or {}).get("facility_aliases") or {}
        districts = (experiment.analysis or {}).get("districts") or {}
        for index, poi in enumerate(no_smoking.get("by_poi") or []):
            if not isinstance(poi, dict):
                continue
            facility = aliases.get(poi.get("facility_type"), poi.get("facility_type"))
            label = FACILITY_LABEL.get(str(facility), str(facility))
            place = districts.get(str(poi.get("district_code")), str(poi.get("district_code") or ""))
            policy = "금연구역 정책 시행 중" if no_smoking.get("policy_active") else (
                "정책 시행 전" if effective and day < effective else "정책 없음(비교 팔)")
            items.append(
                Item(
                    f"{step.arm_id}:{day}:target:{index}",
                    day,
                    "target_facility",
                    f"정책 대상 시설 이용: {place} {label} {poi.get('payment_count')}회 결제, {won(poi.get('revenue_krw'))} ({policy})",
                    poi,
                )
            )
        journal_items, journal_state = _journal_items(store, experiment, step, day, row)
        journals[day] = journal_state
        items.extend(journal_items)
    summary = _summary_text(seen_rows, effective, experiment.kind == "policy_paired", [
        v for v in (dossier or {}).get("visits") or [] if v.get("day") and v["day"] <= through])
    if summary:
        items.insert(1, Item("summary", through, "summary", summary, None))
    if len({item.evidence_id for item in items}) != len(items):
        raise StoreError(500, "근거 id 가 겹칩니다")
    # 정책 있는 쪽 시민은 시행일 뒤로 정책을 겪고 있다. 정책 없는 쪽·시행 전에는 알리지 않는다
    context = ""
    if experiment.interview_context and arm.role == "treatment" and (not effective or through >= effective):
        context = experiment.interview_context
    return {
        "experiment_id": experiment.experiment_id,
        "arm_id": arm.arm_id,
        "context": context,
        "lineage": [step.arm_id for step in lineage],
        "agent_id": agent_id,
        "through_day": through,
        "days": days,
        "missing_days": missing,
        "skipped_days": skipped,
        "journal_status": journals,
        "items": items,
    }


def _mentioned_days(question: str, days: list[str]) -> set[str]:
    found: set[str] = set()
    for match in re.finditer(r"(\d{4})-(\d{1,2})-(\d{1,2})", question):
        found.add(f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}")
    for match in re.finditer(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", question):
        month, day = int(match.group(1)), int(match.group(2))
        found |= {d for d in days if int(d[5:7]) == month and int(d[8:10]) == day}
    for match in re.finditer(r"(\d{1,2})\s*/\s*(\d{1,2})", question):
        month, day = int(match.group(1)), int(match.group(2))
        found |= {d for d in days if int(d[5:7]) == month and int(d[8:10]) == day}
    return found & set(days)


MEMORABLE_VISITS = 6
MEMORABLE_HEARD = 3


def _memorable(items: list[Item], question: str, focus: set[str]) -> set[str]:
    """초점 날짜 밖에서도 꺼낼 경험: 질문에 나온 가게·업종, 그리고 기억에 크게 남은 방문·들은 이야기."""
    words = {word for word in re.findall(r"[가-힣A-Za-z0-9]{2,}", question)}
    visits = [item for item in items if item.kind == "executed_receipt" and isinstance(item.value, dict) and item.value.get("memory")]

    def importance(item: Item) -> float:
        value = item.value.get("memory", item.value) if isinstance(item.value, dict) else {}
        try:
            return float(value.get("importance") or 0)
        except (TypeError, ValueError):
            return 0.0
    asked = []
    for item in visits:
        memory = item.value["memory"]
        names = {str(memory.get("store") or ""), str(memory.get("category") or ""), str(item.value.get("category") or "")}
        if any(name and (name in question or any(word in name for word in words)) for name in names):
            asked.append(item)
    pinned = {item.evidence_id for item in sorted(asked, key=importance, reverse=True)[:MEMORABLE_VISITS + 2]}
    rest = [item for item in visits if item.day not in focus and item.evidence_id not in pinned]
    pinned |= {item.evidence_id for item in sorted(rest, key=importance, reverse=True)[:MEMORABLE_VISITS]}
    heard = [item for item in items if item.kind == "heard" and item.day not in focus]
    pinned |= {item.evidence_id for item in sorted(heard, key=importance, reverse=True)[:MEMORABLE_HEARD]}
    return pinned


def select_items(packet: dict[str, Any], question: str) -> list[Item]:
    """프롬프트에 넣을 근거를 고른다. 하루 상태·정책 대상 시설·건너뛴 날은 모두, 영수증·이유는 초점 날짜만."""
    items: list[Item] = packet["items"]
    days: list[str] = packet["days"]
    focus = set(days[-FOCUS_RECENT_DAYS:]) | _mentioned_days(question, days)
    if any(word in question for word in TARGET_WORDS):
        focus |= {item.day for item in items if item.kind == "target_facility" and item.day}
    always = {"profile", "summary", "day_state", "target_facility", "skipped_day"}
    pinned = _memorable(items, question, focus)
    chosen = [item for item in items if item.kind in always or item.day in focus or item.evidence_id in pinned]
    total = 0
    kept: list[Item] = []
    # 길이를 넘으면 오래된 초점 날짜의 세부부터 뺀다 (하루 상태·정책 대상 기록·기억에 남은 일은 남긴다)
    detail_days = sorted({item.day for item in chosen if item.kind not in always and item.day}, reverse=True)
    allowed = set(detail_days)
    while True:
        kept = [item for item in chosen if item.kind in always or item.evidence_id in pinned or item.day in allowed]
        total = sum(len(item.text) + 16 for item in kept)
        if total <= MAX_PROMPT_CHARS or not allowed:
            break
        allowed.discard(min(allowed))
    return kept


def _prompt(packet: dict[str, Any], exposed: list[Item]) -> str:
    profile = next((item.text for item in exposed if item.kind == "profile"), "")
    today = packet["through_day"]
    today = f"{today[:4]}년 {_day_words(today)}" if DAY.match(today) else today
    lines = [SYSTEM.replace("{today}", today), f"[나] {profile}"]
    if packet.get("context"):
        lines.append(f"[요즘 상황] {packet['context']}")
    lines.append("[내 기록]")
    for number, item in enumerate(exposed, start=1):
        if item.kind == "profile":
            lines.append(f"E{number} (기본) 위 [나]의 내용")  # 같은 글을 두 번 넣지 않는다
            continue
        when = "지금까지" if item.kind == "summary" else _day_words(item.day) or "기본"
        lines.append(f"E{number} ({when}) {item.text}")
    if packet["missing_days"]:
        lines.append("(기록을 받지 못한 날: " + ", ".join(packet["missing_days"]) + ")")
    lines.append(_reminder(packet, exposed))
    return "\n".join(lines)


def _voice(agent: dict[str, Any]) -> str:
    age = agent.get("age")
    if not isinstance(age, (int, float)):
        return ""
    if age < 30:
        return "20대답게 가볍고 짧은 말투"
    if age < 45:
        return "바쁜 30~40대답게 실용적이고 담백한 말투"
    if age < 60:
        return "50대답게 살림과 가족 형편을 따지는 말투"
    return "60대 이상답게 차분하고 지난 경험을 섞는 말투"


def _first_sentence(text: Any) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    return re.split(r"(?<=[.!?다])\s", text.strip(), maxsplit=1)[0].rstrip(".")


def _reminder(packet: dict[str, Any], exposed: list[Item]) -> str:
    """기록 목록이 길면 앞의 [나]가 묻힌다. 이 사람만의 사실을 끝에 한 번 더 놓는다."""
    agent = next((item.value for item in exposed if item.kind == "profile" and isinstance(item.value, dict)), {})
    summary = next((item.text for item in exposed if item.kind == "summary"), "")
    who = " · ".join(
        str(bit) for bit in (
            f"{agent['age']}세" if agent.get("age") else None,
            agent.get("job"),
            f"소득 {agent['income_level']}" if agent.get("income_level") else None,
            f"소비 성향 {agent['spending_tendency']}" if agent.get("spending_tendency") else None,
        ) if bit
    )
    lines = ["[답하기 전에]"]
    if who:
        lines.append(f"- 나는 {who}인 사람입니다." + (f" {_voice(agent)}로 말하세요." if _voice(agent) else ""))
    if _first_sentence(agent.get("lifestyle")):
        lines.append(f"- 내 생활: {_first_sentence(agent.get('lifestyle'))}.")
    if summary:
        lines.append(f"- {summary.replace('내 요약: ', '내 형편: ')}")
    if any(item.kind == "executed_receipt" and isinstance(item.value, dict) and item.value.get("memory") for item in exposed):
        lines.append("- 답할 때 내가 실제로 간 가게·먹은 것·그때 든 생각 가운데 질문에 맞는 것을 한두 개 꺼내 말하세요.")
    lines.append("- 질문 문장을 따라 하며 시작하지 말고, 첫마디부터 내 결론을 내 말로 하세요. 이 사람만 댈 수 있는 이유를 한 가지 이상 넣으세요.")
    lines.append("- 이유는 [나]와 [내 기록]에 있는 것에서만 고르세요. 거기 없는 사정(수입 변화·업무 일정·가족 일·건강)은 지어내지 마세요. 3~4문장이면 충분합니다.")
    return "\n".join(lines)


def check_answer(answer: str, exposed: list[Item]) -> dict[str, Any]:
    cited_numbers = []
    citations, warnings, seen = [], [], set()
    for match in CITE.finditer(answer):
        if match.group(1).startswith("0"):
            warnings.append(f"근거 번호 형식이 아닙니다: E{match.group(1)}")
            continue
        cited_numbers.append(int(match.group(1)))
    for number in cited_numbers:
        if number in seen:
            continue
        seen.add(number)
        if not 1 <= number <= len(exposed):
            warnings.append(f"보여 주지 않은 근거 번호를 댔습니다: E{number}")
            continue
        item = exposed[number - 1]
        citations.append({"n": number, "evidence_id": item.evidence_id, "day": item.day, "kind": item.kind, "text": item.text})
    if not cited_numbers:
        warnings.append("답변에 근거 번호가 없습니다")
    body = CITE.sub("", answer)
    cited_text = " ".join(f"{item.day or ''} {item.text}" for item in exposed).replace(",", "")
    unsupported = []
    for match in NUMBER.finditer(body):
        token = match.group(0).replace(",", "")
        if len(token) <= 1:
            continue  # 한 자리 수(순서·횟수 표현)는 오탐이 많다
        if token not in cited_text:
            unsupported.append(match.group(0))
    if unsupported:
        warnings.append("근거에서 찾지 못한 숫자: " + ", ".join(sorted(set(unsupported))[:8]))
    return {"citations": citations, "warnings": warnings, "unsupported_numbers": sorted(set(unsupported))}


_LOG_LOCK = threading.Lock()


def _log(experiment: Experiment, entry: dict[str, Any]) -> None:
    path = experiment.local_dir / CONSOLE_DIR / "interviews.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_LOCK, path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def interview_agent(store: ExperimentStore, experiment_id: str, arm_id: str, agent_id: str, question: str,
                    history: list[dict[str, str]] | None = None, through_day: str | None = None,
                    *, fetch_evidence: bool = True) -> dict[str, Any]:
    question = (question or "").strip()
    if not question:
        raise StoreError(400, "질문을 입력해 주세요")
    if len(question) > 500:
        raise StoreError(400, "질문이 너무 깁니다. 500자 안으로 줄여 주세요")
    experiment = store.experiment(experiment_id)
    fetch_note = None
    if fetch_evidence:
        fetch_note = _fetch_evidence(store, experiment, agent_id, through_day)
    packet = build_packet(store, experiment, arm_id, agent_id, through_day)
    exposed = select_items(packet, question)
    system = _prompt(packet, exposed)
    try:
        reply = interview.chat(system, question, history, temperature=TEMPERATURE)
        retried = 0
        # K-EXAONE 은 온도와 상관없이 답의 10~20% 에서 다른 문자·영어 낱말을 섞거나 문장을 끊는다(11/01 측정).
        # 흠이 있으면 두 번까지 다시 묻고 흠이 가장 적은 답을 쓴다
        while answer_issues(reply["answer"]) and retried < MAX_RETRIES:
            again = interview.chat(system, question, history, temperature=TEMPERATURE)
            retried += 1
            if answer_issues(again["answer"]) < answer_issues(reply["answer"]):
                reply = again
    except interview.InterviewError as exc:
        raise StoreError(exc.status, exc.message) from exc
    checked = check_answer(reply["answer"], exposed)
    packet_digest = hashlib.sha256(canonical([[i.evidence_id, i.text] for i in exposed]).encode("utf-8")).hexdigest()
    result = {
        "experiment_id": experiment_id,
        "arm_id": arm_id,
        "agent_id": agent_id,
        "through_day": packet["through_day"],
        # 근거 대조는 원문으로 하고, 화면에는 말한 부분만 보낸다
        "answer": spoken(reply["answer"], exposed),
        "model_label": reply.get("model_label"),
        "tokens": reply.get("tokens"),
        **checked,
        "packet": {
            "lineage": packet["lineage"],
            "days": len(packet["days"]),
            "missing_days": packet["missing_days"],
            "skipped_days": packet["skipped_days"],
            "exposed_items": len(exposed),
            "total_items": len(packet["items"]),
            "journal_days_used": sorted(day for day, state in packet["journal_status"].items() if state == "used"),
            "journal_days_missing": sorted(day for day, state in packet["journal_status"].items() if state == "not_fetched"),
            "digest": packet_digest,
        },
        "evidence_fetch": fetch_note,
        "retried": retried,
        "console_only": True,
    }
    _log(experiment, {
        "at": now(), "arm_id": arm_id, "agent_id": agent_id, "through_day": packet["through_day"],
        "question": question, "answer": reply["answer"], "model_label": reply.get("model_label"),
        "citations": [c["evidence_id"] for c in checked["citations"]], "warnings": checked["warnings"],
        "packet_digest": packet_digest,
    })
    return result


def _fetch_evidence(store: ExperimentStore, experiment: Experiment, agent_id: str, through_day: str | None) -> dict[str, Any] | None:
    """원격이 설정돼 있으면 그 사람의 저널을 받아 둔다. 실패해도 인터뷰는 metrics 만으로 계속한다."""
    from . import experiment_sync

    profile = experiment.remote.get("profile")
    if not profile:
        return None
    try:
        experiment_sync.ssh_base(str(profile))
    except experiment_sync.SyncError:
        return {"state": "remote_not_configured"}
    try:
        report = experiment_sync.fetch_agent_evidence(store, experiment, agent_id, through_day=through_day)
    except Exception as exc:  # 원격 장애가 인터뷰 자체를 막지 않게 한다
        return {"state": "failed", "error": str(exc)[:200]}
    return {"state": "ok", **report}


def recent_interviews(store: ExperimentStore, experiment_id: str, limit: int = 50) -> dict[str, Any]:
    experiment = store.experiment(experiment_id)
    path = experiment.local_dir / CONSOLE_DIR / "interviews.jsonl"
    if not path.is_file():
        return {"total": 0, "items": []}
    lines = path.read_text(encoding="utf-8").splitlines()[-max(1, min(limit, 500)):]
    items = []
    for line in reversed(lines):
        try:
            items.append(json.loads(line))
        except ValueError:
            continue
    return {"total": len(items), "items": items}

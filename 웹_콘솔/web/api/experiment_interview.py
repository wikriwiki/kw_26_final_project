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

SYSTEM = """당신은 서울시 정책 시뮬레이션 속 인물 한 명입니다. 아래 [내 기록]은 시뮬레이션이 실제로 남긴 기록입니다.
그 사람 본인의 말투로 질문에 답하세요.

지켜야 할 것:
- [내 기록]에 있는 사실만 말하세요. 기록에 없는 일은 "그건 기억에 없어요"라고 하세요.
- 사실을 말할 때마다 그 근거 번호를 문장 끝에 [E3]처럼 붙이세요. 여러 개면 [E3][E7]처럼 붙이세요.
- 금액·날짜·시각·횟수는 기록에 적힌 값을 그대로 쓰세요. 어림잡거나 합치지 마세요.
- '내가 밝힌 이유' 기록은 그때 스스로 말한 이유일 뿐 확인된 사실이 아닙니다. "그때는 …라고 생각했어요"처럼 말하세요.
- '기록 없음'인 날은 무엇을 했는지 모르는 날입니다. 아무 일도 없었다고 말하지 마세요.
- 오늘은 {today}입니다. 그 뒤의 일은 모릅니다.
- 3~6문장, 존댓말로 답하세요. 자신이 인공지능이나 시뮬레이션이라는 말은 하지 마세요.
"""


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
    if agent.get("smoking_status"):
        parts.append(SMOKING_LABEL.get(str(agent["smoking_status"]), str(agent["smoking_status"])))
    return " · ".join(parts) or "프로필 기록 없음"


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

    items: list[Item] = [Item("profile", None, "profile", _profile_text(roster.get(agent_id) or {"id": agent_id}), roster.get(agent_id))]
    days, missing, skipped, journals = [], [], [], {}
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
        state_bits = []
        for key, label in (("mood", "기분"), ("fatigue", "피로"), ("avg_sat", "만족도")):
            if row.get(key) is not None:
                state_bits.append(f"{label} {row[key]}")
        if row.get("cm_today_total") is not None:
            state_bits.append(f"이날 오프라인 소비 {won(row['cm_today_total'])}")
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
                text += f" · 그중 정책 지원금 {won(paid)}" + (f", 본인 부담 {won(own)}" if own is not None else "")
            items.append(Item(receipt.get("event_id") or f"{step.arm_id}:{day}:receipt:{index}", day, "executed_receipt", text.strip(), receipt))
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
    if len({item.evidence_id for item in items}) != len(items):
        raise StoreError(500, "근거 id 가 겹칩니다")
    return {
        "experiment_id": experiment.experiment_id,
        "arm_id": arm.arm_id,
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


def select_items(packet: dict[str, Any], question: str) -> list[Item]:
    """프롬프트에 넣을 근거를 고른다. 하루 상태·정책 대상 시설·건너뛴 날은 모두, 영수증·이유는 초점 날짜만."""
    items: list[Item] = packet["items"]
    days: list[str] = packet["days"]
    focus = set(days[-FOCUS_RECENT_DAYS:]) | _mentioned_days(question, days)
    if any(word in question for word in TARGET_WORDS):
        focus |= {item.day for item in items if item.kind == "target_facility" and item.day}
    always = {"profile", "day_state", "target_facility", "skipped_day"}
    chosen = [item for item in items if item.kind in always or item.day in focus]
    total = 0
    kept: list[Item] = []
    # 길이를 넘으면 오래된 초점 날짜의 세부부터 뺀다 (하루 상태·정책 대상 기록은 남긴다)
    detail_days = sorted({item.day for item in chosen if item.kind not in always and item.day}, reverse=True)
    allowed = set(detail_days)
    while True:
        kept = [item for item in chosen if item.kind in always or item.day in allowed]
        total = sum(len(item.text) + 16 for item in kept)
        if total <= MAX_PROMPT_CHARS or not allowed:
            break
        allowed.discard(min(allowed))
    return kept


def _prompt(packet: dict[str, Any], exposed: list[Item]) -> str:
    lines = [SYSTEM.replace("{today}", packet["through_day"]), "[내 기록]"]
    for number, item in enumerate(exposed, start=1):
        when = item.day or "기본"
        lines.append(f"E{number} ({when}) {item.text}")
    if packet["missing_days"]:
        lines.append("(기록을 받지 못한 날: " + ", ".join(packet["missing_days"]) + ")")
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
        reply = interview.chat(system, question, history)
    except interview.InterviewError as exc:
        raise StoreError(exc.status, exc.message) from exc
    checked = check_answer(reply["answer"], exposed)
    packet_digest = hashlib.sha256(canonical([[i.evidence_id, i.text] for i in exposed]).encode("utf-8")).hexdigest()
    result = {
        "experiment_id": experiment_id,
        "arm_id": arm_id,
        "agent_id": agent_id,
        "through_day": packet["through_day"],
        "answer": reply["answer"],
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

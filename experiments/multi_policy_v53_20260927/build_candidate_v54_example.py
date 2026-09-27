"""Freeze a policy-neutral v53 example repair without touching live sim code.

This builder makes no model call. Its output is an experiment artifact, not an
active prompt variant until separately registered after the current runs end.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
from prompts.v53 import SYSTEM_PROMPT as V53  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build() -> tuple[str, dict]:
    opening = "예시 (실제 dong_code는 페르소나 블록 참조 / reasoning은 페르소나 → 행동 직결이 아닌 살아있는 흐름):\n"
    closing = "\n※ 하루 일정(events)을 먼저"
    if V53.count(opening) != 1 or V53.count(closing) != 1:
        raise ValueError("v53 output example markers changed")
    prefix, tail = V53.split(opening, 1)
    old_example, suffix = tail.split(closing, 1)
    pattern = re.compile(r'  (\{"time":"(?P<time>\d\d:\d\d)".*?\n\s+"trigger":"[^"]+"\}),?',
                         re.DOTALL)
    found = {match.group("time"): (json.loads(match.group(1)), match.group(1))
             for match in pattern.finditer(old_example)}
    expected = {"08:10", "08:50", "12:00", "15:00", "18:10", "18:40",
                "19:00", "17:30", "16:30", "17:00", "19:20"}
    if set(found) != expected or len(re.findall(r"\n\s+\.\.\.\n", old_example)) != 1:
        raise ValueError("v53 historical example changed")
    # The source example says "on the way to/from work" but has no workplace
    # anchors. Three workplace blocks make its 09–18 work duration coherent.
    work = {
        "09:20": ('{"time":"09:20","anchor":"workplace","category":"직장","intent":"오전 업무",\n'
                  '   "reasoning":"출근해 오전에 맡은 일을 처리한다.","trigger":"none"}'),
        "13:00": ('{"time":"13:00","anchor":"workplace","category":"직장","intent":"오후 업무",\n'
                  '   "reasoning":"점심 뒤 직장으로 돌아와 업무를 잇는다.","trigger":"none"}'),
        "15:30": ('{"time":"15:30","anchor":"workplace","category":"직장","intent":"업무 복귀",\n'
                  '   "reasoning":"짧은 휴식을 마치고 남은 일을 처리한다.","trigger":"none"}'),
    }
    selected = ["08:10", "08:50", "09:20", "12:00", "13:00",
                "15:00", "15:30", "18:10", "19:20"]
    raw = {time: item[1] for time, item in found.items()}
    raw.update(work)
    home = ('{"time":"20:10","anchor":"residence","category":"집","intent":"귀가와 휴식",\n'
            '   "reasoning":"오늘 밖에서 해야 할 일을 마치고 집으로 돌아와 쉰다.",\n'
            '   "trigger":"none"}')
    events = [json.loads(raw[time]) for time in selected] + [json.loads(home)]
    minutes = [int(event["time"][:2]) * 60 + int(event["time"][3:])
               for event in events]
    if (len(events) != 10 or events[0]["anchor"] != "residence"
            or events[-1]["anchor"] != "residence"
            or any(b - a < 20 for a, b in zip(minutes, minutes[1:]))):
        raise ValueError("repaired example violates the current schedule contract")
    work_minutes = sum(min(b, 18 * 60) - max(a, 9 * 60)
                       for event, a, b in zip(events, minutes, minutes[1:])
                       if event["anchor"] == "workplace"
                       and min(b, 18 * 60) > max(a, 9 * 60))
    if work_minutes < 4 * 60:
        raise ValueError("weekday worker example lacks four hours of work")
    replacement = ('{"events": [\n'
                   + ',\n'.join('  ' + raw[time] for time in selected)
                   + ',\n  ' + home + '\n ],\n "daily_propensity": 0.72}\n'
                   + "\n※ 위 예시의 건강·마트 이벤트는 '그런 날'의 예시일 뿐이다. 오늘이 병원 갈 날도,\n"
                   + " 장을 볼 날도 아니면 넣지 않는다. 미용·쇼핑·학원·안경점도 필요와 주기가 돌아온 날에만\n"
                   + " 같은 방식으로 넣는다.")
    if json.loads(replacement.split('\n※', 1)[0])["events"] != events:
        raise ValueError("candidate example did not round-trip as JSON")
    candidate = prefix + opening + replacement + closing + suffix
    if candidate.count(closing) != 1 or candidate == V53:
        raise ValueError("failed to make exactly one example replacement")
    frozen_inputs = ROOT / "output/validation_v53_prodtemp_frozen_v3_20260926/frozen_inputs.json"
    manifest = {
        "schema": "candidate_prompt_freeze_v1",
        "status": "not_executed",
        "candidate": "v54-example-contract",
        "base": "v53",
        "change_scope": "one contiguous Stage1 JSON output example and immediately following explanatory note; policy text and Stage2 unchanged",
        "base_system_sha256": sha(V53.encode("utf-8")),
        "candidate_system_sha256": sha(candidate.encode("utf-8")),
        "supersedes_candidate_system_sha256": "6b102e77d039508a4f7d8b7719ed0c2c3dcaad7ddb540fc9c5b64d5c821bbfe2",
        "superseded_before_calls": True,
        "frozen_inputs_path": str(frozen_inputs.relative_to(ROOT)).replace("\\", "/"),
        "frozen_inputs_sha256": sha(frozen_inputs.read_bytes()),
        "selected_example_event_times": [event["time"] for event in events],
        "removed_optional_examples": ["17:30 tuition", "16:30 shoes",
                                      "17:00 rice cooker", "18:40 pharmacy",
                                      "19:00 haircut"],
        "weekday_workplace_minutes_09_to_18": work_minutes,
        "model_calls": 0,
        "stage2_system_sha256": None,
    }
    return candidate, manifest


if __name__ == "__main__":
    candidate, manifest = build()
    system_path = HERE / "candidate_v54_example_system.txt"
    with system_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(candidate)
    if sha(system_path.read_bytes()) != manifest["candidate_system_sha256"]:
        raise ValueError("saved candidate SHA differs from rendered prompt")
    (HERE / "candidate_v54_example_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(manifest["candidate_system_sha256"])

#!/usr/bin/env python3
"""이어 달린 run 조각들을 **하나의 run 산출물**로 합친다.

왜 필요한가
-----------
이중차분은 정책 시행일 **이전** 일자가 있어야 계산된다. 그런데 실험은 한 디렉터리에
쭉 쌓이지 않는다. 무정책 구간을 돌려 산출물을 닫고, 그래프를 정산·덤프한 뒤, 정책을
주입하고 다음 구간을 새 디렉터리에 돌린다(`chain_p3r.sh` 가 그렇게 짜여 있다).
그래서 사전 구간과 사후 구간이 **다른 폴더에** 있고, 어느 쪽도 혼자서는 DID 를 못 만든다.

같은 그래프 상태를 이어받아 날짜가 연속인 조각들만 합치면, 그 결과는 "정책 시행일을
가운데 둔 하나의 run" 과 같다. 이 스크립트는 그 합치기를 **기록을 남기며** 한다.

무엇을 조심하는가
-----------------
조각마다 내보낸 스크립트가 달라 필드가 어긋난다. 예를 들어 하루치 내보내기는
`poi_dong`(POI.dong_code)을 담고, 구간 전체 내보내기는 `dong`(POI.adm_cd)을 담는데
후자는 값이 비어 있다. 이런 필드를 그대로 두면 **"사전에는 지역 정보가 없고 사후에는
있다"** 가 되어, 정책이 지역 데이터를 만들어낸 것처럼 읽힌다.

그래서 공통 필드만 남기고 나머지는 버린다. 무엇을 버렸는지는 manifest 에 적는다.
같은 날짜가 두 조각에 겹치면 합치지 않고 멈춘다 — 어느 쪽이 옳은지는 사람이 정할 몫이다.

사용::

    python scripts/report/merge_run_segments.py \
        --out <합칠 경로> \
        --segment <사전구간>:events.jsonl \
        --segment <사후구간>:events_2025-07-21.jsonl
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

#: 두 내보내기 경로가 **모두** 담고 있는 필드. 이것만 남긴다.
COMMON_FIELDS = ("day", "day_type", "l1", "sub", "amt", "sp", "ex", "wba", "elig")

#: 공통이 아니라서 버리는 필드를 `dong` 처럼 이름만 다른 경우까지 포함해 적어둔다.
#: 값을 지어내지 않는다 — 한쪽에만 있는 값은 비교에 쓸 수 없다.
DROPPED_NOTE = {
    "dong": "구간 전체 내보내기의 POI.adm_cd — 값이 전부 비어 있다",
    "poi_dong": "하루치 내보내기의 POI.dong_code — 사전 구간에 대응 값이 없다",
    "res_dong": "거주지 코드 — 사전 구간에 없다",
    "res_dong_name": "거주지 이름 — 사전 구간에 없다",
    "work_dong_name": "직장 이름 — 사전 구간에 없다",
    "trigger": "행동 촉발 사유 — 사전 구간에 없다",
    "pick_factor": "선택 가중 — 사전 구간에 없다",
    "time": "결제 시각 — 사전 구간에 없다",
}


def _segment(spec: str) -> tuple[Path, str]:
    """`<디렉터리>:<이벤트파일>` 을 가른다. 윈도우 드라이브 문자(`C:`)를 살린다."""
    head, sep, tail = spec.rpartition(":")
    if not sep or len(head) < 2:
        raise argparse.ArgumentTypeError(f"형식은 <디렉터리>:<이벤트파일> 입니다: {spec}")
    root = Path(head)
    if not root.is_dir():
        raise argparse.ArgumentTypeError(f"디렉터리가 없습니다: {root}")
    if not (root / tail).is_file():
        raise argparse.ArgumentTypeError(f"이벤트 파일이 없습니다: {root / tail}")
    return root, tail


def merge(out: Path, segments: list[tuple[Path, str]]) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(exist_ok=True)

    seen_days: dict[str, str] = {}
    dropped: set[str] = set()
    parts: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    seg_args: list[dict[str, Any]] = []
    stamps: list[str] = []
    total = 0

    with (out / "events.jsonl").open("w", encoding="utf-8", newline="\n") as sink:
        for root, name in segments:
            days: dict[str, int] = {}
            paid = 0
            with (root / name).open(encoding="utf-8") as src:
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    dropped |= set(row) - set(COMMON_FIELDS)
                    day = row.get("day")
                    owner = seen_days.setdefault(day, root.name)
                    if owner != root.name:
                        raise SystemExit(
                            f"같은 날짜가 두 조각에 있습니다: {day} "
                            f"({owner} / {root.name}). 어느 쪽을 쓸지 먼저 정하세요."
                        )
                    kept = {k: row.get(k) for k in COMMON_FIELDS}
                    # 지역 코드는 양쪽이 같은 뜻이 아니라 버렸다. 없는 값은 없다고 적는다
                    kept["dong"] = None
                    sp = kept.get("sp") or "{}"
                    if isinstance(sp, dict):
                        kept["sp"] = json.dumps(sp, ensure_ascii=False)
                        sp_map = sp
                    else:
                        try:
                            sp_map = json.loads(sp)
                        except (TypeError, ValueError):
                            sp_map = {}
                    if sp_map:
                        paid += 1
                    sink.write(json.dumps(kept, ensure_ascii=False) + "\n")
                    days[day] = days.get(day, 0) + 1
                    total += 1

            # 일자별 곁다리 파일도 함께 가져온다. 이게 빠지면 스냅샷 검사가
            # "완료 run 인데 일자 기록이 전부 있지는 않다"며 보고서 생성을 막는다 —
            # 막는 게 맞다. 이벤트만 있고 실행 기록이 없는 run 은 근거가 반쪽이다.
            for day in sorted(days):
                for sub, pattern in (
                    ("metrics", f"day_{day}.jsonl"),
                    ("timing", f"day_{day}.json"),
                    ("timing", f"slow_{day}.json"),
                    ("checkpoints", f"done_{day}.json"),
                    ("checkpoints", f"failed_{day}.json"),
                ):
                    source = root / sub / pattern
                    if source.is_file():
                        (out / sub).mkdir(exist_ok=True)
                        shutil.copy2(source, out / sub / source.name)
            # run 전체에 하나뿐인 곁다리. 먼저 나온 조각의 것을 쓴다 —
            # POI 목록은 구간이 바뀌어도 같은 도시다.
            poi = root / "poi_summary.json"
            if poi.is_file() and poi.stat().st_size > 0 and not (out / poi.name).is_file():
                shutil.copy2(poi, out / poi.name)

            summary = root / "summary.json"
            if summary.is_file():
                payload = json.loads(summary.read_text(encoding="utf-8"))
                summary_rows += [r for r in payload.get("summary", []) if r.get("day") in days]
                seg_args.append(payload.get("args") or {})
                stamp = payload.get("completed_at") or payload.get("updated_at")
                if stamp:
                    stamps.append(stamp)

            parts.append(
                {
                    "root": str(root),
                    "events_file": name,
                    "days": sorted(days),
                    "events": sum(days.values()),
                    "policy_paid_events": paid,
                }
            )

    def agreed(key: str) -> Any:
        """조각들이 같은 값을 말할 때만 그 값을 쓴다. 어긋나면 모른다고 둔다."""
        values = {json.dumps(a.get(key), sort_keys=True) for a in seg_args}
        return json.loads(values.pop()) if len(values) == 1 else None

    days_sorted = sorted(seen_days)
    # 합쳐진 스냅샷의 계획값. 지어낸 값이 아니라 **합쳐진 결과 그 자체**를 적는다:
    # 시작일은 첫 날, 일수는 실제로 들어온 일수. 그래서 `planned == present` 가 되고,
    # 이 스냅샷은 "8일치가 온전히 있는 완결된 기록"으로 읽힌다. 조각 각각의 원래
    # 계획(7일씩)과 어디서 왔는지는 merge_manifest.json 에 그대로 남는다.
    (out / "summary.json").write_text(
        json.dumps(
            {
                "summary": sorted(summary_rows, key=lambda r: r["day"]),
                "args": {
                    "start": days_sorted[0] if days_sorted else None,
                    "days": len(days_sorted),
                    "limit": agreed("limit"),
                    "gu": agreed("gu"),
                    "workers": agreed("workers"),
                },
                "completed_at": max(stamps) if stamps else None,
                "merged": True,
                "merged_note": "조각 산출물을 이어 붙인 스냅샷입니다. 출처는 merge_manifest.json 참고.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "merged_from": parts,
        "days": sorted(seen_days),
        "events": total,
        "kept_fields": list(COMMON_FIELDS) + ["dong(=null)"],
        "dropped_fields": {
            field: DROPPED_NOTE.get(field, "한쪽 조각에만 있어 비교에 쓸 수 없다")
            for field in sorted(dropped)
        },
    }
    (out / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--segment",
        required=True,
        action="append",
        dest="segments",
        type=_segment,
        help="<디렉터리>:<이벤트파일>. 날짜 순서대로 준다",
    )
    args = ap.parse_args(argv)
    manifest = merge(args.out, args.segments)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

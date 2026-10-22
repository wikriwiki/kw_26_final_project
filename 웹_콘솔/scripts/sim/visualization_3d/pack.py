"""3D 시각화 데이터 팩 — 열(column) 단위 바이너리 타임라인 + 대상자별 지연 로딩.

예전 산출물은 JSON 4종(agents/timeline/memories/events)을 JS 리터럴로 HTML 에 통째로
박았다. 시연 표본(1,825명 × 120프레임)만으로 109MB 가 되고, 브라우저는 그걸 객체로
파싱하느라 수 초를 쓰고 1GB 넘는 힙을 잡는다. 힙이 크면 GC 한 번이 길어져 재생 중에도
끊긴다.

팩은 화면이 실제로 쓰는 모양으로 미리 바꿔 둔다.

* ``manifest.json`` — 프레임 목록, 사전(카테고리·동·의도·앵커), 합계, 화면용 메타, 파일 목록·해시
* ``agents.json.gz`` — 대상자 표 (선택·검색·자치구 색에 필요, 부팅 시 1회)
* ``timeline/NNN.bin.gz`` — 하루치 프레임 묶음. 열마다 ``frame × agent`` 순의 typed array
* ``detail/NNN.json.gz`` — 대상자 묶음별 기억·일정. 지도에서 사람을 고를 때만 받는다
* ``extra.json.gz`` — 런타임이 아직 쓰지 않는 파생 레이어(소비 버스트·소문 엣지·POI 색인)

원본 JSON 의 의미는 바꾸지 않는다. 런타임이 쓰던 값(위치, 색 판정에 쓰는 l1/cat, 소비,
residence 앵커, 합계, 약속 여부)은 같은 규칙으로 계산해 넣는다.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import sys
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.sim.visualization_3d.derive import build_viz_meta
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    from visualization_3d.derive import build_viz_meta  # type: ignore[no-redef]

PACK_FORMAT = "sim-viz-pack"
PACK_VERSION = 1

# (열 이름, 원본 키, 종류). 종류: f32 = float32(없음은 NaN), i32 = int32(없음은 0),
# dict = 사전 코드(0 은 없음, 1부터 manifest.dicts[이름][코드-1]).
TIMELINE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("x", "lon", "f32"),
    ("y", "lat", "f32"),
    ("cat", "cat", "dict"),
    ("l1", "l1", "dict"),
    ("dong", "dong", "dict"),
    ("intent", "intent", "dict"),
    ("anchor", "anchor", "dict"),
    ("sat", "sat", "f32"),
    ("spent", "spent", "i32"),
)

# 화면이 부팅할 때 쓰는 메타만 manifest 에 둔다. 나머지 파생 레이어는 extra 로 뺀다.
BOOT_META_KEYS = ("agent_count", "frame_summaries", "policy_zones", "meetups")

DETAIL_SHARD_TARGET_BYTES = 1_500_000
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


class PackError(ValueError):
    """입력이 팩을 만들 수 없는 모양일 때."""


def build_pack(
    agents: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
    memories: dict[str, dict[str, Any]],
    events: dict[str, list[dict[str, Any]]],
    *,
    policy_dongs: list[dict[str, Any]] | None = None,
    viz_meta: dict[str, Any] | None = None,
    pack_id: str | None = None,
    label: str | None = None,
    run_id: str | None = None,
    source: dict[str, Any] | None = None,
    shard_size: int | None = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """팩을 메모리에서 만든다. ``(manifest, {상대경로: gzip 바이트})`` 를 돌려준다."""
    if not isinstance(timeline, list) or not timeline:
        raise PackError("timeline 이 비어 있습니다")
    if viz_meta is None:
        viz_meta = build_viz_meta(agents, timeline, memories, events, policy_dongs or [])

    roster = _roster(agents, timeline)
    index_by_id = {str(agent["id"]): position for position, agent in enumerate(roster)}
    agent_count = len(roster)
    frames = [_frame_info(frame, position) for position, frame in enumerate(timeline)]

    # 사전을 먼저 다 모은다. 그래야 코드 폭(uint8/16/32)이 모든 청크에서 같다.
    dicts, dict_codes = _build_dicts(timeline)
    columns = {}
    for name, key, kind in TIMELINE_COLUMNS:
        if kind == "dict":
            columns[name] = {"source": key, "dtype": _code_dtype(len(dicts[name])), "dict": name}
        else:
            columns[name] = {"source": key, "dtype": "float32" if kind == "f32" else "int32"}

    files: dict[str, bytes] = {}
    chunks = []
    for chunk_number, (frame_start, frame_count) in enumerate(_day_chunks(timeline)):
        raw, offsets = _encode_chunk(
            timeline[frame_start : frame_start + frame_count],
            agent_count,
            index_by_id,
            dict_codes,
            columns,
        )
        name = f"timeline/{chunk_number:03d}.bin.gz"
        files[name] = _gzip(raw)
        chunks.append(
            {
                "file": name,
                "frame_start": frame_start,
                "frame_count": frame_count,
                "day": frames[frame_start]["day"],
                "raw_bytes": len(raw),
                "offsets": offsets,
            }
        )

    files["agents.json.gz"] = _gzip(_json_bytes(roster))

    shard_size = shard_size or _detail_shard_size(roster, memories, events)
    detail_shards = []
    for shard_number, start in enumerate(range(0, agent_count, shard_size)):
        members = roster[start : start + shard_size]
        payload = {"v": PACK_VERSION, "agents": {}}
        for agent in members:
            agent_id = str(agent["id"])
            bundle = _detail_bundle(memories.get(agent_id), events.get(agent_id))
            if bundle:
                payload["agents"][agent_id] = bundle
        name = f"detail/{shard_number:03d}.json.gz"
        files[name] = _gzip(_json_bytes(payload))
        detail_shards.append({"file": name, "agent_start": start, "agent_count": len(members)})

    extra = {key: value for key, value in viz_meta.items() if key not in BOOT_META_KEYS}
    files["extra.json.gz"] = _gzip(_json_bytes(extra))

    meta = {key: viz_meta.get(key) for key in BOOT_META_KEYS if key in viz_meta}
    manifest: dict[str, Any] = {
        "format": PACK_FORMAT,
        "version": PACK_VERSION,
        "pack_id": pack_id,
        "label": label,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "agent_count": agent_count,
        "frame_count": len(frames),
        "frames": frames,
        "first_active_frame": _first_active_frame(meta.get("frame_summaries") or []),
        "columns": columns,
        "dicts": dicts,
        "chunks": chunks,
        "agents": {"file": "agents.json.gz"},
        "detail": {"shard_size": shard_size, "shards": detail_shards},
        "extra": {"file": "extra.json.gz"},
        "totals": _memory_totals(memories),
        "appointment_agents": _appointment_agents(roster, memories),
        "meta": meta,
        "source": source or {},
    }
    _attach_file_digests(manifest, files)
    return manifest, files


def build_pack_from_dir(
    viz_dir: Path,
    *,
    policy_dongs: list[dict[str, Any]] | None = None,
    **options: Any,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """``agents/timeline/memories/events.json`` 이 있는 폴더에서 팩을 만든다."""
    viz_dir = Path(viz_dir)
    payload = {}
    source = {}
    for key, filename in (
        ("agents", "agents.json"),
        ("timeline", "timeline.json"),
        ("memories", "memories.json"),
        ("events", "events.json"),
    ):
        raw = (viz_dir / filename).read_bytes()
        source[filename] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        payload[key] = json.loads(raw.decode("utf-8"))
    return build_pack(
        payload["agents"],
        payload["timeline"],
        payload["memories"],
        payload["events"],
        policy_dongs=policy_dongs,
        source=source,
        **options,
    )


def write_pack(out_dir: Path, manifest: dict[str, Any], files: dict[str, bytes]) -> Path:
    """팩을 폴더에 쓴다. 다 쓴 뒤에 바꿔 끼워, 읽는 쪽이 반쯤 쓴 팩을 보지 않게 한다."""
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = out_dir.with_name(f".{out_dir.name}.tmp{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    for relative, data in files.items():
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (staging / "manifest.json").write_bytes(_json_bytes(manifest, indent=1))

    retired = out_dir.with_name(f".{out_dir.name}.old{os.getpid()}")
    if out_dir.exists():
        out_dir.rename(retired)
    staging.rename(out_dir)
    if retired.exists():
        shutil.rmtree(retired, ignore_errors=True)
    return out_dir


def read_manifest(pack_dir: Path) -> dict[str, Any]:
    manifest = json.loads((Path(pack_dir) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != PACK_FORMAT:
        raise PackError(f"시각화 팩이 아닙니다: {pack_dir}")
    return manifest


def verify_pack(pack_dir: Path) -> list[str]:
    """manifest 의 크기·해시와 실제 파일을 대조한다. 어긋난 항목을 돌려준다."""
    pack_dir = Path(pack_dir)
    manifest = read_manifest(pack_dir)
    problems = []
    for entry in _file_entries(manifest):
        path = pack_dir / entry["file"]
        if not path.is_file():
            problems.append(f"없음: {entry['file']}")
            continue
        data = path.read_bytes()
        if len(data) != entry.get("bytes") or hashlib.sha256(data).hexdigest() != entry.get("sha256"):
            problems.append(f"해시 불일치: {entry['file']}")
    return problems


def _roster(agents: list[dict[str, Any]], timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    roster: list[dict[str, Any]] = []
    seen: set[str] = set()
    for agent in agents or []:
        if not isinstance(agent, dict) or agent.get("id") is None:
            continue
        agent_id = str(agent["id"])
        if agent_id in seen:
            continue
        seen.add(agent_id)
        roster.append({**agent, "id": agent_id})
    # 타임라인에만 있는 대상자도 예전 화면은 집 좌표 없이 그렸다. 빠뜨리지 않는다.
    for frame in timeline:
        for item in frame.get("agents") or []:
            agent_id = item.get("id") if isinstance(item, dict) else None
            if agent_id is None or str(agent_id) in seen:
                continue
            seen.add(str(agent_id))
            roster.append({"id": str(agent_id)})
    if not roster:
        raise PackError("대상자가 없습니다")
    return roster


def _frame_info(frame: dict[str, Any], position: int) -> dict[str, Any]:
    return {
        "day": None if frame.get("day") is None else str(frame.get("day")),
        "hour": frame.get("hour"),
        "label": frame.get("label") or f"Frame {position}",
        "day_idx": frame.get("day_idx"),
    }


def _day_chunks(timeline: list[dict[str, Any]]) -> Iterable[tuple[int, int]]:
    """같은 날짜가 이어지는 프레임을 한 묶음으로 자른다."""
    start = 0
    for position in range(1, len(timeline) + 1):
        if position == len(timeline) or timeline[position].get("day") != timeline[start].get("day"):
            yield start, position - start
            start = position


def _build_dicts(timeline: list[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    """사전 열의 값 목록. 처음 나온 순서대로 1부터 번호를 매긴다(0 은 값 없음)."""
    dict_columns = [(name, key) for name, key, kind in TIMELINE_COLUMNS if kind == "dict"]
    values: dict[str, list[str]] = {name: [] for name, _key in dict_columns}
    codes: dict[str, dict[str, int]] = {name: {} for name, _key in dict_columns}
    for frame in timeline:
        for item in frame.get("agents") or []:
            if not isinstance(item, dict):
                continue
            for name, key in dict_columns:
                text = _dict_text(item.get(key))
                if text is not None and text not in codes[name]:
                    values[name].append(text)
                    codes[name][text] = len(values[name])
    return values, codes


def _dict_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _encode_chunk(
    frames: list[dict[str, Any]],
    agent_count: int,
    index_by_id: dict[str, int],
    dict_codes: dict[str, dict[str, int]],
    columns_spec: dict[str, dict[str, Any]],
) -> tuple[bytes, dict[str, dict[str, Any]]]:
    size = len(frames) * agent_count
    columns: dict[str, list[Any]] = {}
    for name, _key, kind in TIMELINE_COLUMNS:
        columns[name] = [math.nan] * size if kind == "f32" else [0] * size

    for frame_offset, frame in enumerate(frames):
        base = frame_offset * agent_count
        for item in frame.get("agents") or []:
            if not isinstance(item, dict) or item.get("id") is None:
                continue
            slot = base + index_by_id[str(item["id"])]
            for name, key, kind in TIMELINE_COLUMNS:
                value = item.get(key)
                if kind == "f32":
                    columns[name][slot] = _float_or_nan(value)
                elif kind == "i32":
                    columns[name][slot] = _int32(value)
                else:
                    text = _dict_text(value)
                    columns[name][slot] = 0 if text is None else dict_codes[name][text]

    parts: list[bytes] = []
    offsets: dict[str, dict[str, Any]] = {}
    cursor = 0
    for name, _key, kind in TIMELINE_COLUMNS:
        if kind == "f32":
            typed = array("f", columns[name])
            dtype = "float32"
        elif kind == "i32":
            typed = array("i", columns[name])
            dtype = "int32"
        else:
            dtype = columns_spec[name]["dtype"]
            typed = array(_ARRAY_CODES[dtype], columns[name])
        if typed.itemsize != _DTYPE_BYTES[dtype]:  # pragma: no cover - exotic platforms only.
            raise PackError(f"{dtype} 의 크기가 {typed.itemsize} 바이트입니다")
        if sys.byteorder != "little":  # pragma: no cover - big-endian hosts only.
            typed.byteswap()
        padding = (-cursor) % 8
        if padding:
            parts.append(b"\0" * padding)
            cursor += padding
        data = typed.tobytes()
        offsets[name] = {"offset": cursor, "dtype": dtype, "length": len(typed)}
        parts.append(data)
        cursor += len(data)
    return b"".join(parts), offsets


def _code_dtype(dict_size: int) -> str:
    if dict_size < 0xFF:
        return "uint8"
    if dict_size < 0xFFFF:
        return "uint16"
    return "uint32"


_ARRAY_CODES = {"uint8": "B", "uint16": "H", "uint32": "I"}
_DTYPE_BYTES = {"float32": 4, "int32": 4, "uint8": 1, "uint16": 2, "uint32": 4}


def _float_or_nan(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _int32(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(number):
        return 0
    return max(INT32_MIN, min(INT32_MAX, int(round(number))))


def _detail_bundle(memory: Any, agent_events: Any) -> dict[str, Any] | None:
    bundle: dict[str, Any] = {}
    if isinstance(memory, dict):
        # `visited` 는 `memories` 중 type=visited 를 그대로 복사한 목록이다(실측 1,825/1,825
        # 일치). 화면은 합계에만 썼고 그 합계는 manifest.totals 에 미리 넣었다.
        bundle["memories"] = {key: value for key, value in memory.items() if key != "visited"}
    if isinstance(agent_events, list) and agent_events:
        bundle["events"] = agent_events
    return bundle or None


def _detail_shard_size(
    roster: list[dict[str, Any]],
    memories: dict[str, dict[str, Any]],
    events: dict[str, list[dict[str, Any]]],
) -> int:
    sample = roster[:: max(1, len(roster) // 64)][:64]
    sizes = [
        len(_json_bytes(_detail_bundle(memories.get(agent["id"]), events.get(agent["id"])) or {}))
        for agent in sample
    ]
    average = max(1, sum(sizes) // max(1, len(sizes)))
    return max(1, min(256, DETAIL_SHARD_TARGET_BYTES // average))


def _memory_totals(memories: dict[str, dict[str, Any]]) -> dict[str, int]:
    visited = appointments = rumors = 0
    for bundle in (memories or {}).values():
        if not isinstance(bundle, dict):
            continue
        visited += len(bundle.get("visited") or [])
        appointments += len(bundle.get("appointments") or [])
        rumors += sum(1 for item in bundle.get("memories") or [] if isinstance(item, dict) and item.get("type") == "rumor")
    return {"visited": visited, "appointments": appointments, "rumors": rumors}


def _appointment_agents(roster: list[dict[str, Any]], memories: dict[str, dict[str, Any]]) -> list[int]:
    flagged = []
    for position, agent in enumerate(roster):
        bundle = (memories or {}).get(agent["id"])
        if isinstance(bundle, dict) and bundle.get("appointments"):
            flagged.append(position)
    return flagged


def _first_active_frame(summaries: list[dict[str, Any]]) -> int:
    for position, summary in enumerate(summaries):
        if (summary or {}).get("active_agents"):
            return position
    return 0


def _attach_file_digests(manifest: dict[str, Any], files: dict[str, bytes]) -> None:
    for entry in _file_entries(manifest):
        data = files[entry["file"]]
        entry["bytes"] = len(data)
        entry["sha256"] = hashlib.sha256(data).hexdigest()


def _file_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        manifest["agents"],
        manifest["extra"],
        *manifest["chunks"],
        *manifest["detail"]["shards"],
    ]


def _json_bytes(value: Any, indent: int | None = None) -> bytes:
    separators = None if indent else (",", ":")
    return json.dumps(value, ensure_ascii=False, separators=separators, indent=indent, allow_nan=False).encode("utf-8")


def _gzip(data: bytes) -> bytes:
    # mtime 을 고정해 같은 입력이면 같은 바이트가 나오게 한다(해시로 캐시를 깨기 위해).
    return gzip.compress(data, compresslevel=6, mtime=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="3D 시각화 데이터 팩을 만든다")
    parser.add_argument("--src", required=True, type=Path, help="agents/timeline/memories/events.json 폴더")
    parser.add_argument("--out", required=True, type=Path, help="팩을 쓸 폴더")
    parser.add_argument("--id", dest="pack_id", help="팩 이름 (콘솔이 고를 때 쓴다)")
    parser.add_argument("--label", help="화면에 보일 설명")
    parser.add_argument("--run-id", help="이 팩이 담은 실행 id")
    parser.add_argument("--with-policy-zones", action="store_true", help="Neo4j 에서 정책 구역을 읽는다")
    args = parser.parse_args(argv)

    policy_dongs: list[dict[str, Any]] = []
    if args.with_policy_zones:
        try:
            from scripts.sim.build_standalone_html import fetch_policy_dongs_safe
        except ModuleNotFoundError:  # pragma: no cover
            from build_standalone_html import fetch_policy_dongs_safe  # type: ignore[no-redef]
        policy_dongs = fetch_policy_dongs_safe()

    manifest, files = build_pack_from_dir(
        args.src,
        policy_dongs=policy_dongs,
        pack_id=args.pack_id or args.out.name,
        label=args.label,
        run_id=args.run_id,
    )
    write_pack(args.out, manifest, files)
    total = sum(len(data) for data in files.values())
    print(
        f"  -> {args.out}: agents {manifest['agent_count']:,}, frames {manifest['frame_count']},"
        f" {len(files)} files, {total / 1024 / 1024:.1f} MB"
    )
    problems = verify_pack(args.out)
    if problems:
        print("  검증 실패:", *problems, sep="\n    ")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())

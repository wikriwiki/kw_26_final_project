#!/usr/bin/env python3
"""3D 지도 산출물에서 **볼 것이 없는 부분**을 덜어낸다.

두 가지를 덜어낸다.

1. **아무도 움직이지 않는 새벽 시간대.** 에이전트는 자는 동안에도 집에 "있으므로"
   프레임이 비어 있지는 않다. 비어 있지 않을 뿐 아무 일도 일어나지 않아서,
   재생하면 사용자가 멍하니 넘겨야 하는 구간이 된다. 지울 시간대를 인자로 받는다.

2. **클릭 상세 기록**(memories·events). 지도를 그리는 데 쓰이지 않으면서 용량의
   대부분을 차지한다. 대상자 개인 기록은 `대상자 문답` 화면이 맡는다.

에이전트는 **한 명도 빼지 않는다.** 지도는 표본이 아니라 이 실행에 있던
사람들을 보여주는 자리다.

사용::

    python scripts/sim/trim_standalone.py --in <원본> --out <결과> [--gzip]
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

DATA_PREFIX = {
    "agents": "window.__AGENTS__ = ",
    "timeline": "window.__TIMELINE__ = ",
    "memories": "window.__MEMORIES__ = ",
    "events": "window.__EVENTS__ = ",
}


def _payload(line: str, prefix: str):
    body = line[len(prefix) :].rstrip()
    return json.loads(body[:-1] if body.endswith(";") else body)


def trim(src: Path, out: Path, drop: set[int]) -> dict[str, int]:
    stats = {"frames_in": 0, "frames_out": 0, "agents": 0}
    with src.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            if line.startswith(DATA_PREFIX["timeline"]):
                frames = _payload(line, DATA_PREFIX["timeline"])
                stats["frames_in"] = len(frames)
                kept = [f for f in frames if f.get("hour") not in drop and f.get("agents")]
                stats["frames_out"] = len(kept)
                fout.write(DATA_PREFIX["timeline"] + json.dumps(kept, ensure_ascii=False) + ";\n")
            elif line.startswith(DATA_PREFIX["memories"]):
                fout.write(DATA_PREFIX["memories"] + "{};\n")
            elif line.startswith(DATA_PREFIX["events"]):
                fout.write(DATA_PREFIX["events"] + "{};\n")
            else:
                if line.startswith(DATA_PREFIX["agents"]):
                    stats["agents"] = len(_payload(line, DATA_PREFIX["agents"]))
                fout.write(line)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--drop-hours", default="0-5",
                    help="지울 시간대 (예: 0-5). 비우면 시간대를 지우지 않는다")
    ap.add_argument("--gzip", action="store_true", help="전송용 압축본도 함께 만든다")
    args = ap.parse_args()

    drop: set[int] = set()
    if args.drop_hours.strip():
        lo, _, hi = args.drop_hours.partition("-")
        drop = set(range(int(lo), int(hi or lo) + 1))
    stats = trim(args.src, args.out, drop)
    size = args.out.stat().st_size
    print(
        f"에이전트 {stats['agents']:,}명 (그대로)  "
        f"프레임 {stats['frames_in']} → {stats['frames_out']} "
        f"(새벽 등 {stats['frames_in'] - stats['frames_out']}개 제거)"
    )
    print(f"본문 {size / 1e6:.1f} MB")
    if args.gzip:
        packed = args.out.with_suffix(args.out.suffix + ".gz")
        with args.out.open("rb") as fin, gzip.open(packed, "wb", compresslevel=6) as fout:
            shutil.copyfileobj(fin, fout, 4 * 1024 * 1024)
        print(f"전송 {packed.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""기존 Leaflet standalone HTML 에서 시각화 데이터 4종을 추출한다.

왜 필요한가
-----------
`scripts/sim/visualization_3d/` (deck.gl 3D) 는 코드만 있고 산출물이 없다.
빌더 `build_standalone_html.py` 는 `VIZ_OUT_DIR` 의 agents/timeline/memories/events.json
을 읽는데, 이 JSON 들은 `export_visualization.py` 가 Neo4j 에서 뽑아야 생긴다.

다행히 2026-06 에 만들어진 Leaflet standalone 이 **같은 전역 규약**을 쓴다:
    window.__AGENTS__ / __TIMELINE__ / __MEMORIES__ / __EVENTS__
따라서 그 HTML 에서 값만 도로 꺼내면 Neo4j 없이 3D 빌드를 돌릴 수 있다.

scripts/ 아래 파일은 읽기만 하고 수정하지 않는다.

사용법
------
    python web/tools/extract_viz_json.py \
        --src output/sim/visualization/sim_standalone_fast.html \
        --out web/viz_store/demo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

KEYS = ("__AGENTS__", "__TIMELINE__", "__MEMORIES__", "__EVENTS__")
OUT_NAMES = {
    "__AGENTS__": "agents.json",
    "__TIMELINE__": "timeline.json",
    "__MEMORIES__": "memories.json",
    "__EVENTS__": "events.json",
}


def find_value(text: str, key: str) -> str:
    """`window.__KEY__ = <JSON>;` 의 JSON 부분을 괄호 균형으로 잘라낸다.

    정규식으로 끊지 않는 이유: 값 안에 `;` 와 `}` 가 무수히 들어 있어
    탐욕/비탐욕 어느 쪽으로도 경계를 못 잡는다. 여는 괄호부터 세면서
    문자열 리터럴과 이스케이프를 건너뛰는 방식이 유일하게 안전하다.
    """
    marker = f"window.{key}"
    at = text.find(marker)
    if at < 0:
        raise SystemExit(f"{key} 를 찾지 못했습니다")
    eq = text.index("=", at + len(marker))
    i = eq + 1
    while text[i] in " \t\r\n":
        i += 1
    if text[i] not in "[{":
        raise SystemExit(f"{key} 값이 배열/객체가 아닙니다: {text[i]!r}")

    start = i
    depth = 0
    in_str = False
    esc = False
    while i < len(text):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1
    raise SystemExit(f"{key} 값의 끝을 찾지 못했습니다")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    print(f"읽는 중: {args.src} ({args.src.stat().st_size / 1024 / 1024:.0f} MB)")
    text = args.src.read_text(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for key in KEYS:
        raw = find_value(text, key)
        value = json.loads(raw)  # 유효성 확인 — 깨진 조각을 그대로 쓰지 않는다
        path = args.out / OUT_NAMES[key]
        path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        n = len(value) if isinstance(value, (list, dict)) else "?"
        mb = path.stat().st_size / 1024 / 1024
        summary[OUT_NAMES[key]] = {"items": n, "mb": round(mb, 1)}
        print(f"  {OUT_NAMES[key]:16s} {n:>8} 건  {mb:6.1f} MB")

    (args.out / "_extract.json").write_text(
        json.dumps({"source": str(args.src), "files": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"완료: {args.out}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()

"""3D 지도 저장본을 미리 채운다 — 시연장 인터넷 없이도 지도가 바로 뜨게.

    python web/tools/prefetch_map.py --cache <데이터 폴더>/map_cache

받는 것 (인터넷이 되는 곳에서 한 번, 다시 돌리면 없는 것만 받는다)
- 지도 라이브러리 3개 (maplibre-gl js·css, deck.gl)            약 2MB
- 바탕 지도 스타일·타일 설명·아이콘·라틴 글꼴
- 서울 일대 벡터 조각, 확대 0~14단계 (15단계 이상은 14단계를 키워 그린다)
- 서울 일대 음영 기복 그림, 확대 0~6단계

한글 글자는 브라우저가 컴퓨터 글꼴로 그린다(maplibre 기본값). 그래서 한글 글꼴 조각은 받지 않는다.
저장본 밖(서울 밖으로 멀리 끌고 간 곳)은 서버가 인터넷이 될 때 그때 받아 저장한다.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# 서울 경계 + 여백 (경도 126.76~127.19, 위도 37.41~37.72)
BBOX = (126.70, 37.38, 127.27, 37.76)
FONT_RANGES = [(start, start + 255) for start in range(0, 2048, 256)] + [
    (8192, 8447), (8448, 8703), (12288, 12543), (65280, 65535)
]


def tile_range(zoom: int) -> tuple[range, range]:
    def x_of(lon: float) -> int:
        return int((lon + 180.0) / 360.0 * (1 << zoom))

    def y_of(lat: float) -> int:
        rad = math.radians(lat)
        return int((1.0 - math.asinh(math.tan(rad)) / math.pi) / 2.0 * (1 << zoom))

    west, south, east, north = BBOX
    return range(x_of(west), x_of(east) + 1), range(y_of(north), y_of(south) + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="3D 지도 저장본 채우기")
    parser.add_argument("--cache", required=True, help="저장할 폴더 (예: <데이터 폴더>/map_cache)")
    parser.add_argument("--max-zoom", type=int, default=14)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cache = Path(args.cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["SIM_MAP_CACHE_DIR"] = str(cache)
    os.environ.setdefault("SIM_MAP_FETCH_TIMEOUT", "20")
    from web.api import map_cache

    started = time.perf_counter()
    # 1) 라이브러리
    vendor = cache / "vendor"
    vendor.mkdir(exist_ok=True)
    for name, url in map_cache.VENDOR.items():
        target = vendor / name
        if not target.is_file():
            target.write_bytes(map_cache.download(url))
        print(f"라이브러리 {name} {target.stat().st_size // 1024}KB")

    # 2) 스타일 → 타일 설명 → 아이콘·글꼴 이름을 스타일에서 읽는다
    style_raw = map_cache.get("styles/bright")
    if style_raw is None:
        sys.exit("바탕 지도 스타일을 받지 못했습니다. 인터넷 연결을 확인하세요.")
    style = json.loads(style_raw)
    tilejson = json.loads(map_cache.get("planet") or b"{}")
    tile_template = (tilejson.get("tiles") or [""])[0]
    if not tile_template.startswith(map_cache.UPSTREAM):
        sys.exit("타일 설명을 받지 못했습니다.")
    tile_prefix = tile_template[len(map_cache.UPSTREAM) + 1:].split("/{z}")[0]  # planet/<판>
    sprite = str(style.get("sprite") or "")[len(map_cache.UPSTREAM) + 1:]
    fonts = sorted({font for layer in style.get("layers", []) for font in (layer.get("layout") or {}).get("text-font", []) if isinstance(font, str)})

    jobs: list[str] = []
    if sprite:
        jobs += [f"{sprite}{suffix}" for suffix in (".json", ".png", "@2x.json", "@2x.png")]
    jobs += [f"fonts/{font}/{lo}-{hi}.pbf" for font in fonts for lo, hi in FONT_RANGES]
    max_zoom = min(args.max_zoom, int(tilejson.get("maxzoom") or 14))
    for zoom in range(0, max_zoom + 1):
        xs, ys = tile_range(zoom)
        jobs += [f"{tile_prefix}/{zoom}/{x}/{y}.pbf" for x in xs for y in ys]
    for zoom in range(0, 7):
        xs, ys = tile_range(zoom)
        jobs += [f"natural_earth/ne2sr/{zoom}/{x}/{y}.png" for x in xs for y in ys]

    print(f"받을 것 {len(jobs)}개 (이미 있는 것은 건너뜀)…", flush=True)
    missing = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(map_cache.get, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            if future.result() is None:
                missing += 1
            if index % 100 == 0 or index == len(jobs):
                print(f"  {index}/{len(jobs)}", flush=True)
    size = sum(path.stat().st_size for path in cache.rglob("*") if path.is_file())
    print(f"끝 — {time.perf_counter() - started:.0f}초, 저장본 {size / 1e6:.0f}MB, 못 받은 것 {missing}개")
    if missing:
        print("  (글꼴 범위 일부는 원래 없는 파일이라 못 받은 것으로 셀 수 있습니다. 지도에는 지장 없음)")


if __name__ == "__main__":
    main()

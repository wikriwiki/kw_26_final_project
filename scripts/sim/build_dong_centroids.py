"""행정동 중심좌표 추출 (Problem A — Huff 거리 계산용, Neo4j 불필요).

기존 시각화 산출물(`sim_standalone`)에 박혀 있는 POI 좌표(lon/lat/dong)를 동별로 평균내
**행정동 중심좌표**를 만든다. 이 좌표로 거주지→허브 거리를 순수 Python(haversine)으로
계산 → 런타임 Neo4j 거리쿼리 없이 광역 목적지 prior 구성.

입력 (로컬):
  output/sim/visualization/sim_standalone.zip (또는 .html) — window.__EVENTS__ 에 POI lon/lat/dong
  output/stats/agent_profiles.json — dong명 ↔ 8자리 코드 매핑

출력:
  output/stats/dong_centroids.json
    { "centroids": {dong8: [lon, lat]}, "gu_fallback": {gu5: [lon, lat]}, "_meta": {...} }

사용: python build_dong_centroids.py
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STATS = REPO / "output" / "stats"
VIZ_ZIP = REPO / "output" / "sim" / "visualization" / "sim_standalone.zip"
VIZ_HTML = REPO / "output" / "sim" / "visualization" / "sim_standalone.html"


def _canon(s: str) -> str:
    """동명 정규화 — '독산제1동' → '독산1동'."""
    return re.sub(r"제(?=\d)", "", s or "")


def _load_events_html() -> str:
    if VIZ_ZIP.exists():
        z = zipfile.ZipFile(VIZ_ZIP)
        name = [n for n in z.namelist() if n.endswith(".html")][0]
        return z.read(name).decode("utf-8")
    return io.open(VIZ_HTML, encoding="utf-8").read()


def _grab_events(html: str) -> dict:
    marker = "window.__EVENTS__ = "
    for ln in html.split("\n"):
        if ln.startswith(marker):
            body = ln[len(marker):].rstrip()
            body = body[:-1] if body.endswith(";") else body
            return json.loads(body)
    raise RuntimeError("window.__EVENTS__ 를 viz에서 찾지 못함")


def build_centroids() -> dict:
    # 동명(정규화) → 8자리 코드 (agent_profiles 기준; 중복명은 last-wins 근사)
    profiles = json.load(io.open(STATS / "agent_profiles.json", encoding="utf-8"))
    name2code: dict[str, str] = {}
    for v in profiles.values():
        loc = v.get("location") or {}
        if loc.get("dong") and loc.get("adm_cd_8"):
            name2code[_canon(loc["dong"])] = loc["adm_cd_8"]

    html = _load_events_html()
    events = _grab_events(html)

    acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])  # code → [sum_lon, sum_lat, n]
    seen_poi: set[str] = set()
    for evs in events.values():
        for e in evs:
            lon, lat, dong = e.get("lon"), e.get("lat"), e.get("dong")
            poi = e.get("poi_id")
            if lon is None or lat is None or not dong:
                continue
            if poi in seen_poi:        # POI 1개당 1회만 (좌표 중복 가중 방지)
                continue
            seen_poi.add(poi)
            code = name2code.get(_canon(dong))
            if not code:
                continue
            a = acc[code]
            a[0] += float(lon); a[1] += float(lat); a[2] += 1

    centroids = {code: [round(s_lon / n, 6), round(s_lat / n, 6)]
                 for code, (s_lon, s_lat, n) in acc.items() if n > 0}

    # 자치구(5자리) 대체 중심 — 미커버 동의 fallback
    gu_acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    for code, (lon, lat) in centroids.items():
        g = gu_acc[code[:5]]
        g[0] += lon; g[1] += lat; g[2] += 1
    gu_fallback = {gu: [round(s_lon / n, 6), round(s_lat / n, 6)]
                   for gu, (s_lon, s_lat, n) in gu_acc.items() if n > 0}

    return {
        "_meta": {
            "source": "sim_standalone __EVENTS__ POI 좌표 평균 (POI 중복 제거)",
            "n_dong_centroids": len(centroids),
            "n_gu_fallback": len(gu_fallback),
            "n_poi_used": len(seen_poi),
        },
        "centroids": centroids,
        "gu_fallback": gu_fallback,
    }


def main() -> None:
    cat = build_centroids()
    out = STATS / "dong_centroids.json"
    out.write_text(json.dumps(cat, ensure_ascii=False, indent=1), encoding="utf-8")
    m = cat["_meta"]
    print(f"[dong_centroids] 동 중심 {m['n_dong_centroids']}개 / 자치구 fallback {m['n_gu_fallback']}개 "
          f"(POI {m['n_poi_used']}개) → {out}")

    # 허브 카탈로그 커버리지 점검
    hub_path = STATS / "hub_catalog.json"
    if hub_path.exists():
        hubs = json.load(io.open(hub_path, encoding="utf-8"))["hubs"]
        top = [h for h in hubs if h.get("is_top_hub")]
        cov_direct = sum(1 for h in top if h["code"] in cat["centroids"])
        cov_gu = sum(1 for h in top if h["code"] not in cat["centroids"]
                     and h["code"][:5] in cat["gu_fallback"])
        print(f"  상위허브 {len(top)}개 커버: 직접 {cov_direct} + 자치구fallback {cov_gu} "
              f"= {cov_direct + cov_gu}/{len(top)}")
        miss = [h["code"] for h in top if h["code"] not in cat["centroids"]
                and h["code"][:5] not in cat["gu_fallback"]]
        if miss:
            print(f"  미커버 허브 코드: {miss}")


if __name__ == "__main__":
    main()

"""허브 상권 성격(signature) 산출 — 페르소나별 차등 반응용 (Problem A 세분화).

각 광역상권 허브가 어떤 성격인지(쇼핑/미용/교육/건강/주점 등)를 viz POI의 L1 분포로
오프라인 추출한다. 서울 평균 대비 **가장 두드러진 카테고리**를 signature로 지정
(식사·편의점처럼 어디에나 흔한 건 제외). 런타임에서 나이·성별·소득 페르소나와 결합해
허브 매력도를 약하게 가감 → 압구정(쇼핑·미용)·대치(교육)·서초(건강)에 다른 사람이 끌리게.

입력: sim_standalone(viz)의 POI l1·dong, hub_catalog.json
출력: output/stats/hub_signature.json
      { code: {name, signature, l1_share:{L1:비율}, n_poi} }

성능: 전부 오프라인. 런타임 추가 쿼리·토큰 0.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STATS = REPO / "output" / "stats"
VIZ_ZIP = REPO / "output" / "sim" / "visualization" / "sim_standalone.zip"
VIZ_HTML = REPO / "output" / "sim" / "visualization" / "sim_standalone.html"

# 어디에나 흔해 변별력 없는 카테고리는 signature 후보에서 제외
UBIQUITOUS = {"식사", "편의점", "집", "직장", "기타", "마트"}
MIN_POI = 15          # 신뢰 가능한 최소 POI 수
MIN_DEVIATION = 0.03  # 서울 평균 대비 최소 초과분(3%p) 있어야 signature 부여


def _canon(s: str) -> str:
    return re.sub(r"제(?=\d)", "", s or "")


def _load_events() -> dict:
    html = (zipfile.ZipFile(VIZ_ZIP).read(
                [n for n in zipfile.ZipFile(VIZ_ZIP).namelist() if n.endswith(".html")][0]
            ).decode("utf-8")
            if VIZ_ZIP.exists() else io.open(VIZ_HTML, encoding="utf-8").read())
    for ln in html.split("\n"):
        if ln.startswith("window.__EVENTS__ = "):
            body = ln[len("window.__EVENTS__ = "):].rstrip()
            return json.loads(body[:-1] if body.endswith(";") else body)
    raise RuntimeError("window.__EVENTS__ 없음")


def build_signatures() -> dict:
    prof = json.load(io.open(STATS / "agent_profiles.json", encoding="utf-8"))
    name2code = {}
    for v in prof.values():
        loc = v.get("location") or {}
        if loc.get("dong") and loc.get("adm_cd_8"):
            name2code[_canon(loc["dong"])] = loc["adm_cd_8"]

    hubs = json.load(io.open(STATS / "hub_catalog.json", encoding="utf-8"))["hubs"]
    top_codes = {h["code"]: h["name"] for h in hubs if h.get("is_top_hub")}

    # dong → L1 카운트 (POI 중복 제거)
    seen = set()
    dong_l1: dict[str, Counter] = defaultdict(Counter)
    city_l1 = Counter()
    for evs in _load_events().values():
        for e in evs:
            poi, l1, dong = e.get("poi_id"), e.get("l1"), e.get("dong")
            if not poi or not l1 or not dong or poi in seen:
                continue
            seen.add(poi)
            code = name2code.get(_canon(dong))
            if code:
                dong_l1[code][l1] += 1
                city_l1[l1] += 1

    city_tot = sum(city_l1.values()) or 1
    city_share = {k: v / city_tot for k, v in city_l1.items()}

    out: dict[str, dict] = {}
    for code, name in top_codes.items():
        cnt = dong_l1.get(code)
        n = sum(cnt.values()) if cnt else 0
        if not cnt or n < MIN_POI:
            out[code] = {"name": name, "signature": "general", "n_poi": n, "l1_share": {}}
            continue
        share = {k: v / n for k, v in cnt.items()}
        # 변별 카테고리 중 서울 평균 초과분 최대
        best, best_dev = "general", MIN_DEVIATION
        for l1, sh in share.items():
            if l1 in UBIQUITOUS:
                continue
            dev = sh - city_share.get(l1, 0.0)
            if dev > best_dev:
                best, best_dev = l1, dev
        out[code] = {
            "name": name, "signature": best, "n_poi": n,
            "l1_share": {k: round(v, 3) for k, v in sorted(share.items(), key=lambda x: -x[1])[:5]},
        }
    return {"_meta": {"source": "viz POI l1 분포 vs 서울 평균",
                      "n_hubs": len(out), "city_share": {k: round(v, 3) for k, v in city_share.items()}},
            "hubs": out}


def main() -> None:
    sig = build_signatures()
    out = STATS / "hub_signature.json"
    out.write_text(json.dumps(sig, ensure_ascii=False, indent=1), encoding="utf-8")
    bysig = Counter(h["signature"] for h in sig["hubs"].values())
    print(f"[hub_signature] {len(sig['hubs'])}개 허브 → {out}")
    print(f"  signature 분포: {dict(bysig)}")
    for code, h in list(sig["hubs"].items()):
        if h["signature"] != "general":
            print(f"   {h['name']:<8} → {h['signature']}  {h['l1_share']}")


if __name__ == "__main__":
    main()

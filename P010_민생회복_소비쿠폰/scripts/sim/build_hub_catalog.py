"""광역 상권 허브 카탈로그 빌더 (Problem A 데이터 기반).

BDC `dong_context.json`의 b069_sales(행정동별 매출지수)를 상권 **매력도(attraction)**로
삼아 행정동을 랭킹한다. Stage1에서 거주/직장 동을 넘어 "오늘 끌리는 광역 상권" 후보를
주기 위한 정적 카탈로그. (Huff 중력모형의 A_j 항)

입력 (전부 로컬, Neo4j 불필요):
  output/stats/dong_context.json     — {dong8: {b069_sales, b069_store, b069_pop, ...}}
  output/stats/agent_profiles.json   — location {adm_cd_8, dong, gu}, demographics.population

출력:
  output/stats/hub_catalog.json
    {
      "_meta": {...},
      "hubs": [ {code, name, gu, attraction, attraction_norm, rank, is_top_hub,
                 store, pop, pop_resident}, ... ]   # 매력도 내림차순 전체 동
    }

사용:
  python build_hub_catalog.py [--top-k 40]
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STATS = REPO / "output" / "stats"


def _load(name: str) -> dict:
    return json.load(io.open(STATS / name, encoding="utf-8"))


def build_catalog(top_k: int = 40) -> dict:
    dong_ctx = _load("dong_context.json")
    profiles = _load("agent_profiles.json")

    # dong8 → (name, gu, 거주인구) 매핑 — agent_profiles.location 기준
    name_of: dict[str, str] = {}
    gu_of: dict[str, str] = {}
    pop_of: dict[str, float] = {}
    for v in profiles.values():
        loc = v.get("location") or {}
        code = loc.get("adm_cd_8")
        if not code:
            continue
        name_of.setdefault(code, loc.get("dong") or "")
        gu_of.setdefault(code, loc.get("gu") or "")
        pop_of[code] = pop_of.get(code, 0.0) + float((v.get("demographics") or {}).get("population", 0) or 0)

    # 매력도 = b069_sales (행정동 매출지수). 결측은 제외.
    rows: list[dict] = []
    for code, ctx in dong_ctx.items():
        if not isinstance(ctx, dict):
            continue
        sales = ctx.get("b069_sales")
        if sales is None:
            continue
        rows.append({
            "code": code,
            "name": name_of.get(code, ""),
            "gu": gu_of.get(code, ""),
            "attraction": round(float(sales), 4),
            "store": ctx.get("b069_store"),
            "pop": ctx.get("b069_pop"),
            "pop_resident": round(pop_of.get(code, 0.0), 1),
        })

    if not rows:
        raise RuntimeError("dong_context.json에 b069_sales가 없습니다.")

    rows.sort(key=lambda r: r["attraction"], reverse=True)
    amax = rows[0]["attraction"] or 1.0
    amin = rows[-1]["attraction"]
    span = (amax - amin) or 1.0
    for i, r in enumerate(rows):
        r["rank"] = i + 1
        # 0~1 정규화 (Huff 가중 직관용; 실제 Huff엔 raw attraction 사용 가능)
        r["attraction_norm"] = round((r["attraction"] - amin) / span, 4)
        r["is_top_hub"] = i < top_k

    return {
        "_meta": {
            "source": "BDC dong_context.b069_sales (행정동 매출지수) = Huff 중력모형의 매력도 A_j",
            "n_dongs": len(rows),
            "top_k": top_k,
            "attraction_max": amax,
            "attraction_min": amin,
            "note": "Stage1 광역 상권 후보 prior. 런타임 거리(home→hub)는 별도 Cypher로 결합.",
        },
        "hubs": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=40, help="상위 허브로 표시할 동 수")
    ap.add_argument("--out", default=str(STATS / "hub_catalog.json"))
    args = ap.parse_args()

    cat = build_catalog(top_k=args.top_k)
    out = Path(args.out)
    out.write_text(json.dumps(cat, ensure_ascii=False, indent=1), encoding="utf-8")

    hubs = cat["hubs"]
    print(f"[hub_catalog] {len(hubs)}개 동 랭킹 → {out}")
    print(f"  매력도(b069_sales) 범위: {cat['_meta']['attraction_min']} ~ {cat['_meta']['attraction_max']}")
    print(f"  상위 {min(args.top_k, 15)}개 허브:")
    for r in hubs[: min(args.top_k, 15)]:
        print(f"   #{r['rank']:>2} {r['gu']:<6} {r['name']:<10} attraction={r['attraction']:>7.2f} "
              f"(norm {r['attraction_norm']:.2f})  거주인구 {r['pop_resident']:,.0f}")


if __name__ == "__main__":
    main()

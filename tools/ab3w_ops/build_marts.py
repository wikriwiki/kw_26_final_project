"""P016 참여·비참여 마트 POI 목록(2026-10-10).
입력: 카카오맵 검색 결과(audit_tools/kakao_marts_20261010.json), 그래프의 기존 POI(bolt 7687).
출력: /data/ab3w/p016_prep/marts_20261010.json — 기존 POI 에 체인 표시를 붙일 것(tag)과 새로 넣을 것(add).
규칙(값을 지어내지 않는다):
  · 체인은 이름으로만 정한다(아래 BRANDS). 형식(hypermarket/ssm)은 카카오 분류(대형마트/대형슈퍼)를 따른다.
  · 2020-07-30 에 없던 것이 분명한 매장(이름이 2021년 뒤 새 이름·새 매장)은 opened_after_2020 으로 표시해 넣지 않는다.
  · 같은 체인 기존 POI 가 150m 안에 있으면 새로 넣지 않고 그 POI 에 표시만 붙인다.
  · 행정동은 가까운 기존 POI 5곳의 다수결로 정한다.
"""
import json, math, os, re, collections
from neo4j import GraphDatabase
BRANDS = [("emart", r"^이마트(?!24|에브리데이| ?에브리데이)"), ("emart_everyday", r"^이마트 ?에브리데이"),
          ("lottemart", r"^롯데마트"), ("lottesuper", r"^롯데슈퍼"), ("hanaro", r"하나로(마트|클럽)"),
          ("gs_fresh", r"^(GS ?더 ?프레시|GS ?수퍼|지에스더프레시|GS THE FRESH)"),
          ("homeplus", r"^홈플러스(?! ?익스프레스)"), ("homeplus_express", r"^홈플러스 ?익스프레스"),
          ("traders", r"^트레이더스"), ("nobrand", r"^노브랜드")]
# 2020-07-30 뒤에 생긴 매장·형식(이름으로 확인되는 것만). 개점 시점이 이름에 드러나지 않는 매장은 남기고 한계로 적는다.
AFTER_2020 = [r"푸드마켓 고덕", r"메가푸드마켓", r"그랑그로서리", r"제타플렉스", r"맥스 ", r"트레이더스 홀세일 클럽 마곡"]
PART_2020_1ST = {"emart", "lottemart", "hanaro", "gs_fresh"}   # 1차(7/30~8/9) 참여: 이마트·롯데마트·농협하나로마트·GS슈퍼(언론 보도)
def brand(name):
    for b, pat in BRANDS:
        if re.search(pat, name or ""): return b
def hav(a, b, c, d):
    R=6371000; p1,p2=math.radians(a),math.radians(c); dp=p2-p1; dl=math.radians(d-b)
    return 2*R*math.asin(math.sqrt(math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2))
kk = json.load(open("/data/ab3w/audit_tools/kakao_marts_20261010.json"))
stores = []
for v in kk.values():
    if not (v.get("address") or "").startswith("서울"): continue
    d2 = v.get("cate_name_depth2"); b = brand(v["name"])
    if not b or d2 not in ("대형마트", "대형슈퍼", "슈퍼마켓"): continue
    if "폐점" in v["name"] or "주차장" in v["name"]: continue
    fmt = "hypermarket" if d2 == "대형마트" else "ssm"
    stores.append({"kakao_id": v["confirmid"], "name": v["name"], "brand": b, "format": fmt,
                   "lat": float(v["lat"]), "lon": float(v["lon"]), "address": v.get("address"),
                   "after_2020": any(re.search(p, v["name"]) for p in AFTER_2020)})
drv = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", os.environ["NEO4J_PASSWORD_ON"]))
with drv.session() as s:
    g = s.run("MATCH (p:POI {type:'commerce'}) WHERE p.lat IS NOT NULL RETURN p.id AS id, p.name AS name, p.lat AS lat, p.lon AS lon, p.dong_code AS dong, head([(p)-[:IN_CATEGORY]->(c)|c.name]) AS sub").data()
drv.close()
gb = [dict(r, brand=brand((r["name"] or "").replace("주식회사", "").replace("(주)", ""))) for r in g]
grid = collections.defaultdict(list)
for r in g: grid[(round(r["lat"], 2), round(r["lon"], 2))].append(r)
def near(lat, lon, k=5):
    cand = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cand += grid.get((round(lat + dy * 0.01, 2), round(lon + dx * 0.01, 2)), [])
    return sorted(cand, key=lambda r: hav(lat, lon, r["lat"], r["lon"]))[:k]
tag, add, skip = [], [], []
same_brand = collections.defaultdict(list)
for r in gb:
    if r["brand"]: same_brand[r["brand"]].append(r)
for st in stores:
    hits = [r for r in same_brand[st["brand"]] if hav(st["lat"], st["lon"], r["lat"], r["lon"]) <= 150]
    if hits:
        h = min(hits, key=lambda r: hav(st["lat"], st["lon"], r["lat"], r["lon"]))
        tag.append({"poi_id": h["id"], "graph_name": h["name"], **st}); continue
    if st["after_2020"]:
        skip.append(st); continue
    nb = near(st["lat"], st["lon"])
    dong = collections.Counter(r["dong"] for r in nb if r["dong"]).most_common(1)
    add.append({**st, "poi_id": f"K_{st['kakao_id']}", "dong_code": dong[0][0] if dong else None,
                "nearest_m": round(hav(st["lat"], st["lon"], nb[0]["lat"], nb[0]["lon"])) if nb else None})
# 그래프에만 있는 같은 체인 POI 도 표시 대상(카카오에 안 잡힌 지점)
tagged_ids = {t["poi_id"] for t in tag}
graph_only = [r for r in gb if r["brand"] in ("hanaro", "gs_fresh", "emart", "lottemart") and r["sub"] == "슈퍼마켓" and r["id"] not in tagged_ids]
os.makedirs("/data/ab3w/p016_prep", exist_ok=True)
out = {"made": "2026-10-10", "source": "카카오맵 검색(scripts/scrape/kakao/client.py) + 그래프 bolt 7687",
       "participating_2020_1st": sorted(PART_2020_1ST), "tag_existing": tag, "add_new": add,
       "skipped_after_2020": skip, "graph_only_same_chain": [{"poi_id": r["id"], "name": r["name"], "brand": r["brand"]} for r in graph_only]}
json.dump(out, open("/data/ab3w/p016_prep/marts_20261010.json", "w"), ensure_ascii=False, indent=1)
c = collections.Counter((x["brand"], x["format"]) for x in add)
print("기존 POI 에 표시", len(tag), collections.Counter(t["brand"] for t in tag).most_common())
print("새로 넣을 것", len(add), c.most_common())
print("2020 뒤 매장이라 뺌", [x["name"] for x in skip])
print("그래프에만 있는 같은 체인(슈퍼마켓)", len(graph_only), collections.Counter(r["brand"] for r in graph_only).most_common())
print("새 POI 의 가장 가까운 기존 POI 거리(m) 최대", max((x["nearest_m"] or 0) for x in add) if add else None, "동 없음", sum(1 for x in add if not x["dong_code"]))

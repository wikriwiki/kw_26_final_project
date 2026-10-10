# 참여 매장 몫이 왜 작은가(읽기 전용). 정책 전 날들만. python p016_grocery_why.py <bolt> <roster> <첫날> <끝날> <정책 파일>
# 1) 장보기가 적은가: 페르소나(카드 실측) '할인점/슈퍼마켓' 몫 대 시뮬 장보기 몫
# 2) 후보에 참여 매장이 있었나: 엔진의 가까운 대형 형태 매장 3곳 규칙(장면 동 기준)을 그대로 재현 + 장면 동 안 참여 매장
# 3) 있었는데 안 골랐나: 무엇을 골랐나(같은 동 작은 가게 / 대형 형태 비참여 / 참여)
import json, os, re, sys, collections, statistics as st
from neo4j import GraphDatabase
bolt, roster_p, d0, d1, polp = sys.argv[1:6]
ids = json.load(open(roster_p))
pol = json.load(open(polp, encoding="utf-8"))
rx = re.compile(pol["eligibility"]["include"]["name_regex"]); subs = set(pol["eligibility"]["include"]["subs"])
Q_PAY = """
UNWIND $ids AS aid
MATCH (a:Agent {id: aid})-[h:HAS_PLAN]->(pl:Plan)-[i:INCLUDES]->(poi:POI)
WHERE h.day >= date($d0) AND h.day <= date($d1) AND coalesce(i.actual_spent, 0) > 0
RETURN aid, toString(h.day) AS day, i.category AS cat, i.sub_category AS plan_sub, i.anchor AS anchor,
       poi.name AS name, poi.id AS pid, poi.mart_format AS fmt, poi.dong_code AS pdong,
       head([(poi)-[:IN_CATEGORY]->(c:Category) | c.name]) AS poi_sub, i.actual_spent AS amt
"""
Q_PERSONA = "UNWIND $ids AS aid MATCH (a:Agent {id: aid}) RETURN aid, a.spending_top_wd_json AS top, a.p_spending_top_wd_json AS top2, keys(a) AS k"
Q_OFFER = """
MATCH (dg:Dong {code: $zone})
OPTIONAL MATCH (a:Agent {id: $aid})-[:LIVES_AT|WORKS_AT]->(anc:POI) WHERE anc.dong_code = $zone
WITH dg, head(collect(anc)) AS anc
WITH coalesce(anc.lon, dg.lon) AS olon, coalesce(anc.lat, dg.lat) AS olat
MATCH (p:POI) WHERE p.mart_format IN ['hypermarket','ssm'] AND p.type = 'commerce' AND p.lon IS NOT NULL
WITH p, point.distance(point({longitude: p.lon, latitude: p.lat}), point({longitude: olon, latitude: olat})) / 1000.0 AS km
WHERE km <= 3 RETURN p.name AS name, km ORDER BY km, p.id LIMIT 3
"""
Q_INDONG = """
MATCH (p:POI {type:'commerce'})-[:IN_DONG]->(:Dong {code: $zone}) MATCH (p)-[:IN_CATEGORY]->(c:Category) WHERE c.name IN $subs
RETURN p.name AS name
"""
d = GraphDatabase.driver(f"bolt://localhost:{bolt}", auth=("neo4j", os.environ["NEO4J_PASSWORD_ON"]))
with d.session() as s:
    pays = s.run(Q_PAY, ids=ids, d0=d0, d1=d1).data()
    pers = {r["aid"]: r for r in s.run(Q_PERSONA, ids=ids).data()}
    gro = [p for p in pays if p["poi_sub"] in subs]
    planned_gro = [p for p in pays if p["cat"] == "마트" or p["plan_sub"] in subs | {"장보기", "대형마트", "마트"}]
    offer_cache = {}
    rows = []
    for p in gro:
        zone = (p["anchor"] or "")[5:] if (p["anchor"] or "").startswith("zone:") else p["pdong"]
        key = (p["aid"], zone)
        if key not in offer_cache:
            near = [r["name"] for r in s.run(Q_OFFER, aid=p["aid"], zone=zone)]
            indong = [r["name"] for r in s.run(Q_INDONG, zone=zone, subs=list(subs))]
            offer_cache[key] = (any(rx.match(n or "") for n in near), sum(1 for n in indong if rx.match(n or "")), len(indong))
        rows.append((p, zone) + offer_cache[key])
d.close()
n_days = len({p["day"] for p in pays}) or 1
tot = sum(p["amt"] for p in pays); gamt = sum(p["amt"] for p in gro)
print(f"== {len(ids)}명 · {d0}~{d1}: 결제 {len(pays)}건 {tot:,.0f}원 · 장보기(가게 업종) {len(gro)}건 {gamt:,.0f}원 = 지출의 {100*gamt/max(1,tot):.1f}%")
print(f"   계획 단계에서 장보기로 잡힌 결제 {len(planned_gro)}건 (가게 업종이 장보기가 아닌 것 {sum(1 for p in planned_gro if p['poi_sub'] not in subs)})")
print(f"   사람-주당 장보기 {7*len(gro)/len(ids)/n_days:.2f}회")
# 페르소나 몫
shares = []
k0 = next(iter(pers.values()))["k"] if pers else []
for aid, r in pers.items():
    raw = r["top"] or r["top2"]
    try:
        top = json.loads(raw) if raw else {}
    except Exception:
        top = {}
    shares.append(float(top.get("할인점/슈퍼마켓", 0) or 0) + float(top.get("기타음/식료품", 0) or 0) + float(top.get("정육점", 0) or 0) + float(top.get("농수산물", 0) or 0))
if any(shares):
    print(f"   페르소나 카드 실측 '할인점/슈퍼마켓+식료품+정육+농수산' 몫 평균 {100*st.mean(shares):.1f}% (중앙 {100*st.median(shares):.1f}%)")
else:
    print("   페르소나 소비 구성 칸을 못 찾음. Agent 키 예:", [k for k in k0 if "spend" in k][:10])
# 후보와 선택
c = collections.Counter()
for p, zone, offered_near, indong_part, indong_n in rows:
    part = bool(rx.match(p["name"] or ""))
    offered = offered_near or indong_part > 0
    c["장보기"] += 1
    c["참여매장_후보에_있었음(재현)"] += offered
    c["골라진_참여매장"] += part
    c["후보에_있었는데_참여매장_안고름"] += offered and not part
    c["골라진_대형형태_비참여"] += (not part) and bool(p["fmt"])
    c["골라진_작은가게"] += (not part) and not p["fmt"]
    c["장면동_장보기가게수_중앙용"] += 0
print("   장보기 장면:", json.dumps(dict(c), ensure_ascii=False))
ind = [r[4] for r in rows]
if ind:
    print(f"   장면 동 안 장보기 업종 가게 수 중앙 {st.median(ind)} (후보 풀은 이 중 일부 + 가까운 대형 형태 3곳)")
names = collections.Counter(("참여 " if rx.match(p['name'] or '') else ("대형형태 " if p['fmt'] else "작은 ")) + (p["name"] or "") for p, *_ in rows)
print("   골라진 가게:", names.most_common(25))

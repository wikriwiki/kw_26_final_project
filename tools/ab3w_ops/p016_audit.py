# P016 시험 검수(읽기 전용). python3 p016_audit.py <bolt> <day_from> <day_to> [label]
import json, os, re, sys, collections
from neo4j import GraphDatabase
bolt, d0, d1 = sys.argv[1], sys.argv[2], sys.argv[3]
label = sys.argv[4] if len(sys.argv) > 4 else bolt
PW = os.environ["NEO4J_PASSWORD_ON"]
CH = re.compile(r"^(이마트(?!24| ?에브리데이)|롯데마트|롯데쇼핑롯데마트|.*하나로(마트|클럽)|GS ?더 ?프레시|지에스더프레시|GS ?수퍼|지에스리테일 ?GS ?수퍼)")
GRO = {"슈퍼마켓", "식료품", "청과", "정육", "장보기"}
drv = GraphDatabase.driver(f"bolt://localhost:{bolt}", auth=("neo4j", PW))
with drv.session() as s:
    rows = s.run("""
    MATCH (a:Agent)-[h:HAS_PLAN]->(p:Plan)-[i:INCLUDES]->(poi:POI)
    WHERE h.day >= date($d0) AND h.day <= date($d1)
    OPTIONAL MATCH (poi)-[:IN_CATEGORY]->(c:Category)
    RETURN a.id AS aid, toString(h.day) AS day, i.order AS o, i.category AS cat, i.sub_category AS sub,
           c.name AS pcat, poi.name AS name, poi.mart_format AS fmt, poi.source AS src,
           i.actual_spent AS spent, i.produce_spent AS prod, i.produce_share AS share,
           i.discount_total AS disc, i.instant_discount AS idisc, i.menu AS menu, i.pick_reason AS why
    """, d0=d0, d1=d1).data()
    mems = s.run("""
    MATCH (a:Agent)-[r:REMEMBERS]->(m:Memory {type:'visited'})
    WHERE r.day >= date($d0) AND r.day <= date($d1) AND (m.produce_spent IS NOT NULL OR coalesce(m.discount_total,0) > 0)
    RETURN a.id AS aid, toString(r.day) AS day, m.summary AS summary, m.store AS store LIMIT 12
    """, d0=d0, d1=d1).data()
    used = s.run("""
    MATCH (a:Agent)-[h:HAS_STATE]->(st:State) WHERE h.day >= date($d0) AND h.day <= date($d1)
    RETURN a.id AS aid, toString(h.day) AS day, st.policy_used AS used, st.instant_discount_today AS dt
    """, d0=d0, d1=d1).data()
drv.close()
print(f"===== {label} {d0}~{d1}: 결제 {len(rows)}건")
gro = [r for r in rows if (r["pcat"] in GRO or r["sub"] in GRO or r["cat"] == "마트")]
print(f"장보기(가게 업종이 장보기류 또는 사건 업종 마트): {len(gro)}건")
for r in gro:
    ch = bool(CH.match(r["name"] or ""))
    print(f"  {r['day']} {r['aid'][-14:]} [{r['cat']}/{r['sub']}→{r['pcat']}] {r['name']} fmt={r['fmt']} src={r['src']} 체인={ch} "
          f"결제={r['spent']} 농축산물={r['prod']} 몫={None if r['share'] is None else round(r['share'],2)} 할인={r['disc']} {r['idisc']} | {(r['menu'] or '')[:40]} | {(r['why'] or '')[:60]}")
nong = [r for r in rows if r["prod"] is not None]
print(f"produce_spent 채워진 결제: {len(nong)} · 장보기 아닌데 채워진 것: {sum(1 for r in nong if r not in gro)}")
over = [r for r in nong if (r["prod"] or 0) > (r["spent"] or 0)]
print(f"produce > 결제: {len(over)}")
disc = [r for r in rows if (r["disc"] or 0) > 0]
print(f"할인 결제 {len(disc)}건 · 체인 아닌데 할인: {sum(1 for r in disc if not CH.match(r['name'] or ''))}")
for r in disc:
    exp = round((r["spent"] or 0) * (r["share"] or 0) * 0.2)
    print(f"  {r['name']} 결제 {r['spent']} 농축산물 {r['prod']} 할인 {r['disc']} (20%×몫 {exp}) {r['idisc']}")
print("기억 문장:")
for m in mems:
    print(f"  {m['day']} {m['aid'][-14:]} [{m['store']}] {m['summary'][:160]}")
print("policy_used:")
for u in used:
    if u["used"] and u["used"] not in ("{}", None):
        print(f"  {u['day']} {u['aid'][-14:]} {u['used']} today={u['dt']}")

import json, os, sys
from neo4j import GraphDatabase
d = json.load(open("/data/repo_ab3w_20261011/data/experiments/p016_marts_20261010.json"))
rows = [{"poi_id": r["poi_id"], "name": r.get("name"), "brand": r.get("brand"), "format": r.get("format")} for r in d["tag_existing"]]
drv = GraphDatabase.driver(f"bolt://localhost:{sys.argv[1]}", auth=("neo4j", os.environ["NEO4J_PASSWORD_ON"]))
with drv.session() as s:
    out = s.run("""UNWIND $rows AS r MATCH (p:POI {id: r.poi_id}) WHERE p.mart_format IS NULL
      OPTIONAL MATCH (p)-[:IN_CATEGORY]->(c:Category)
      RETURN r.name AS kname, r.format AS fmt, p.name AS gname, p.type AS type, collect(c.name) AS cats, collect(c.parent) AS l1""", rows=rows).data()
    for o in out: print(o)
    print(s.run("MATCH (c:Category) WHERE c.name CONTAINS '마트' OR c.name CONTAINS '종합' OR c.name CONTAINS '백화' RETURN c.name, c.parent").data())
drv.close()

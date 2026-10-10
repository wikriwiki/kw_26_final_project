import json, os, sys
from neo4j import GraphDatabase
CH = r"^(이마트(?!24| ?에브리데이)|롯데마트|롯데쇼핑롯데마트|.*하나로(마트|클럽)|GS ?더 ?프레시|지에스더프레시|GS ?수퍼|지에스리테일 ?GS ?수퍼)"
drv = GraphDatabase.driver(f"bolt://localhost:{sys.argv[1]}", auth=("neo4j", os.environ["NEO4J_PASSWORD_ON"]))
with drv.session() as s:
    print(s.run("MATCH (c:Category {parent:'마트'}) OPTIONAL MATCH (p:POI)-[:IN_CATEGORY]->(c) RETURN c.name AS c, count(p) AS n ORDER BY n DESC").data())
    print(s.run("MATCH (p:POI)-[:IN_CATEGORY]->(c:Category {name:'종합소매'}) RETURN p.name AS n, p.mart_format AS f LIMIT 25").data())
    print(s.run("MATCH (p:POI)-[:IN_CATEGORY]->(c:Category) WHERE p.name =~ $ch RETURN c.name AS c, count(*) AS n ORDER BY n DESC", ch=CH).data())
drv.close()
d = json.load(open("/data/repo_ab3w_20261011/data/experiments/p016_marts_20261010.json"))
print(sorted(d["tag_existing"][0].keys()), sorted(d["add_new"][0].keys()))

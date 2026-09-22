"""agents_final.json 의 업종 비중을 그래프에 채워 넣는다.

덤프 복원본에는 cat_ratio_wd / cat_ratio_we 가 없다. 04_agents.py 를 다시
돌리면 WORKS_AT 앵커가 날아가므로, 이 필드만 덧쓴다.

**원본은 agents_final.json.bak_orig 다.** 현행 agents_final.json 은 15,000명
0-index 의 다른 세대로, ID 가 한 칸 밀려 있어 그대로 조인하면 전부 남의 값이
붙는다. probe(소비분위·나이·성별) 대조가 통과할 때만 적재한다.
"""
import json, os, sys
from neo4j import GraphDatabase

SRC = os.environ.get("CATRATIO_SRC", "/data/agent_catratio.jsonl")
URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
USER = os.environ.get("NEO4J_USER", "neo4j")
PW = os.environ.get("NEO4J_PASSWORD", "exp001pass")

CYPHER = """
UNWIND $batch AS a
MATCH (n:Agent {id: a.id})
SET n.cat_ratio_wd = a.wd,
    n.cat_ratio_we = a.we,
    n.spending_top_wd_json = a.wd,
    n.spending_top_we_json = a.we
RETURN count(n) AS c
"""

def main() -> int:
    rows = []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"입력 {len(rows):,}건", flush=True)

    drv = GraphDatabase.driver(URI, auth=(USER, PW))
    total = 0
    with drv.session() as s:
        for i in range(0, len(rows), 2000):
            chunk = rows[i:i + 2000]
            total += s.execute_write(
                lambda tx, b=chunk: tx.run(CYPHER, batch=b).single()["c"])
            print(f"  {total:,}/{len(rows):,}", flush=True)

        have = s.run(
            "MATCH (n:Agent) WHERE n.cat_ratio_wd IS NOT NULL "
            "RETURN count(n) AS c").single()["c"]
        allc = s.run("MATCH (n:Agent) RETURN count(n) AS c").single()["c"]
    drv.close()
    print(f"적재 {total:,} / 보유 {have:,} / 전체 {allc:,}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

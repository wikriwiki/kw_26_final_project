"""P007 정책 적재 + P001 비활성화 — WSL 내부에서 실행 권장."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import driver_session

POLICY_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "neo4j_load" / "policies" / "P007.json"


def main():
    with open(POLICY_PATH, encoding="utf-8") as f:
        pol = json.load(f)
    print(f"[load_p007] spec = {pol['name']}")

    with driver_session() as s:
        # 1. P001 비활성화 (시뮬 기간 5/01~ 전으로 effective_until 당김)
        r = s.run("""
            MATCH (p:Policy {id:'P001'})
            SET p.effective_until = date('2026-04-30'),
                p.notes = coalesce(p.notes,'') + ' [DEACTIVATED for P007 sim]'
            RETURN toString(p.effective_until) AS u
        """).single()
        print(f"  P001 비활성: until={r['u']}" if r else "  P001 노드 없음")

        # 2. P007 Policy 노드 MERGE
        s.run("""
            MERGE (p:Policy {id:$id})
            SET p.name=$name, p.type=$type, p.description=$desc,
                p.benefit_rate=$rate, p.cap_per_agent=$cap,
                p.announce_date=date($announce),
                p.effective_from=date($from_), p.effective_until=date($until_)
        """,
            id=pol["id"], name=pol["name"], type=pol["type"], desc=pol["description"],
            rate=pol["benefit_rate"], cap=pol["cap_per_agent"],
            announce=pol["announce_date"], from_=pol["effective_from"],
            until_=pol["effective_until"],
        )
        print(f"  P007 Policy 노드 MERGE 완료")

        # 3. applied_to: 25 districts
        r = s.run("""
            MATCH (p:Policy {id:'P007'})
            MATCH (d:District) WHERE d.name IN $names
            MERGE (p)-[:applied_to]->(d)
            RETURN count(DISTINCT d) AS n
        """, names=pol["target_districts"]).single()
        print(f"  P007 applied_to → {r['n']} districts")

        # 4. 검증: 5/01 시점 활성 정책
        print()
        print("=== 활성 정책 (2026-05-01 기준) ===")
        for r in s.run("""
            MATCH (p:Policy)
            WHERE date('2026-05-01') >= p.effective_from
              AND date('2026-05-01') <= p.effective_until
            OPTIONAL MATCH (p)-[:applied_to]->(t)
            RETURN p.id AS id, p.name AS name, p.benefit_rate AS rate,
                   p.cap_per_agent AS cap, count(DISTINCT t) AS reg_n,
                   toString(p.effective_from) AS fr, toString(p.effective_until) AS un
            ORDER BY p.id
        """):
            print(f"  {r['id']}: {r['name']}")
            print(f"    rate={r['rate']} cap={r['cap']} regions={r['reg_n']} | {r['fr']}~{r['un']}")


if __name__ == "__main__":
    main()

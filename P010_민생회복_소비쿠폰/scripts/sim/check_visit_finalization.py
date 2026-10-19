"""Integration check in one ALWAYS-ROLLED-BACK Neo4j transaction."""
import uuid
from plan_writer import NIGHT_VISITED_CYPHER, driver_session


def main():
    marker="validation_check_"+uuid.uuid4().hex
    with driver_session() as session:
        tx=session.begin_transaction()
        try:
            tx.run("""
                CREATE (a:Agent {id:$id}), (p:Plan {id:$id}), (poi:POI {id:$id})
                CREATE (a)-[:HAS_PLAN {day:date('2020-01-01')}]->(p)
                CREATE (p)-[:INCLUDES {order:1,anchor:'zone:11680521',category:'카페',
                    actual_satisfaction:0.6,actual_spent:10}]->(poi)
                CREATE (a)-[:KNOWS_POI {visit_count:2,avg_satisfaction:0.2}]->(poi)
                """,id=marker).consume()
            first=tx.run(NIGHT_VISITED_CYPHER,aid=marker,yesterday="2020-01-01",ingest_token=uuid.uuid4().hex).single()
            before=dict(tx.run("MATCH (:Agent {id:$id})-[k:KNOWS_POI]->(:POI {id:$id}) RETURN k.visit_count AS n,k.avg_satisfaction AS avg",id=marker).single())
            second=tx.run(NIGHT_VISITED_CYPHER,aid=marker,yesterday="2020-01-01",ingest_token=uuid.uuid4().hex).single()
            after=dict(tx.run("MATCH (:Agent {id:$id})-[k:KNOWS_POI]->(:POI {id:$id}) RETURN k.visit_count AS n,k.avg_satisfaction AS avg",id=marker).single())
            assert first["n_memories"]==1 and second["n_memories"]==0
            assert before==after and after["n"]==3
            assert abs(after["avg"]-1/3)<1e-9, after
        finally:
            tx.rollback()
        assert session.run("MATCH (n {id:$id}) RETURN count(n) AS n",id=marker).single()["n"]==0
    print("PASS: visit finalization is idempotent; average reconciles; temporary graph rolled back.")


if __name__=="__main__": main()

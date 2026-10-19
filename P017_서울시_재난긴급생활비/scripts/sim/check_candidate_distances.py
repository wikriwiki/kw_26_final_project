"""Live Neo4j regression check using only a transaction that is always rolled back."""
import uuid
from dawn_context import STAGE2_CANDIDATE_CYPHER
from neo4j_load._common import driver_session


def main():
    token='CODEX_CANDIDATE_'+uuid.uuid4().hex
    with driver_session() as session:
        tx=session.begin_transaction()
        try:
            tx.run('''CREATE (a:Agent {id:$aid}), (h:POI {id:$home,lon:127.0,lat:37.0}),
                (w:POI {id:$work,lon:127.0,lat:37.1}), (d:Dong {code:$dong}),
                (c:Category {name:$category}),
                (p:POI {id:$pa,type:'commerce',name:'A',lon:127.0,lat:37.001}),
                (q:POI {id:$pb,type:'commerce',name:'B',lon:127.0,lat:37.102}),
                (a)-[:LIVES_AT]->(h), (a)-[:WORKS_AT]->(w),
                (p)-[:IN_DONG]->(d), (q)-[:IN_DONG]->(d),
                (p)-[:IN_CATEGORY]->(c), (q)-[:IN_CATEGORY]->(c)''',
                aid=token,home=token+'_H',work=token+'_W',dong=token+'_D',category=token+'_CAT',pa=token+'_A',pb=token+'_B').consume()
            kwargs=dict(aid=token,dong_code=token+'_D',sub_category=token+'_CAT',limit=12)
            rows=[dict(r) for r in tx.run(STAGE2_CANDIDATE_CYPHER,**kwargs)]
            assert len(rows)==2 and len({r['poi_id'] for r in rows})==2, rows
            assert all(not r['known'] and r['km'] is not None for r in rows), rows
            assert [r['poi_id'] for r in rows]==[token+'_A',token+'_B'], rows
            assert .10 < rows[0]['km'] < .12 and .21 < rows[1]['km'] < .24, rows
            assert rows==[dict(r) for r in tx.run(STAGE2_CANDIDATE_CYPHER,**kwargs)]
        finally:
            tx.rollback()
        assert session.run('MATCH (a:Agent {id:$aid}) RETURN count(a) AS n',aid=token).single()['n']==0
    print('PASS: unknown POIs retain anchor distance, multiple anchors do not duplicate candidates, stable ordering, transaction rolled back.')


if __name__=='__main__': main()

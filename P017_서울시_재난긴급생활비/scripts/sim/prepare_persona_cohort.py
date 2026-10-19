"""Read-only, disjoint-person cohort preparation; no prompt or model execution."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from validate_prompt_v3 import atomic


def select(population, excluded, n, seed):
    pop={int(d):sorted(set(ids)-set(excluded)) for d,ids in population.items()}
    total=sum(map(len,pop.values()))
    if n<=0 or n>total:raise ValueError('Invalid cohort size')
    flat=[aid for ids in pop.values() for aid in ids]
    if len(flat)!=len(set(flat)):raise ValueError('Citizen belongs to multiple strata')
    quota={d:len(ids)*n//total for d,ids in pop.items()}
    order=sorted(pop,key=lambda d:(-(len(pop[d])*n/total-quota[d]),d))
    for d in order[:n-sum(quota.values())]:quota[d]+=1
    chosen=sorted(a for d in sorted(pop) for a in random.Random(f'{seed}-{d}').sample(pop[d],quota[d]))
    return chosen,quota


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--exclude',type=Path,action='append',required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--n',type=int,required=True);ap.add_argument('--seed',required=True)
    args=ap.parse_args()
    if args.out.exists():raise ValueError('Refusing to overwrite frozen cohort')
    excluded=set();sources={}
    for p in args.exclude:
        raw=p.read_bytes();sources[str(p)]=hashlib.sha256(raw).hexdigest()
        excluded.update(v['id'] for v in json.loads(raw)['personas'])
    from neo4j_load._common import driver_session
    from dawn_context import PERSONA_CYPHER
    with driver_session() as session:
        rows=list(session.run('MATCH (a:Agent) WHERE (a)-[:LIVES_AT]->() RETURN coalesce(a.spending_level_wd,0) AS d,collect(a.id) AS ids'))
        ids,quota=select({int(r['d']):r['ids'] for r in rows},excluded,args.n,args.seed)
        personas=[]
        for aid in ids:
            record=session.run(PERSONA_CYPHER,aid=aid).single()
            if record is None:raise ValueError('Missing sampled citizen')
            personas.append(dict(record))
    if {p['id'] for p in personas}&excluded:raise ValueError('Cohort overlap')
    atomic(args.out,{'personas':personas,'selection':{'seed':args.seed,'quota':quota,'excluded_ids':sorted(excluded),
        'source_sha256':sources,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'prepared_at':datetime.now(timezone.utc).isoformat(),
        'scope':'Disjoint-person cohort, same known policy mechanisms. No LLM calls. Future input preparation and candidate freeze required before confirmation.'}})
    print(json.dumps({'people':len(personas),'excluded_people':len(excluded),'overlap':0,'sha256':hashlib.sha256(args.out.read_bytes()).hexdigest()}))


if __name__=='__main__':main()

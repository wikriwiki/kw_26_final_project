"""다각도 정밀 점검 — 표면 cron 안 잡히는 잠재 이슈."""
import json, sys, glob
from collections import Counter, defaultdict
sys.path.insert(0, 'scripts/neo4j_load')
from _common import driver_session

print("=" * 70)
print("9d baseline 다각도 정밀 점검")
print("=" * 70)

# 1) Day별 분포 일관성 — n_events·n_includes·소비액 분포가 day간 비슷한가
print("\n[1] Day별 분포 일관성 (안정성)")
DAYS = ['2026-05-25', '2026-05-26', '2026-05-27']
for d in DAYS:
    fp = f'C:/Users/Administrator/sim_output_9d/metrics/day_{d}.jsonl'
    if not __import__('os').path.exists(fp):
        continue
    rows = []
    with open(fp, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: rows.append(json.loads(line))
    ok = [r for r in rows if r.get('status') == 'ok']
    nev = [r.get('n_events', 0) for r in ok]
    ninc = [r.get('n_includes', 0) for r in ok]
    bal = [r.get('balance') for r in ok if r.get('balance') is not None]
    tin = [r.get('tokens_in', 0) for r in ok]
    elap = [r.get('elapsed', 0) for r in ok]
    s1a = [r.get('s1_attempts', 0) for r in ok]
    s2a = [r.get('s2_attempts', 0) for r in ok]
    print(f"  {d}: ok={len(ok)} | n_events avg={sum(nev)/len(nev):.1f} | n_includes avg={sum(ninc)/len(ninc):.1f}", end='')
    if bal: print(f" | balance avg={sum(bal)/len(bal):,.0f}", end='')
    print(f" | tokens_in avg={sum(tin)/len(tin):.0f} | elapsed avg={sum(elap)/len(elap):.1f}s")
    if s1a:
        s1_retry = sum(1 for x in s1a if x > 1)
        s2_retry = sum(1 for x in s2a if x > 1)
        print(f"           Stage1 retry>1: {s1_retry}, Stage2 retry>1: {s2_retry}")

# 2) State 연속성 — balance 추세, month_spent 누적, 음수 잔액
print("\n[2] State 연속성")
with driver_session() as s:
    for d in DAYS + ['2026-05-24']:
        r = s.run("""
            MATCH (st:State) WHERE st.day = date($d)
            RETURN count(st) AS n,
                   min(st.balance) AS bmin, max(st.balance) AS bmax, avg(st.balance) AS bavg,
                   sum(CASE WHEN st.balance < 0 THEN 1 ELSE 0 END) AS negative,
                   avg(st.month_spent) AS msavg, max(st.month_spent) AS msmax
        """, d=d).single()
        if r['n'] > 0:
            print(f"  {d}: State {r['n']}, balance min={r['bmin']:,} max={r['bmax']:,} avg={r['bavg']:,.0f}, 음수 잔액 {r['negative']}명, month_spent avg={r['msavg']:,.0f} max={r['msmax']:,}")

# 3) Conversation·Memory 일관성 — Day3 짧은 잔재 식별
print("\n[3] Conversation 분포 — Day3 짧은 잔재 식별")
with driver_session() as s:
    # Conversation 속성
    ck = s.run("MATCH (c:Conversation) RETURN keys(c) AS k LIMIT 1").single()
    if ck:
        print(f"  Conversation keys: {ck['k']}")
    # day 속성으로 분리 가능?
    by_day = s.run("MATCH (c:Conversation) RETURN c.day AS day, count(c) AS n ORDER BY day").data()
    if by_day and by_day[0].get('day'):
        for r in by_day:
            print(f"  {r['day']}: {r['n']} conv")
    else:
        # day 속성 없으면 시간 기반?
        ts = s.run("MATCH (c:Conversation) RETURN c.created_at AS t LIMIT 3").data()
        print(f"  day 속성 없음. 샘플 created_at: {[r.get('t') for r in ts]}")

# 4) Memory·KNOWS_POI 일별 갱신
with driver_session() as s:
    print("\n[4] Memory·KNOWS_POI 일별 갱신")
    for d in DAYS:
        mr = s.run("MATCH (m:Memory) WHERE m.day = date($d) RETURN count(m) AS n", d=d).single()
        if mr:
            print(f"  {d}: Memory 신규 {mr['n']}")

# 5) Day별 카테고리·소비 분포 (상위 5)
print("\n[5] Day별 commerce 카테고리 상위 5 (안정성)")
with driver_session() as s:
    for d in DAYS:
        rows = s.run("""
            MATCH (:Plan {day: date($d)})-[i:INCLUDES]->()
            WHERE i.category IS NOT NULL
            WITH i.category AS cat, count(i) AS n, avg(i.actual_spent) AS sp
            RETURN cat, n, sp ORDER BY n DESC LIMIT 5
        """, d=d).data()
        print(f"  {d}:", [(r['cat'], r['n'], f"{r['sp']:,.0f}원") for r in rows])

# 6) 시스템 리소스
print("\n[6] 시스템 리소스")
import subprocess
try:
    out = subprocess.run(['wsl.exe', '-e', 'bash', '-c', 'free -h | head -3'], capture_output=True, text=True, timeout=10).stdout
    print("  WSL:", out.strip().split('\n')[1] if out else 'N/A')
except Exception as e:
    print(f"  WSL: {e}")
try:
    out = subprocess.run(['nvidia-smi.exe', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True, timeout=10).stdout
    print(f"  GPU: {out.strip()}")
except Exception as e:
    print(f"  GPU: {e}")

# 7) Neo4j 헬스
print("\n[7] Neo4j 헬스")
with driver_session() as s:
    r = s.run("CALL dbms.queryJmx('java.lang:type=Memory') YIELD attributes RETURN attributes.HeapMemoryUsage.value.value AS heap").single()
    if r:
        heap = r['heap']
        print(f"  Heap usage: used={heap.get('used',0)/1e9:.2f}GB / max={heap.get('max',0)/1e9:.2f}GB")

# 8) vLLM
print("\n[8] vLLM")
try:
    import urllib.request
    text = urllib.request.urlopen('http://localhost:8000/metrics', timeout=5).read().decode()
    for ln in text.split('\n'):
        if 'prefix_cache_queries_total{' in ln or 'prefix_cache_hits_total{' in ln or 'num_requests_running' in ln or 'num_requests_waiting' in ln:
            print(f"  {ln[:140]}")
except Exception as e:
    print(f"  vLLM metrics: {e}")

"""정책 전 기준액 지도 {aid: 하루 소비 기준액} — 소득분위(P013 H1·H2)를 결과가 아니라 정책 전 속성으로 가른다.

    NEO4J_URI=... python tools/ab3w_anchor_map.py <roster.json> <out.json>
옛 P013 러너(tools/run_p013m_20261003.sh)와 같은 정의: (5 x 평일 기준액 + 2 x 주말 기준액) / 7, 한쪽이 비면 다른 쪽.
"""
import json
import sys

from scripts.neo4j_load._common import driver_session

ids = json.load(open(sys.argv[1], encoding="utf-8"))
with driver_session() as s:
    rows = [dict(r) for r in s.run(
        "MATCH (a:Agent) WHERE a.id IN $ids RETURN a.id AS aid, a.s_daily_wd AS wd, a.s_daily_we AS we", ids=ids)]
out = {}
for r in rows:
    wd = float(r["wd"] or 0)
    we = float(r["we"] or 0)
    if wd <= 0:
        wd = we
    if we <= 0:
        we = wd
    out[r["aid"]] = round((5 * wd + 2 * we) / 7) if wd > 0 else 0
missing = [a for a in ids if a not in out]
if missing:
    raise SystemExit(f"그래프에 없는 사람 {len(missing)}명: {missing[:3]}")
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
print("기준액 지도 %d명 · 기준액 없음 %d명" % (len(out), sum(1 for v in out.values() if v <= 0)))

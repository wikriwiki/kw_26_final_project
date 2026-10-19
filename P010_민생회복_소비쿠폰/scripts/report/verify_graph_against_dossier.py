"""지금 올라와 있는 그래프가 dossier 와 **사람 단위로 같은지** 대조한다.

    python scripts/report/verify_graph_against_dossier.py \
        --dossier <arm>/dossier.jsonl --sample 30

## 왜 필요한가

덤프의 체크섬은 "파일이 변하지 않았다"만 증명한다. 그 파일 안에 정말로 각 에이전트의
기억·계획·상태가 들어 있는지는 **복원해서 읽어 봐야** 안다. 이 도구는 덤프를 복원한
직후 돌려, 무작위로 뽑은 에이전트의 기억 수·계획 항목 수·상태 날짜가 dossier 와
정확히 같은지 본다. 같으면 두 가지가 함께 증명된다 — 덤프가 복원 가능하고, dossier 가
그 그래프를 빠짐없이 옮겼다.

뽑는 규칙은 seed 로 고정한다. 기억이 많은 사람만 고르지 않도록 기억 수 분포 전체에서
고르게 뽑는다(정렬 후 등간격).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

Q = """
MATCH (a:Agent {id: $aid})
OPTIONAL MATCH (a)-[:REMEMBERS]->(m:Memory)
  WHERE m.day >= date($start) AND m.day <= date($end)
WITH a, count(m) AS mem
OPTIONAL MATCH (a)-[:HAS_PLAN]->(p:Plan)-[:INCLUDES]->(:POI)
  WHERE p.day >= date($start) AND p.day <= date($end)
WITH a, mem, count(p) AS items
OPTIONAL MATCH (a)-[:HAS_STATE]->(s:State)
  WHERE s.day >= date($start) AND s.day <= date($end)
RETURN mem, items, collect(DISTINCT toString(s.day)) AS days
"""


def load(p: str) -> list[dict]:
    out = []
    with io.open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def expected(rec: dict) -> dict:
    return {
        "mem": len(rec.get("memories") or []),
        "items": sum(len(x.get("items") or []) for x in rec.get("plans") or []),
        "days": sorted(x["day"] for x in rec.get("states") or []),
    }


def sample(recs: list[dict], n: int) -> list[dict]:
    """기억 수로 정렬해 등간격으로 — 많은 사람만 고르지 않는다."""
    if n >= len(recs):
        return list(recs)
    ordered = sorted(recs, key=lambda r: (len(r.get("memories") or []), r["aid"]))
    step = (len(ordered) - 1) / max(1, n - 1)
    return [ordered[round(i * step)] for i in range(n)]


def compare(recs: list[dict], window: tuple[str, str], query) -> list[dict]:
    """query(aid, start, end) -> {'mem','items','days'} . 어긋난 사람 목록을 돌려준다."""
    bad = []
    for r in recs:
        want = expected(r)
        got = query(r["aid"], *window)
        got = {"mem": int(got["mem"]), "items": int(got["items"]),
               "days": sorted(got["days"])}
        if got != want:
            bad.append({"aid": r["aid"], "dossier": want, "graph": got})
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dossier", required=True)
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()
    recs = load(a.dossier)
    if not recs:
        raise SystemExit("dossier 가 비어 있다")
    w = recs[0].get("window") or [None, None]
    picked = sample(recs, a.sample)

    from neo4j import GraphDatabase
    drv = GraphDatabase.driver(os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                               auth=(os.environ.get("NEO4J_USER", "neo4j"),
                                     os.environ.get("NEO4J_PASSWORD", "")))
    try:
        with drv.session() as s:
            def query(aid, start, end):
                r = s.run(Q, aid=aid, start=start, end=end).single()
                return {"mem": r["mem"], "items": r["items"], "days": r["days"]}
            bad = compare(picked, (w[0], w[1]), query)
    finally:
        drv.close()

    print("# 복원한 그래프 vs dossier — %s" % a.dossier)
    print("  창 %s ~ %s · 대조 %d명 (기억 수 분포 등간격)" % (w[0], w[1], len(picked)))
    print("  일치 **%d명** · 불일치 **%d명**" % (len(picked) - len(bad), len(bad)))
    for b in bad[:5]:
        print("     %s dossier=%s graph=%s" % (b["aid"], b["dossier"], b["graph"]))
    if a.json_out:
        io.open(a.json_out, "w", encoding="utf-8", newline="\n").write(json.dumps(
            {"dossier": a.dossier, "window": w, "sampled": [r["aid"] for r in picked],
             "mismatch": bad, "ok": not bad}, ensure_ascii=False, indent=1))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())

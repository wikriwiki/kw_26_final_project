"""Night2 쌍이 적재에서 빠지는 이유를 **런 코드를 건드리지 않고** 찾는다.

    python tools/diagnose_night2_pairs.py --day 2021-10-01

## 왜 따로 있는가

Night2 는 계획한 쌍 수와 적재된 대화 수가 하나라도 다르면 하루를 버린다
(라이브: 1,547 계획 · 1,546 적재 → 3시간치 하루가 죽었다). 적재 Cypher 는
`MATCH (a:Agent) MATCH (b:Agent)` 라서 에이전트가 안 맞는 행을 **조용히 버린다.**

원인을 보려고 런 코드에 진단을 넣으면 실행 지문이 바뀌어 이어하기가 거부된다 —
그 관문은 옳다. 그래서 같은 쌍 선택을 **읽기 전용으로 따로** 돌려 본다. 그래프가
Night2 직전 상태라면(부분 적재를 지운 뒤라면) 다음 시도가 만들 쌍과 같다.

## 무엇을 보는가

  · 쌍의 두 사람이 모두 `:Agent` 로 있는가 — 없으면 그 행은 버려진다
  · 같은 쌍이 두 번 있는가 (a,b)=(a,b) 또는 (a,b)=(b,a)
  · 자기 자신과의 쌍 a==b
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
sys.path.insert(0, str(ROOT / "scripts" / "neo4j_load"))
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from night_interaction import select_interaction_pairs
    from _common import driver_session

    day = date.fromisoformat(a.day)
    with driver_session() as s:
        existing = s.run("MATCH (c:Conversation) WHERE c.day = date($d) RETURN count(c) AS n",
                         d=a.day).single()["n"]
    print("# Night2 쌍 진단 — %s" % a.day)
    print()
    if existing:
        print("  ** 이 날 대화가 이미 %d건 있다. 부분 적재를 지운 뒤 돌려야 다음 시도와 같은"
              " 쌍이 나온다. **" % existing)
        print()

    pairs = select_interaction_pairs(day, verbose=False)
    print("  계획 쌍 %d" % len(pairs))
    ids = set()
    for p in pairs:
        ids.add(str(p["aid_a"]))
        ids.add(str(p["aid_b"]))
    with driver_session() as s:
        found = {r["id"] for r in s.run(
            "MATCH (a:Agent) WHERE a.id IN $ids RETURN a.id AS id", ids=sorted(ids))}
    missing = sorted(ids - found)
    self_pairs = [p for p in pairs if p["aid_a"] == p["aid_b"]]
    ordered = Counter((p["aid_a"], p["aid_b"]) for p in pairs)
    unordered = Counter(tuple(sorted((p["aid_a"], p["aid_b"]))) for p in pairs)
    dup_o = {k: v for k, v in ordered.items() if v > 1}
    dup_u = {k: v for k, v in unordered.items() if v > 1}
    bad_rows = [p for p in pairs if p["aid_a"] in missing or p["aid_b"] in missing]

    print("  쌍에 나온 사람 %d · 그래프에 없는 사람 **%d**" % (len(ids), len(missing)))
    for x in missing[:10]:
        print("     없음: %r" % x)
    print("  그 사람 때문에 버려질 행 **%d**" % len(bad_rows))
    print("  자기 자신과의 쌍 %d" % len(self_pairs))
    print("  같은 순서쌍 중복 %d · 순서 무시 중복 %d" % (len(dup_o), len(dup_u)))
    for k, v in list(dup_u.items())[:5]:
        print("     중복: %s x%d" % (k, v))
    print()
    if bad_rows:
        print("  → 원인: 그래프에 없는 에이전트가 쌍에 들어온다. 쌍 선택이 어디서 사람을")
        print("    읽는지(fetch_all) 확인하라 — 명부 밖이거나 id 표기가 다를 수 있다.")
    elif dup_o:
        print("  → 원인 후보: 같은 쌍이 두 번 계획된다. 적재 id 는 행마다 새로 만들므로")
        print("    이것만으로는 대화가 줄지 않는다 — 다른 원인도 함께 본다.")
    else:
        print("  → 이 쌍 집합에서는 빠질 행이 없다. 다음 시도는 통과해야 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

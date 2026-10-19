"""3주 A/B 실행기: 정책 전 주로 만든 계획 기준선이 명부의 몇 %를 덮는지. 90% 미만이면 멈춘다.

기준선이 없는 사람은 두 갈래 모두 옛 공식(평소 소비와 계획 중 큰 값)으로 돈다 — 짝은 유지되지만
계획 통로를 거치지 않으므로, 그 비율을 기록해 둔다.
"""
import json
import sys

base = json.load(open(sys.argv[1], encoding="utf-8"))
ids = json.load(open(sys.argv[2], encoding="utf-8"))
have = sum(1 for a in ids if a in base)
print("  기준선이 있는 사람 %d/%d (%.1f%%)" % (have, len(ids), 100 * have / len(ids)))
# 빠져도 되는 사람 수: 10% (명부가 작으면 최소 1명) — 10명 시험에서 한 명 때문에 멈추지 않게.
allowed = max(1, int(0.1 * len(ids)))
if len(ids) - have > allowed:
    raise SystemExit("기준선이 없는 사람이 %d명(허용 %d명) — 정책 전 주 출력을 먼저 본다" % (len(ids) - have, allowed))

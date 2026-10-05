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
if have < 0.9 * len(ids):
    raise SystemExit("기준선이 90% 미만이다 — 정책 전 주 출력을 먼저 본다")

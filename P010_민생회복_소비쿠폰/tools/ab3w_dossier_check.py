"""3주 A/B 실행기: 기억 모음(사람 단위 상태·기억·계획·지출)에 빠진 날이 없는지. 하나라도 빠지면 멈춘다."""
import json
import sys

m = json.load(open(sys.argv[1], encoding="utf-8"))
days = int(sys.argv[2])
assert m["expected_days"] == days and not m["state_day_gaps"], (m["expected_days"], m["state_day_gaps"][:1])
assert m["totals"]["states"] == m["agents"] * m["expected_days"], m["totals"]
assert m["totals"]["memories"] > 0 and m["totals"]["plan_items"] > 0, m["totals"]
print("  기억 모음 %d명 · 상태 %d · 기억 %d · 계획항목 %d"
      % (m["agents"], m["totals"]["states"], m["totals"]["memories"], m["totals"]["plan_items"]))

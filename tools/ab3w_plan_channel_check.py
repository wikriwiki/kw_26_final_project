"""3주 A/B 실행기: 두 갈래에서 계획 통로가 실제로 켜졌는지 — 소비가 있는 사람-날의 90% 이상에 기준선이 쓰였어야 한다.

consumption._plan_baseline() 은 파일이 없거나 못 읽으면 조용히 빈 표가 되어 옛 공식으로 돈다. 그것을 여기서 잡는다.
"""
import glob
import json
import sys

for d in sys.argv[1:]:
    rows = [json.loads(line) for f in sorted(glob.glob(d + "/day_*.jsonl"))
            for line in open(f, encoding="utf-8")]
    spend = [r for r in rows if r.get("status") == "ok" and (r.get("cm_planned_total") or 0) > 0]
    based = sum(1 for r in spend if r.get("cm_plan_baseline"))
    print("  %s: 소비가 있는 사람-날 %d 중 계획 기준선이 쓰인 것 %d (%.1f%%)"
          % (d, len(spend), based, 100 * based / max(1, len(spend))))
    if not spend or based < 0.9 * len(spend):
        raise SystemExit("계획 통로가 꺼져 있다 — 기준선 파일 경로(EXP_PLAN_BASELINE_FILE)를 본다: " + d)

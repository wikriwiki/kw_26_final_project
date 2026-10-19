"""3주 A/B 실행기: 하루 실행이 끝났는지 — 명부 전원이 ok, 실패·건너뜀 0 — 확인하고 그날 요약을 남긴다.

    python tools/ab3w_check_day.py <summary.json> <day_DATE.json> <DATE> <명부 인원>
"""
import json
import sys
from pathlib import Path

summary, out, day, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
d = json.loads(Path(summary).read_text(encoding="utf-8"))
assert d.get("completed_at") and len(d["summary"]) == 1, "summary.json 이 하루 한 줄이 아니다"
r = d["summary"][0]
assert r["day"] == day, (r["day"], day)
assert r["ok"] == n and r["err"] == 0 and r.get("skipped", 0) == 0, r
Path(out).write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

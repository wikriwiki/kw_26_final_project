"""정책 전 기준선 — 주말 칸이 빈 사람도 엔진이 옛 공식으로 돌아가지 않게 사람 하나의 값을 함께 쓴다."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _day(folder, day, rows):
    (folder / f"day_{day}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_person_key_fills_missing_weekend(tmp_path):
    m = tmp_path / "metrics"; m.mkdir()
    # 2020-05-08 금(평일), 2020-05-09 토(주말)
    _day(m, "2020-05-08", [{"aid": "A", "status": "ok", "cm_planned_total": 100},
                           {"aid": "B", "status": "ok", "cm_planned_total": 300}])
    _day(m, "2020-05-09", [{"aid": "A", "status": "ok", "cm_planned_total": 0},
                           {"aid": "B", "status": "ok", "cm_planned_total": 500}])
    out = tmp_path / "base.json"
    subprocess.run([sys.executable, str(ROOT / "tools/build_plan_baseline_20261003.py"),
                    "--metrics", str(m), "--days", "2020-05-08,2020-05-09", "--out", str(out)],
                   check=True, capture_output=True)
    base = json.loads(out.read_text(encoding="utf-8"))
    assert base["A|wd"] == 100 and "A|we" not in base and base["A"] == 100
    assert base["B|wd"] == 300 and base["B|we"] == 500 and base["B"] == 400


def test_policy_day_is_refused(tmp_path):
    m = tmp_path / "metrics"; m.mkdir()
    _day(m, "2020-05-11", [{"aid": "A", "status": "ok", "cm_planned_total": 100,
                            "grant_applied_today": 400000}])
    r = subprocess.run([sys.executable, str(ROOT / "tools/build_plan_baseline_20261003.py"),
                        "--metrics", str(m), "--days", "2020-05-11", "--out", str(tmp_path / "b.json")],
                       capture_output=True)
    assert r.returncode != 0

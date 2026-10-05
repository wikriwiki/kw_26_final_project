"""3주 A/B 채점기 — 같은 사람·같은 날 차이, 사람·날이 어긋나면 멈춘다."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/report/score_ab3w.py"


def _write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _row(aid, day, total):
    return {"aid": aid, "day": day, "total_spent": total, "offline_spent": total, "online_spent": 0,
            "policy_funded_won": 0, "by_l1": {"식사": total}}


def test_paired_difference_and_direction(tmp_path):
    days = ["2020-05-11", "2020-05-12"]
    _write(tmp_path / "on.jsonl", [_row(a, d, 1200) for a in ("a", "b", "c") for d in days])
    _write(tmp_path / "off.jsonl", [_row(a, d, 1000) for a in ("a", "b", "c") for d in days])
    subprocess.run([sys.executable, str(SCRIPT), "--on", str(tmp_path / "on.jsonl"),
                    "--off", str(tmp_path / "off.jsonl"), "--draws", "200",
                    "--out", str(tmp_path / "s.json")], check=True, capture_output=True)
    s = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    t = s["measures"]["total_spent"]
    assert t["diff"] == 200 and t["diff_pct"] == 20.0 and t["direction"] == "+" and t["people_up"] == 3
    assert s["by_l1"]["식사"]["diff"] == 200


def test_mismatched_people_refused(tmp_path):
    _write(tmp_path / "on.jsonl", [_row("a", "2020-05-11", 1)])
    _write(tmp_path / "off.jsonl", [_row("b", "2020-05-11", 1)])
    r = subprocess.run([sys.executable, str(SCRIPT), "--on", str(tmp_path / "on.jsonl"),
                        "--off", str(tmp_path / "off.jsonl"), "--out", str(tmp_path / "s.json")],
                       capture_output=True)
    assert r.returncode != 0

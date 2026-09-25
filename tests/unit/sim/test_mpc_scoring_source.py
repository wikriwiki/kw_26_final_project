"""MPC 채점은 원장 비율을 읽고 총지출 차이로 대체하지 않는다."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/sim"))
from score_policy import metric_values, score_mpc_from_metrics  # noqa: E402


def test_mpc는_영수증_차이로_계산하지_않는다():
    off, on = metric_values("mpc_amount", [], [], [], [])
    assert off is None and on is None


def test_mpc는_해당_on일의_원장만_사용한다(tmp_path):
    for day, rows in {
        "2025-07-21": [('a', 1.0, 100), ('b', 1.0, 100)],
        "2025-07-22": [('a', 0.0, 10), ('b', 1.0, 30)],
    }.items():
        (tmp_path / f"day_{day}.jsonl").write_text(''.join(
            json.dumps({'aid': a, 'status': 'ok', 'cm_mpc_new_share': m,
                        'policy_spend_today': w}) + '\n' for a, m, w in rows),
            encoding='utf-8')
    score = score_mpc_from_metrics(tmp_path, ["2025-07-22"])
    assert score['mean'] == pytest.approx(0.75)
    assert score['n'] == 2 and score['n_cells'] == 2
    assert score['unit'] == 'ratio' and score['bootstrap_unit'] == 'aid'
    with pytest.raises(ValueError, match='관측일 누락'):
        score_mpc_from_metrics(tmp_path, ["2025-07-22", "2025-07-23"])


def test_mpc_원장_인자_없이_채점하면_명시적_오류():
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/sim/score_policy.py'),
         '--policy', 'P010', '--off', '2025-07-15:2025-07-16',
         '--on', '2025-07-22:2025-07-23'],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    assert result.returncode != 0
    assert '--metrics-dir' in result.stderr

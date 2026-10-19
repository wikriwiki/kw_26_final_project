"""Legacy reports must not turn proxy values into empirical magnitude scores."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _generate(script: str, tmp_path: Path, *args: str) -> str:
    output = tmp_path / 'report.md'
    subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'report' / script),
         *args, '--out', str(output)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return output.read_text(encoding='utf-8')


def test_single_probe_keeps_empirical_and_simulated_magnitudes_separate(tmp_path):
    report = _generate(
        'build_indicator_comparison.py', tmp_path,
        '--sim', 'data/experiments/indicators_v25_block.json',
    )
    assert '### 외부 참고값' in report
    assert '### 당시 시뮬레이션 대리값' in report
    assert '보고된 구간' in report
    assert '반복 4회' in report
    assert '크기 비교 불가' in report
    assert '```' not in report
    assert '배**' not in report


def test_candidate_probe_does_not_score_distance_to_empirical_value(tmp_path):
    report = _generate(
        'build_candidate_indicator_view.py', tmp_path,
        '--cand', 'v25=data/experiments/v7_v25_pooled.json',
        '--cand', 'v30=data/experiments/v7_v30_pooled.json',
    )
    assert '### 외부 참고값' in report
    assert '### 후보별 시뮬레이션 대리값' in report
    assert '후보 점수로 사용 불가' in report
    assert '```' not in report
    assert '판정 변화' not in report
    assert '부호 일치' not in report

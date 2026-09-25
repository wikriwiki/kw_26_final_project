"""기간·분모가 다른 역사적 결과를 결측이나 직접 오차로 세지 않는다."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "error_budget", ROOT / "scripts/report/error_budget.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_p012_부분월_상한비율은_미측정이_아니라_정의확인():
    rows = {r['id']: r for r in module.collect()}
    for iid in ('P012-4', 'P012-6'):
        assert rows[iid]['kind'] == '정의확인'
        assert rows[iid]['err'] is None
    assert rows['P012-6']['sim'] == 0


def test_현재_직접_비교_가능한_크기_오차는_없다():
    assert all(r['err'] is None for r in module.collect())

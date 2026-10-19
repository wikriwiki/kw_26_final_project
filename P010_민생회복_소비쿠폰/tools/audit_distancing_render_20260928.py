"""Freeze the 2020 distancing ON/OFF environment facts before simulation."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sim.environments.covid_2021 import disease_facts
from scripts.sim.environments.registry import build_environment

out = ROOT / 'experiments/multi_policy_v53_20260927/distancing_render_audit_20260928.json'
sources = [
    'scripts/sim/environments/registry.py',
    'scripts/sim/environments/covid_2021.py',
    'scripts/sim/environments/covid_no_distancing.py',
    'data/experiments/covid_support_2021/distancing_schedule.json',
    'data/experiments/covid_support_2021/seoul_cases_daily.json',
]
source_hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                 for path in sources}
daily = []
for offset in range(3):
    day = date(2020, 11, 24) + timedelta(days=offset)
    on = build_environment('covid_2021', day)
    off = build_environment('covid_no_distancing', day)
    common = disease_facts(day)
    assert on and off
    assert on['facts'][:len(common)] == common
    assert off['facts'][:len(common)] == common
    assert len(on['facts']) > len(common)
    assert len(off['facts']) == len(common) + 2
    assert any('카페는 시간과 무관하게' in fact for fact in on['facts'])
    assert any('추가 방역' in fact and '제한 없음' in fact for fact in off['facts'])
    daily.append({'date': day.isoformat(), 'shared_disease_facts': common,
                  'on': on, 'off': off})
result = {
    'purpose': 'read-only frozen input audit; no observed policy outcomes or target numbers',
    'on_environment_id': 'covid_2021',
    'off_environment_id': 'covid_no_distancing',
    'source_sha256': source_hashes,
    'daily': daily,
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + '\n',
               encoding='utf-8')
print(f'{out} SHA256={hashlib.sha256(out.read_bytes()).hexdigest()} PASS')

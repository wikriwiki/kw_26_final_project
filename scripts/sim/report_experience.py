"""Aggregate completed agent snapshots for one run/day, without any LLM call."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

try:
    from .experience import aggregate
except ImportError:
    from experience import aggregate


def build_report(run_dir, day, group_by='income'):
    date.fromisoformat(day)
    path = Path(run_dir) / 'metrics' / f'day_{day}.jsonl'
    rows, malformed = [], 0
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('not a record')
            rows.append(row)
        except ValueError:
            malformed += 1
    successful = {r['aid'] for r in rows if r.get('status') == 'ok' and r.get('aid')}
    failed = {r['aid'] for r in rows if r.get('status') != 'ok' and r.get('aid')} - successful
    relevant = [r for r in rows if r.get('experience_day') == day]
    return {
        'schema_version': 1, 'day': day, 'group_by': group_by,
        'measure': 'simulated_expressed_policy_stance',
        'source': str(path.resolve()),
        'quality': {'completed_agents': len(successful), 'failed_agents': len(failed),
                    'with_experience_schema': len({r['aid'] for r in relevant if r.get('status') == 'ok'}),
                    'malformed_rows': malformed},
        'groups': aggregate(relevant, group_by),
        'notes': ['Unmeasured is not neutral.',
                  'Appraisal dates are reported; same-day receipts are interpreted at the next Dawn.',
                  'Referenced facts are structurally validated; reason text is not proof of causality.'],
    }


def main():
    # The public CLI always uses the evidence gate. The original build_report
    # remains a legacy exploratory API and must not be used for validated export.
    from experience_export import main as validated_main
    return validated_main()


if __name__ == '__main__':
    main()

"""Give citizens an income, because without one a month cannot be observed.

The opening balance is the citizen's own spending anchor times a fixed number of days and
nothing ever adds to it, so the purse only falls. Measured over twenty-eight days that reads
as a sixty-five per cent collapse in spending, which is not behaviour: the citizens whose
balance never reached zero spent the same amount on the last day as on the first
(experiments/THE_PURSE_RUNS_DRY.md). Two thirds simply ran out.

Every validated result so far used a window of twelve days or fewer, which fits inside the
median purse of about twenty-six days. The published comparisons we have not matched are
month-long, and that is the window this unlocks.

**Off unless asked for.** `EXP_DAILY_INCOME` unset means zero, so every earlier run
reproduces exactly. This is a change to the apparatus, not to any prompt.

    EXP_DAILY_INCOME=anchor        earn your own daily spending anchor
    EXP_DAILY_INCOME=anchor:1.2    that, times 1.2
    EXP_DAILY_INCOME=45000         a flat amount per day

The original design treated `anchor` as the neutral setting: a citizen would earn
what they were expected to spend. Enlarging the purse also changes the balance
shown to Stage 2 and therefore changes behavior.

Later ledger audits found that `anchor` is *not* policy-exogenous: the anchor uses
the day's LLM propensity and selected POI prices, so a policy can change the
credited income. `baseline` instead reads a frozen, policy-free per-agent map
(`EXP_DAILY_INCOME_MAP`). Use the same map in both arms of a paired experiment.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

OFF = {'', '0', 'off', 'none', 'no', 'false'}


def parse(spec):
    """(kind, value) for an income setting. Raises on anything it does not understand.

    A typo must not read as 'no income': that failure is silent and would be blamed on the
    window, which is the mistake this whole line of work is trying not to repeat.
    """
    if spec is None:
        return ('off', 0.0)
    text = str(spec).strip().lower()
    if text in OFF:
        return ('off', 0.0)
    if text == 'anchor':
        return ('anchor', 1.0)
    if text == 'baseline':
        return ('baseline', 1.0)
    if text.startswith('anchor:'):
        try:
            factor = float(text.split(':', 1)[1])
        except ValueError:
            raise ValueError('Income factor is not a number: %r' % spec)
        if factor < 0:
            raise ValueError('Income factor cannot be negative: %r' % spec)
        return ('anchor', factor)
    try:
        amount = float(text)
    except ValueError:
        raise ValueError(
            'Unrecognised EXP_DAILY_INCOME %r. Use "anchor", "anchor:<factor>", "baseline", '
            'a number of won, or leave it unset for no income.' % spec)
    if amount < 0:
        raise ValueError('Income cannot be negative: %r' % spec)
    return ('flat', amount)


@lru_cache(maxsize=4)
def _baseline_map(path: str) -> tuple[dict[str, int], str]:
    source = Path(path)
    raw = source.read_bytes()
    data = json.loads(raw)
    values = data.get('daily_income_by_aid')
    if (data.get('schema') != 'baseline_income_v1'
            or data.get('policy_free_success_rows_verified') is not True
            or not isinstance(values, dict)
            or len(values) != data.get('citizen_count')):
        raise ValueError('Invalid policy-free baseline income map')
    if not values or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0
                         for v in values.values()):
        raise ValueError('Baseline income values must be positive integer won')
    return values, hashlib.sha256(raw).hexdigest()


def preflight_baseline_income(spec, path, aids) -> dict | None:
    """Validate the full citizen set before any expensive LLM call."""
    if parse(spec)[0] != 'baseline':
        return None
    if not path:
        raise ValueError('EXP_DAILY_INCOME_MAP is required for baseline income')
    values, sha = _baseline_map(str(Path(path).resolve()))
    wanted = {str(a) for a in aids}
    missing = wanted - set(values)
    if missing:
        raise ValueError(f'Baseline income map citizen set mismatch: missing={len(missing)}')
    return {'citizens': len(wanted), 'map_citizens': len(values),
            'map_sha256': sha}


def daily_income(anchor, spec, *, aid=None, baseline_map_path=None):
    """Won credited to this citizen tonight. Zero when the setting is off or absent."""
    kind, value = parse(spec)
    if kind == 'off':
        return 0
    if kind == 'flat':
        return int(round(value))
    if kind == 'baseline':
        if not baseline_map_path or aid is None:
            raise ValueError('Baseline income requires an agent and map path')
        values, _sha = _baseline_map(str(Path(baseline_map_path).resolve()))
        try:
            return values[str(aid)]
        except KeyError as exc:
            raise ValueError(f'Agent missing from baseline income map: {aid}') from exc
    return int(round((anchor or 0) * value))


def describe(spec):
    """One line for the run log, so a run's own output says whether it had income."""
    kind, value = parse(spec)
    if kind == 'off':
        return '소득 없음 (지갑이 마른다 — 창을 14일 이하로 두어야 한다)'
    if kind == 'flat':
        return '소득 하루 %s원 정액' % format(int(round(value)), ',d')
    if kind == 'baseline':
        return '정책 전 관측으로 고정한 개인별 하루 예산 보충액'
    if value == 1.0:
        return '소득 하루 = 본인 소비 앵커'
    return '소득 하루 = 본인 소비 앵커 x %g' % value

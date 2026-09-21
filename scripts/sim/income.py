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

`anchor` is the neutral setting: a citizen earns what they are expected to spend, so the
purse is stationary rather than large. Enlarging the purse instead is not neutral - the
balance is written into the Stage 2 prompt, so tripling it changes what the model reads.
"""
from __future__ import annotations

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
            'Unrecognised EXP_DAILY_INCOME %r. Use "anchor", "anchor:<factor>", '
            'a number of won, or leave it unset for no income.' % spec)
    if amount < 0:
        raise ValueError('Income cannot be negative: %r' % spec)
    return ('flat', amount)


def daily_income(anchor, spec):
    """Won credited to this citizen tonight. Zero when the setting is off or absent."""
    kind, value = parse(spec)
    if kind == 'off':
        return 0
    if kind == 'flat':
        return int(round(value))
    return int(round((anchor or 0) * value))


def describe(spec):
    """One line for the run log, so a run's own output says whether it had income."""
    kind, value = parse(spec)
    if kind == 'off':
        return '소득 없음 (지갑이 마른다 — 창을 14일 이하로 두어야 한다)'
    if kind == 'flat':
        return '소득 하루 %s원 정액' % format(int(round(value)), ',d')
    if value == 1.0:
        return '소득 하루 = 본인 소비 앵커'
    return '소득 하루 = 본인 소비 앵커 x %g' % value

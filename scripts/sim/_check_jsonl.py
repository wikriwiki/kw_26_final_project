"""baseline jsonl 초기 품질 점검."""
import json, sys
from collections import Counter

fp = r'C:\Users\Administrator\sim_output_9d\metrics\day_2026-05-25.jsonl'
rows = []
with open(fp, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print(f"총 {len(rows)}건")
st = Counter(r.get('status') for r in rows)
print(f"status: {dict(st)}")

ok = [r for r in rows if r.get('status') == 'ok']
if ok:
    # 정책 미적용 확인
    grant = sum(r.get('grant_applied_today', 0) for r in ok)
    phits = sum(r.get('policy_hits', 0) for r in ok)
    pspend = sum(r.get('policy_spend_today', 0) for r in ok)
    print(f"\n정책 (전부 0이어야 baseline): grant_applied={grant}, policy_hits={phits}, policy_spend={pspend}")

    # 소비·이벤트 분포
    nev = [r.get('n_events', 0) for r in ok]
    ninc = [r.get('n_includes', 0) for r in ok]
    bal = [r.get('balance') for r in ok if r.get('balance') is not None]
    sat = [r.get('avg_sat') for r in ok if r.get('avg_sat') is not None]
    print(f"\nn_events: min={min(nev)} max={max(nev)} avg={sum(nev)/len(nev):.1f}")
    print(f"n_includes(거래): min={min(ninc)} max={max(ninc)} avg={sum(ninc)/len(ninc):.1f}")
    if bal:
        print(f"balance: min={min(bal):,} max={max(bal):,}")
    if sat:
        print(f"avg_sat: min={min(sat):.2f} max={max(sat):.2f}")

    # fallback/환각 지표
    s2fb = sum(r.get('s2_fallback', 0) or 0 for r in ok)
    psc = sum(r.get('policy_spend_corrected', 0) or 0 for r in ok)
    print(f"\ns2_fallback: {s2fb}, policy_spend_corrected(환각보정): {psc}")

    # 샘플 1건
    print(f"\n샘플 ok 1건:")
    s = ok[0]
    for k in ['aid', 'status', 'n_events', 'n_includes', 'balance', 'avg_sat', 'grant_applied_today', 'policy_hits', 'tokens_in', 'tokens_out', 'elapsed']:
        print(f"  {k}: {s.get(k)}")

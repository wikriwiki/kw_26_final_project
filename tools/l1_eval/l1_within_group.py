"""업종을 같게 놓고 시간대·요일 비교 — 시간대 어긋남이 업종 쏠림 탓인지 가린다."""
import sys, json, pickle
import numpy as np, pandas as pd
from datetime import date, timedelta
sys.path.insert(0, 'code')
from l1_baseline import PROJ, RUNS, GROUPS, GI, SLOTS, DOW, TRUTH_MAP, sim_group, slot_of, sim1, SLOT_HOURS
df = pd.read_csv(PROJ + '서울시 상권분석서비스(추정매출-행정동)_2024년.csv', encoding='cp949')
df['grp'] = df['서비스_업종_코드_명'].map(TRUTH_MAP)
out = {}
for lbl, q in RUNS.items():
    d = pickle.load(open(f'cache2/{lbl}.pkl', 'rb')); m = d['manifest']
    days = [(date.fromisoformat(m['pre_start']) + timedelta(days=k)).isoformat() for k in range(14)]
    S = {g: np.zeros(6) for g in GROUPS}; Dw = {g: np.zeros(7) for g in GROUPS}
    for a in d['agents']:
        for day in days:
            src = 'on' if day < m['start'] else 'off'
            wd = date.fromisoformat(day).weekday()
            for (c, trig, sp, tm, sub, dong) in d['data'][(src, a)]['plans'].get(day, []):
                g = sim_group(c, sub); s_ = slot_of(tm)
                if sp > 0 and g and s_ is not None:
                    S[g][s_] += sp; Dw[g][wd] += sp
    x = df[(df['기준_년분기_코드'] == q) & df['grp'].notna()]
    rec = {}
    for g in ['음식점', '마트·식료품', '편의점', '카페·디저트']:
        xs = x[x['grp'] == g]
        ts = np.array([xs[f'시간대_{s}_매출_금액'].sum() for s in SLOTS]); td = np.array([xs[f'{w}요일_매출_금액'].sum() for w in DOW])
        rec[g] = {'n_sim_won': float(S[g].sum()),
                  'slot_sim': (S[g] / S[g].sum()).round(3).tolist(), 'slot_truth': (ts / ts.sum()).round(3).tolist(),
                  'slot_sim_vs_truth': round(sim1(S[g], ts), 3), 'slot_uniform_vs_truth': round(sim1(SLOT_HOURS / SLOT_HOURS.sum(), ts), 3),
                  'dow_sim': (Dw[g] / Dw[g].sum()).round(3).tolist(), 'dow_truth': (td / td.sum()).round(3).tolist(),
                  'dow_sim_vs_truth': round(sim1(Dw[g], td), 3), 'dow_uniform_vs_truth': round(sim1(np.ones(7), td), 3)}
        r = rec[g]
        print(f"{lbl} {g}: 시간대 sim {r['slot_sim_vs_truth']} / 균등 {r['slot_uniform_vs_truth']} | 요일 sim {r['dow_sim_vs_truth']} / 균등 {r['dow_uniform_vs_truth']} | sim 11~14 {r['slot_sim'][2]} vs {r['slot_truth'][2]} | sim 주말 {round(sum(r['dow_sim'][5:]),3)} vs {round(sum(r['dow_truth'][5:]),3)}", flush=True)
    out[lbl] = rec
json.dump(out, open('results/l1_within_group.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

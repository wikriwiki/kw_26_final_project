"""실험 간 사람 일관성 — 같은 페르소나는 실험(시기·사회 배경)이 바뀌어도 비슷하게 사는가.

각 실험의 정책 전 1주(정책이 없는 일상, 두 갈래 공통)를 사람마다 한 주 묶음으로 요약한다.
  · 주간 활동 구성: 장소 유형별 일정 수 비율
  · 주간 외출 구성: 집 밖 일정의 장소 유형 비율
  · 주간 지출 구성: 장소 유형별 지출 비율
  · 하루 평균 지출: exp(-|log(1+a)-log(1+b)|)
두 실험에 모두 들어간 사람 S 에 대해
  · 같은 사람:   요약(실험1, a) vs 요약(실험2, a)
  · 닮은 다른 사람: a 와 인물·장소 조건이 가장 가까운 다른 사람 b (실험2)
  · 무작위 다른 사람: S 안에서 무작위 b ≠ a
같은 사람 > 닮은 사람 > 무작위 이면, 행동이 사람 조건에 묶여 있고 그 묶임이 실험이 바뀌어도 유지된다는 뜻.
"""
import itertools
import json
import os
import pickle
import sys
from datetime import date, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from obs_consistency import INCOME, TEND, cat_dist, jsd_sim, normalize_rows, num, numeric_dist, total_sim  # noqa: E402

RUNS = ['P010', 'P012', 'P013', 'P017', 'DISTANCING']
SEED = 20261019


def week_profiles(d, cats, ci):
    pre = d['manifest']['pre_start']
    days = [(date.fromisoformat(pre) + timedelta(days=k)).isoformat() for k in range(7)]
    K = len(cats)
    home = ci.get('집')
    other = ci['(기타)']
    out = {}
    for aid in d['agents']:
        plans = d['data'][('on', aid)]['plans']
        act = np.zeros(K)
        outg = np.zeros(K)
        sp = np.zeros(K)
        tot = 0.0
        for day in days:
            for c, _, s, _ in plans.get(day, []):
                k = ci.get(c, other)
                act[k] += 1
                if k != home:
                    outg[k] += 1
                sp[k] += s
                tot += s
        out[aid] = (act, outg, sp, tot / 7)
    return out


def main(cache_dir, out_json):
    rng = np.random.default_rng(SEED)
    data = {r: pickle.load(open(f'{cache_dir}/{r}.pkl', 'rb')) for r in RUNS}
    # 공통 어휘(모든 실험의 일정 장소 유형, 0.05% 이상)
    cnt = {}
    for d in data.values():
        for (arm, a), v in d['data'].items():
            if arm != 'on':
                continue
            for items in v['plans'].values():
                for c, _, _, _ in items:
                    cnt[c] = cnt.get(c, 0) + 1
    tot = sum(cnt.values())
    cats = [c for c in sorted(cnt, key=lambda c: -cnt[c]) if cnt[c] / tot >= 0.0005] + ['(기타)']
    ci = {c: k for k, c in enumerate(cats)}
    prof = {r: week_profiles(d, cats, ci) for r, d in data.items()}
    persona = {}
    for d in data.values():
        persona.update(d['agents'])

    results = []
    for r1, r2 in itertools.combinations(RUNS, 2):
        S = sorted(set(prof[r1]) & set(prof[r2]))
        n = len(S)
        if n < 30:
            continue
        P = [persona[a] for a in S]
        age = np.array([num(p.get('age')) for p in P])
        sd = np.log1p(np.array([num(p.get('s_daily_wd')) for p in P]))
        inc = np.array([INCOME.get(p.get('income_level'), np.nan) for p in P], float)
        ten = np.array([TEND.get(p.get('spending_tendency'), np.nan) for p in P], float)
        def codes(key):
            vals = [str(p.get(key) or '') for p in P]
            u = {v: k for k, v in enumerate(sorted(set(vals)))}
            return np.array([u[v] for v in vals])
        parts = [numeric_dist(x, np.nanmax(x) - np.nanmin(x)) for x in (age, sd, inc, ten)]
        parts += [cat_dist(codes(k)) for k in ('gender', 'job', 'life_stage', 'workplace_dong')]
        res = np.array([a.split('_')[1] for a in S])
        parts.append(np.where(res[:, None] == res[None, :], 0.0, np.where(res[:, None].astype('U5') == res[None, :].astype('U5'), 0.5, 1.0)))
        D = np.mean(parts, axis=0)
        np.fill_diagonal(D, np.inf)
        nn = np.argmin(D, axis=1)
        A1 = [np.array([prof[r1][a][k] for a in S]) for k in range(4)]
        A2 = [np.array([prof[r2][a][k] for a in S]) for k in range(4)]
        def sims(J):
            return {'act': jsd_sim(normalize_rows(A1[0]), normalize_rows(A2[0][J])),
                    'out': jsd_sim(normalize_rows(A1[1]), normalize_rows(A2[1][J])),
                    'spend': jsd_sim(normalize_rows(A1[2]), normalize_rows(A2[2][J])),
                    'total': total_sim(A1[3], A2[3][J])}
        idx = np.arange(n)
        same = sims(idx)
        near = sims(nn)
        rand = {k: [] for k in same}
        for _ in range(50):
            J = rng.permutation(n)
            fix = J == idx
            J[fix] = (J[fix] + 1) % n
            for k, v in sims(J).items():
                rand[k].append(v)
        rand = {k: np.concatenate(v) for k, v in rand.items()}
        rec = {'pair': [r1, r2], 'n_shared': n, 'outcomes': {}}
        B = 1000
        Wb = rng.multinomial(n, np.ones(n) / n, size=B).astype(float)
        for k in same:
            def bmean(v, I=None):
                I = idx if I is None else I
                ok = ~np.isnan(v)
                S_ = np.bincount(I[ok], weights=v[ok], minlength=n)
                C_ = np.bincount(I[ok], minlength=n).astype(float)
                return S_.sum() / C_.sum(), (Wb @ S_) / (Wb @ C_)
            s_pt, s_b = bmean(same[k])
            n_pt, n_b = bmean(near[k])
            r_pt, r_b = bmean(rand[k], np.tile(idx, 50))
            rec['outcomes'][k] = {
                'same': [float(s_pt), *map(float, np.percentile(s_b, [2.5, 97.5]))],
                'near': [float(n_pt), *map(float, np.percentile(n_b, [2.5, 97.5]))],
                'rand': [float(r_pt), *map(float, np.percentile(r_b, [2.5, 97.5]))],
                'same_minus_rand': [float(s_pt - r_pt), *map(float, np.percentile(s_b - r_b, [2.5, 97.5]))],
                'near_share': float((n_pt - r_pt) / (s_pt - r_pt)) if s_pt != r_pt else None,
            }
        results.append(rec)
        o = rec['outcomes']
        print(f"{r1}-{r2} n={n} | " + ' | '.join(f"{k}: same {o[k]['same'][0]:.3f} near {o[k]['near'][0]:.3f} rand {o[k]['rand'][0]:.3f}" for k in o), flush=True)
    json.dump({'cats': cats, 'pairs': results}, open(out_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

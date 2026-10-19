"""금연구역 v22 (1,154명, 노원·서초·송파) — 같은 관찰 시험을 이전 엔진 실험에 적용한다.

이 실험은 하루 일정이 그래프 덤프에만 있어 결제 영수증으로 결과를 잰다.
  · 결제 활동 구성 1 - JSD(결제 건의 장소 유형 비율)
  · 지출 구성     1 - JSD(장소 유형별 지출 비율)
  · 총지출        exp(-|log(1+a) - log(1+b)|)
그리고 입장(설문 v1, 찬성·반대·잘 모르겠다)이 같은지를 본다.
  · 같은 사람 재질문: 같은 문항·같은 보기 순서로 두 번 더 물은 120명(시행 전, ON 사후)
조건 묶음: 인물(흡연 여부·성·연령대·생애단계·소득·소비 분위·평일 소비액·이동성·직업·성향),
장소(거주 동/구), 기억(어제·최근 7일 결제 구성, 최근 7일 지출, 어제 기분·피로),
상호작용(최근 7일 밤 대화 후보로 묶인 횟수). 갈래 안 비교라 정책 묶음은 없다.
"""
import glob
import json
import os
import sys
from datetime import date, timedelta

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from obs_consistency import cat_dist, jsd_sim, normalize_rows, numeric_dist, total_sim, tv_dist  # noqa: E402

NSZ = 'C:/Users/srdyh/sim_data/nsz_final'
ARMS = {'pre': 'shared-pre', 'on': 'post-on', 'off': 'post-off'}
AGE = {'10대': 1, '20대': 2, '30대': 3, '40대': 4, '50대': 5, '60대': 6, '70대이상': 7}
INCOME = {'하': 0, '중하': 1, '중': 2, '중상': 3, '상': 4}
SEED = 20261019
N_BOOT = 1000
R_RANDOM = 10


def add_days(day, k):
    return (date.fromisoformat(day) + timedelta(days=k)).isoformat()


def load():
    # 인물: 설문 기록의 근거 패킷(시행 전 셀)에서
    prof = {}
    for f in glob.glob(f'{NSZ}/stance_kexaone/pre/records/*.json'):
        r = json.load(open(f, encoding='utf-8'))
        v = {}
        for it in r['evidence_packet']['evidence_items']:
            if it['kind'] == 'persona_snapshot':
                v.update(json.loads(it['text'])['values'])
        prof[r['agent_id']] = v
    # 결제·상태: 갈래별 날짜별 metrics
    days = {}
    for arm, folder in ARMS.items():
        for f in sorted(glob.glob(f'{NSZ}/full_runs/{folder}/metrics/day_*.jsonl')):
            day = os.path.basename(f)[4:14]
            for line in open(f, encoding='utf-8'):
                r = json.loads(line)
                rec = [(e.get('category') or '?', float(e.get('own_paid') or e.get('amount') or 0))
                       for e in (r.get('execution_receipts') or []) if e.get('purchase_status') == 'purchased']
                days[(arm, r['aid'], day)] = {'rec': rec, 'mood': r.get('mood'), 'fat': r.get('fatigue')}
    # 상호작용: 하루 단위 밤 대화 기록(evidence/night) — 대화 한 건마다 양쪽 시민에게 1회
    conv = {}
    for arm, folder in ARMS.items():
        for f in glob.glob(f'{NSZ}/full_runs/{folder}/evidence/night/*.json'):
            r = json.load(open(f, encoding='utf-8'))
            day = r['day']
            for c in r.get('results') or []:
                ids = c.get('evidence_agent_ids') or [c.get('initiator_id'), c.get('recipient_id')]
                for a in {x for x in ids if x}:
                    conv[(arm, a, day)] = conv.get((arm, a, day), 0) + 1
    # 설문
    survey = {}
    for cell in ('pre', 'on_post', 'off_post'):
        for f in glob.glob(f'{NSZ}/survey_kexaone/{cell}/v1_*.jsonl'):
            tag = os.path.basename(f).split('.')[0]          # v1_r1 / v1_r2 / v1_r3 / v1_t07_r1 ...
            if tag not in ('v1_r1', 'v1_r2', 'v1_r3'):
                continue
            for line in open(f, encoding='utf-8'):
                r = json.loads(line)
                survey[(cell, tag, r['agent_id'])] = r.get('choice')
    return prof, days, conv, survey


def persona_matrices(aids, prof):
    P = [prof.get(a, {}) for a in aids]
    def nums(key, f=float):
        out = []
        for p in P:
            try:
                out.append(f(p.get(key)))
            except (TypeError, ValueError):
                out.append(np.nan)
        return np.array(out, float)
    def codes(vals):
        u = {v: k for k, v in enumerate(sorted(set(vals)))}
        return np.array([u[v] for v in vals])
    age = np.array([AGE.get(a.split('_')[3], np.nan) for a in aids], float)
    inc = np.array([INCOME.get(p.get('income'), np.nan) for p in P], float)
    dec = nums('spend_decile')
    dwd = np.log1p(nums('daily_wd'))
    mob = nums('mobility')
    parts = [numeric_dist(x, np.nanmax(x) - np.nanmin(x)) for x in (age, inc, dec, dwd, mob)]
    parts += [cat_dist(codes([str(p.get(k) or '') for p in P])) for k in ('smoking_status', 'life_stage', 'job', 'tendency')]
    parts.append(cat_dist(codes([a.split('_')[2] for a in aids])))           # 성별
    D_persona = np.mean(parts, axis=0)
    dong = np.array([a.split('_')[1] for a in aids])
    gu = np.array([d[:5] for d in dong])
    D_place = np.where(dong[:, None] == dong[None, :], 0.0, np.where(gu[:, None] == gu[None, :], 0.5, 1.0)).astype(np.float32)
    smoke = np.array([str(p.get('smoking_status') or '') for p in P])
    return D_persona, D_place, smoke


def boot_mean(i, v, N, W):
    ok = ~np.isnan(v)
    S = np.bincount(i[ok], weights=v[ok], minlength=N)
    C = np.bincount(i[ok], minlength=N).astype(float)
    return S.sum() / C.sum(), (W @ S) / (W @ C), int(C.sum())


def summarize_pairs(sets, outcome_keys, N, W, rng, groups):
    """sets: name -> dict(i, dist, d_<g>, <outcome>...). nn/rand 필수, self 선택."""
    out = {}
    for o in outcome_keys:
        rec = {}
        bs = {}
        for name, F in sets.items():
            if F is None or o not in F:
                continue
            pt, b, n = boot_mean(F['i'], F[o], N, W)
            rec[name] = [float(pt), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
            rec[name + '_n'] = n
            bs[name] = b
        if 'self' in bs and 'nn_ceil' in bs:
            z = (bs['nn_ceil'] - bs['rand_ceil']) / (bs['self'] - bs['rand_ceil'])
            pt = (rec['nn_ceil'][0] - rec['rand_ceil'][0]) / (rec['self'][0] - rec['rand_ceil'][0])
            rec['normalized'] = [float(pt), float(np.percentile(z, 2.5)), float(np.percentile(z, 97.5))]
        diff = bs['nn'] - bs['rand']
        rec['nn_minus_rand'] = [float(rec['nn'][0] - rec['rand'][0]), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
        F = sets['rand']
        ok = ~np.isnan(F[o])
        q = np.quantile(F['dist'][ok], np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(q, F['dist'], side='right') - 1, 0, 9)
        bins = []
        for k in range(10):
            sel = ok & (b == k)
            S = np.bincount(F['i'][sel], weights=F[o][sel], minlength=N)
            C = np.bincount(F['i'][sel], minlength=N).astype(float)
            bb = (W[:300] @ S) / (W[:300] @ C)
            bins.append({'x': float(F['dist'][sel].mean()), 'y': float(S.sum() / C.sum()),
                         'lo': float(np.nanpercentile(bb, 2.5)), 'hi': float(np.nanpercentile(bb, 97.5)), 'n': int(sel.sum())})
        rec['dose'] = bins
        ys = [x['y'] for x in bins]
        rec['monotone_violations'] = int(sum(1 for k in range(9) if ys[k + 1] > ys[k]))
        okk = np.where(ok)[0]
        sub = rng.choice(okk, size=min(80000, okk.size), replace=False)
        rec['spearman'] = float(spearmanr(F['dist'][sub], F[o][sub]).correlation)
        gs = [g for g in groups if ('d_' + g) in F]
        Xr = np.column_stack([F['d_' + g][ok] for g in gs])
        mu, sd = Xr.mean(0), Xr.std(0)
        sd[sd == 0] = 1
        X = np.column_stack([np.ones(len(Xr)), (Xr - mu) / sd])
        y = F[o][ok]
        ii = F['i'][ok]
        p = X.shape[1]
        XtX = np.zeros((N, p, p))
        Xty = np.zeros((N, p))
        for a_ in range(p):
            Xty[:, a_] = np.bincount(ii, weights=X[:, a_] * y, minlength=N)
            for b_ in range(a_, p):
                col = np.bincount(ii, weights=X[:, a_] * X[:, b_], minlength=N)
                XtX[:, a_, b_] = col
                XtX[:, b_, a_] = col
        beta = np.linalg.solve(XtX.sum(0), Xty.sum(0))
        bsb = np.array([np.linalg.solve(np.tensordot(w, XtX, 1), w @ Xty) for w in W[:300]])
        rec['factors'] = {g: [float(beta[k + 1]), float(np.percentile(bsb[:, k + 1], 2.5)),
                              float(np.percentile(bsb[:, k + 1], 97.5))] for k, g in enumerate(gs)}
        out[o] = rec
    return out


def cat_concat(rows):
    keys = rows[0].keys()
    return {k: np.concatenate([r[k] for r in rows]) for k in keys}


def main(out_json):
    rng = np.random.default_rng(SEED)
    prof, days, conv, survey = load()
    aids = sorted(prof)
    N = len(aids)
    W = rng.multinomial(N, np.ones(N) / N, size=N_BOOT).astype(float)
    D_persona, D_place, smoke = persona_matrices(aids, prof)
    cats_cnt = {}
    for v in days.values():
        for c, _ in v['rec']:
            cats_cnt[c] = cats_cnt.get(c, 0) + 1
    cats = sorted(cats_cnt, key=lambda c: -cats_cnt[c])
    ci = {c: k for k, c in enumerate(cats)}
    K = len(cats)
    arm_days = {arm: sorted({d for (a_, _, d) in days if a_ == arm}) for arm in ARMS}

    def vecs(arm, day):
        cnt = np.zeros((N, K), np.float32)
        spv = np.zeros((N, K), np.float32)
        tot = np.zeros(N)
        ok = np.zeros(N, bool)
        mood = np.full(N, np.nan)
        fat = np.full(N, np.nan)
        for i, a in enumerate(aids):
            v = days.get((arm, a, day))
            if v is None:
                continue
            ok[i] = True
            for c, amt in v['rec']:
                cnt[i, ci[c]] += 1
                spv[i, ci[c]] += amt
                tot[i] += amt
            mood[i] = v['mood'] if v['mood'] is not None else np.nan
            fat[i] = v['fat'] if v['fat'] is not None else np.nan
        return cnt, spv, tot, ok, mood, fat

    def hist_days(arm, day, k):
        """arm 의 day 직전 k일(정책 갈래 첫 주는 정책 전 기록으로 이어 붙인다)."""
        out = []
        for j in range(1, k + 1):
            dd = add_days(day, -j)
            src = arm if dd in arm_days[arm] else 'pre'
            out.append((src, dd))
        return out

    # ---------- 행동(결제) 일관성
    beh_rows = {'nn': [], 'rand': []}
    for arm in ('pre', 'on', 'off'):
        for day in arm_days[arm][2:] if arm == 'pre' else arm_days[arm]:
            cnt, spv, tot, ok, _, _ = vecs(arm, day)
            src_y, y = hist_days(arm, day, 1)[0]
            ycnt, _, _, _, mood, fat = vecs(src_y, y)
            h7 = np.zeros((N, K), np.float32)
            s7 = np.zeros(N)
            c7 = np.zeros(N)
            for src, dd in hist_days(arm, day, 7):
                c_, _, t_, _, _, _ = vecs(src, dd)
                h7 += c_
                s7 += t_
                for i, a in enumerate(aids):
                    c7[i] += conv.get((src, a, dd), 0)
            s7 = np.log1p(s7)
            D_mem = np.mean([tv_dist(normalize_rows(ycnt)), tv_dist(normalize_rows(h7)),
                             numeric_dist(s7, s7.max() - s7.min()), numeric_dist(mood, 1.0), numeric_dist(fat, 1.0)], axis=0)
            D_int = numeric_dist(c7, c7.max() - c7.min() if c7.max() > 0 else 1.0)
            D = np.mean([D_persona, D_place, D_mem, D_int], axis=0)
            act, sp_, = normalize_rows(cnt), normalize_rows(spv)
            valid = np.where(ok)[0]
            Dv = D[np.ix_(valid, valid)].copy()
            np.fill_diagonal(Dv, np.inf)
            nn = valid[np.argmin(Dv, axis=1)]
            pairs = [('nn', valid, nn)]
            for _ in range(R_RANDOM):
                j = rng.choice(valid, size=valid.size)
                bad = j == valid
                while bad.any():
                    j[bad] = rng.choice(valid, size=bad.sum())
                    bad = j == valid
                pairs.append(('rand', valid, j))
            for typ, I, J in pairs:
                beh_rows[typ].append({'i': I, 'dist': D[I, J], 'd_persona': D_persona[I, J], 'd_place': D_place[I, J],
                                      'd_memory': D_mem[I, J], 'd_interaction': D_int[I, J],
                                      'act': jsd_sim(act[I], act[J]), 'spend': jsd_sim(sp_[I], sp_[J]),
                                      'total': total_sim(tot[I], tot[J])})
            print('  beh', arm, day, flush=True)
    beh = summarize_pairs({k: cat_concat(v) for k, v in beh_rows.items()}, ['act', 'spend', 'total'], N, W, rng,
                          ['persona', 'place', 'memory', 'interaction'])

    # ---------- 입장 일관성
    CELL_ARM = {'pre': 'pre', 'on_post': 'on', 'off_post': 'off'}
    CODE = {'찬성': 0, '반대': 1, '잘 모르겠다': 2}
    st = {'nn': [], 'rand': [], 'self': [], 'nn_ceil': [], 'rand_ceil': []}
    marg = {}
    for cell, arm in CELL_ARM.items():
        ch = np.array([CODE.get(survey.get((cell, 'v1_r1', a)), -1) for a in aids])
        marg[cell] = {k: int((ch == c).sum()) for k, c in CODE.items()}
        period = arm_days[arm]
        cnt = np.zeros((N, K), np.float32)
        tot = np.zeros(N)
        cv = np.zeros(N)
        for dd in period:
            c_, _, t_, _, _, _ = vecs(arm, dd)
            cnt += c_
            tot += t_
            for i, a in enumerate(aids):
                cv[i] += conv.get((arm, a, dd), 0)
        lt = np.log1p(tot)
        D_mem = np.mean([tv_dist(normalize_rows(cnt)), numeric_dist(lt, lt.max() - lt.min())], axis=0)
        D_int = numeric_dist(cv, cv.max() - cv.min() if cv.max() > 0 else 1.0)
        D = np.mean([D_persona, D_place, D_mem, D_int], axis=0)
        valid = np.where(ch >= 0)[0]
        Dv = D[np.ix_(valid, valid)].copy()
        np.fill_diagonal(Dv, np.inf)
        nn = valid[np.argmin(Dv, axis=1)]
        def rec(I, J):
            return {'i': I, 'dist': D[I, J], 'd_persona': D_persona[I, J], 'd_place': D_place[I, J],
                    'd_memory': D_mem[I, J], 'd_interaction': D_int[I, J], 'stance': (ch[I] == ch[J]).astype(float)}
        st['nn'].append(rec(valid, nn))
        for _ in range(20):
            j = rng.choice(valid, size=valid.size)
            bad = j == valid
            while bad.any():
                j[bad] = rng.choice(valid, size=bad.sum())
                bad = j == valid
            st['rand'].append(rec(valid, j))
        # 같은 사람 재질문(120명): r1 vs r2, r1 vs r3
        rep = [i for i, a in enumerate(aids) if (cell, 'v1_r2', a) in survey]
        if rep:
            I = np.array(rep)
            for tag in ('v1_r2', 'v1_r3'):
                c2 = np.array([CODE.get(survey.get((cell, tag, aids[i])), -1) for i in I])
                keep = (c2 >= 0) & (ch[I] >= 0)
                st['self'].append({'i': I[keep], 'stance': (ch[I][keep] == c2[keep]).astype(float)})
            # 같은 120명 안에서의 최근접·무작위(재질문과 같은 표본으로 정규화)
            sub = I[ch[I] >= 0]
            Ds = D[np.ix_(sub, sub)].copy()
            np.fill_diagonal(Ds, np.inf)
            st['nn_ceil'].append({'i': sub, 'stance': (ch[sub] == ch[sub[np.argmin(Ds, axis=1)]]).astype(float)})
            for _ in range(50):
                j = rng.choice(sub, size=sub.size)
                bad = j == sub
                while bad.any():
                    j[bad] = rng.choice(sub, size=bad.sum())
                    bad = j == sub
                st['rand_ceil'].append({'i': sub, 'stance': (ch[sub] == ch[j]).astype(float)})
        print('  stance', cell, marg[cell], flush=True)
    sets = {k: cat_concat(v) for k, v in st.items() if v}
    stance = summarize_pairs(sets, ['stance'], N, W, rng, ['persona', 'place', 'memory', 'interaction'])
    # 흡연 여부만 같은 짝 vs 다른 짝 (정책 표적 속성)
    F = sets['rand']
    sm = smoke[F['i']]
    stance['by_smoking_match'] = {}
    out = {'run': 'NSZ', 'N': N, 'behavior': beh, 'stance': stance, 'marginals': marg, 'cats': cats}
    json.dump(out, open(out_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    o = beh
    print('NSZ behavior: ' + ' | '.join(f"{k}: nn {o[k]['nn'][0]:.3f} rand {o[k]['rand'][0]:.3f} ρ {o[k]['spearman']:+.3f}" for k in o))
    s = stance['stance']
    print(f"NSZ stance: nn {s['nn'][0]:.3f} rand {s['rand'][0]:.3f} self {s.get('self',[None])[0]} norm {s.get('normalized')}")


if __name__ == '__main__':
    main(sys.argv[1])

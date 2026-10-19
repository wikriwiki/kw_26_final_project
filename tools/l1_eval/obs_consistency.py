"""관찰 시험 — 조건이 비슷한 에이전트는 비슷한 활동·지출을 고르는가 (보존된 본런 5건).

설계
- 비교 단위: 같은 실험·같은 갈래·같은 날의 두 시민. 날짜·사회 배경·요일이 같으므로 남는 차이는 사람 조건뿐이다.
- 조건 거리: Gower(1971) 방식. 변수마다 [0,1]로 맞춘 뒤 다섯 묶음(인물·장소·정책·기억·상호작용)의
  평균을 내고, 묶음 거리의 평균을 전체 거리로 쓴다(묶음마다 같은 무게).
  · 인물: 성별·연령·소득·직업·생애단계·소비 성향·평일 소비 수준·하루 소비액·통근 시간·이동성
  · 장소: 거주 동(같은 동 0, 같은 구 0.5, 다른 구 1)·직장 동
  · 정책: 받았는지·받은 뒤 날수·남은 잔액(정책 갈래, 지원금형) / 등록 여부·적립액(상생)
  · 기억: 어제 활동 구성, 최근 7일 방문 기억의 업종 구성, 최근 7일 지출, 어제 기분·피로
  · 상호작용: 최근 7일 소문 기억 수, 최근 7일 약속 계기 일정 수
- 결과 유사도(그날, LLM 이 고른 것):
  · 활동 구성   1 - JSD(장소 유형별 일정 비율)
  · 외출 구성   1 - JSD(집 밖 일정의 장소 유형 비율), 둘 다 외출이 있을 때
  · 지출 구성   1 - JSD(업종별 지출 비율), 둘 다 지출이 있을 때
  · 총지출      exp(-|log(1+a) - log(1+b)|)
- 짝: 최근접 짝(조건 거리 최소), 무작위 짝(사람마다 10명), 같은 사람 재실행(정책 갈래 첫날 아직 받지 않은
  사람의 정책 있음·없음 두 결과 — 입력이 같고 LLM 샘플링만 다르다).
- 정규화 일관성 = (최근접 − 무작위) / (재실행 − 무작위). Park et al.(2024)이 사람 2주 재검사 일치로 정확도를
  나눈 것과 같은 생각: 1 이면 조건이 같은 다른 사람이 '자기 자신을 다시 돌린 것'만큼 닮았다는 뜻.
- 불확실성: 첫 사람(i) 단위 군집 붓스트랩.
"""
import json
import math
import pickle
import sys
from datetime import date, timedelta

import numpy as np

RUNS = ['P010', 'P012', 'P013', 'P017', 'DISTANCING']
CEILING_OK = {'P010', 'P013', 'P017'}          # 첫날 정책 노출이 없는 실험만
INCOME = {'하': 0, '중하': 1, '중': 2, '중상': 3, '상': 4}
TEND = {'절약형': 0, '보통': 1, '소비형': 2}
OUTCOMES = ['act', 'out', 'spend', 'total']
GROUPS = ['persona', 'place', 'policy', 'memory', 'interaction']
R_RANDOM = 10
N_BOOT = 1000
SEED = 20261019


def add_days(day, k):
    return (date.fromisoformat(day) + timedelta(days=k)).isoformat()


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------- 거리 계산
def numeric_dist(x, rng_):
    """x: (N,) → (N,N) |xi-xj|/범위, 결측이 끼면 1."""
    r = rng_ if rng_ > 0 else 1.0
    d = np.abs(x[:, None] - x[None, :]) / r
    d = np.where(np.isnan(d), 1.0, d)
    return np.clip(d, 0, 1).astype(np.float32)


def cat_dist(codes):
    return (codes[:, None] != codes[None, :]).astype(np.float32)


def tv_dist(P):
    """P: (N,K) 행 정규화된 분포(합 0 이면 빈 분포) → (N,N) 전변동 거리 = 1 - Σ min(p,q).
    둘 다 비면 0, 한쪽만 비면 1."""
    N, K = P.shape
    overlap = np.zeros((N, N), np.float32)
    for k in range(K):
        col = P[:, k]
        if not col.any():
            continue
        overlap += np.minimum(col[:, None], col[None, :])
    out = (1.0 - overlap).astype(np.float32)
    empty = P.sum(1) == 0
    both_empty = empty[:, None] & empty[None, :]
    one_empty = empty[:, None] ^ empty[None, :]
    out[both_empty] = 0.0
    out[one_empty] = 1.0
    return out


def normalize_rows(M):
    s = M.sum(1, keepdims=True)
    return np.divide(M, s, out=np.zeros_like(M), where=s > 0)


# ---------------------------------------------------------------- 결과 유사도
def jsd_sim(P, Q):
    """P,Q: (n,K) 정규화된 분포 → 1 - JSD(log2), 둘 중 하나라도 비면 nan."""
    M = 0.5 * (P + Q)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(P > 0, P * np.log2(P / M), 0.0).sum(1)
        b = np.where(Q > 0, Q * np.log2(Q / M), 0.0).sum(1)
    s = 1.0 - 0.5 * (a + b)
    bad = (P.sum(1) == 0) | (Q.sum(1) == 0)
    s[bad] = np.nan
    return s


def total_sim(a, b):
    return np.exp(-np.abs(np.log1p(a) - np.log1p(b)))


# ---------------------------------------------------------------- 한 실험
def analyze_run(label, cache_dir, rng):
    d = pickle.load(open(f'{cache_dir}/{label}.pkl', 'rb'))
    m = d['manifest']
    pre_start, start, end = m['pre_start'], m['start'], m['post_end']
    aids = sorted(d['agents'])
    N = len(aids)
    idx = {a: i for i, a in enumerate(aids)}
    prof = [d['agents'][a] for a in aids]

    # 어휘: 일정의 장소 유형(활동·지출 결과용)과 방문 기억의 업종(기억 조건용)을 따로 둔다.
    # 드문 값(0.05% 미만)·상위 30개 밖의 기억 업종은 '기타'로 묶는다.
    pv, mv = {}, {}
    for (arm, a), v in d['data'].items():
        for items in v['plans'].values():
            for c, _, _, _ in items:
                pv[c] = pv.get(c, 0) + 1
        for (_, typ, c, _, _) in v['mems']:
            if typ == 'visited' and c:
                mv[c] = mv.get(c, 0) + 1
    tot_p = sum(pv.values())
    cats = [c for c in sorted(pv, key=lambda c: -pv[c]) if pv[c] / tot_p >= 0.0005] + ['(기타)']
    ci = {c: k for k, c in enumerate(cats)}
    other = ci['(기타)']
    K = len(cats)
    home = ci.get('집')
    mcats = sorted(mv, key=lambda c: -mv[c])[:30]
    mci = {c: k for k, c in enumerate(mcats)}
    MK = len(mcats) + 1
    def pk(c):
        return ci.get(c, other)
    def mk(c):
        return mci.get(c, MK - 1)

    # 인물·장소 거리 (정적)
    age = np.array([num(p.get('age')) for p in prof])
    sdaily = np.log1p(np.array([num(p.get('s_daily_wd')) for p in prof]))
    commute = np.array([num(p.get('commute_min')) for p in prof])
    mob = np.array([num(p.get('mobility_level')) for p in prof])
    slev = np.array([num(p.get('spending_level_wd')) for p in prof])
    inc = np.array([INCOME.get(p.get('income_level'), np.nan) for p in prof], float)
    ten = np.array([TEND.get(p.get('spending_tendency'), np.nan) for p in prof], float)

    def codes(key):
        vals = [str(p.get(key) or '') for p in prof]
        u = {v: k for k, v in enumerate(sorted(set(vals)))}
        return np.array([u[v] for v in vals])

    persona_parts = [numeric_dist(x, np.nanmax(x) - np.nanmin(x)) for x in (age, sdaily, commute, mob, slev, inc, ten)]
    persona_parts += [cat_dist(codes(k)) for k in ('gender', 'job', 'life_stage')]
    D_persona = np.mean(persona_parts, axis=0)
    res = np.array([a.split('_')[1] for a in aids])
    gu = np.array([r[:5] for r in res])
    D_res = np.where(res[:, None] == res[None, :], 0.0, np.where(gu[:, None] == gu[None, :], 0.5, 1.0))
    D_work = cat_dist(codes('workplace_dong'))
    D_place = ((D_res + D_work) / 2).astype(np.float32)
    del persona_parts

    def arm_days():
        out = []
        for k in range(2, 7):                      # 정책 전 주: 어제와 최근 기억이 있는 셋째 날부터
            out.append(('pre', add_days(pre_start, k)))
        for arm in ('on', 'off'):
            k = 0
            while add_days(start, k) <= end:
                out.append((arm, add_days(start, k)))
                k += 1
        return out

    def plan_of(arm, a, day):
        src = 'on' if arm == 'pre' else arm
        return d['data'][(src, a)]['plans'].get(day)

    # 기억·상호작용용 범위(실험 전체 사람-날)
    def day_feats(arm, day):
        src = 'on' if arm == 'pre' else arm
        y = add_days(day, -1)
        yact = np.zeros((N, K), np.float32)
        hist7 = np.zeros((N, MK), np.float32)
        spend7 = np.zeros(N)
        mood = np.full(N, np.nan)
        fat = np.full(N, np.nan)
        rumor7 = np.zeros(N)
        appt7 = np.zeros(N)
        lo = add_days(day, -7)
        for i, a in enumerate(aids):
            v = d['data'][(src, a)]
            for c, trig, sp, _ in v['plans'].get(y, []):
                yact[i, pk(c)] += 1
            for k in range(1, 8):
                dd = add_days(day, -k)
                for c, trig, sp, _ in v['plans'].get(dd, []):
                    spend7[i] += sp
                    if trig == 'appointment':
                        appt7[i] += 1
            for (md, typ, c, trig, sp) in v['mems']:
                if lo <= md < day:
                    if typ == 'visited' and c:
                        hist7[i, mk(c)] += 1
                    elif typ == 'rumor':
                        rumor7[i] += 1
            st = v['states'].get(y)
            if st:
                mood[i], fat[i] = num(st[0]), num(st[1])
        return dict(yact=normalize_rows(yact), hist7=normalize_rows(hist7), spend7=np.log1p(spend7),
                    mood=mood, fat=fat, rumor7=rumor7, appt7=appt7)

    def policy_feats(arm, day):
        if arm != 'on' or label == 'DISTANCING':
            return None
        P = d['policy']
        if label == 'P012':
            enr = np.array([1.0 if P.get(('on', a, day), {}).get('enrolled') else 0.0 for a in aids])
            cb = np.log1p(np.array([P.get(('on', a, add_days(day, -1)), {}).get('cashback', 0.0) for a in aids]))
            return [cat_dist(enr), numeric_dist(cb, cb.max() - cb.min())]
        exposed = np.zeros(N)
        since = np.zeros(N)
        remain = np.zeros(N)
        for i, a in enumerate(aids):
            if P.get(('on', a, day), {}).get('received', 0) > 0:
                exposed[i] = 1
                k = 0
                while add_days(start, k) <= day and P.get(('on', a, add_days(start, k)), {}).get('received', 0) == 0:
                    k += 1
                since[i] = (date.fromisoformat(day) - date.fromisoformat(add_days(start, k))).days
                prev = P.get(('on', a, add_days(day, -1)), {})
                remain[i] = prev.get('remaining', 0) if prev.get('received', 0) > 0 else P[('on', a, day)].get('received', 0)
        if exposed.sum() == 0:
            return None
        lr = np.log1p(remain)
        return [cat_dist(exposed), numeric_dist(since, 6.0), numeric_dist(lr, lr.max() - lr.min() if lr.max() > 0 else 1)]

    def outcomes(arm, day, who=None):
        rows = aids if who is None else who
        act = np.zeros((len(rows), K), np.float32)
        out = np.zeros((len(rows), K), np.float32)
        spv = np.zeros((len(rows), K), np.float32)
        tot = np.zeros(len(rows))
        ok = np.zeros(len(rows), bool)
        nsp = np.zeros(len(rows))
        for r, a in enumerate(rows):
            items = plan_of(arm, a, day)
            if items is None:
                continue
            ok[r] = True
            for c, trig, sp, _ in items:
                k = pk(c)
                act[r, k] += 1
                if k != home:
                    out[r, k] += 1
                spv[r, k] += sp
                tot[r] += sp
                if sp > 0:
                    nsp[r] += 1
        outcomes.last_nsp = nsp
        return normalize_rows(act), normalize_rows(out), normalize_rows(spv), tot, ok

    # 범위 정규화를 위해 기억 수치의 실험 전체 범위를 먼저 모은다
    plan = arm_days()
    feats = {(arm, day): day_feats(arm, day) for arm, day in plan}
    def rng_of(key):
        allv = np.concatenate([f[key] for f in feats.values()])
        allv = allv[~np.isnan(allv)]
        return float(allv.max() - allv.min()) if allv.size else 1.0
    ranges = {k: rng_of(k) for k in ('spend7', 'mood', 'fat', 'rumor7', 'appt7')}

    rows = []                                      # 짝 기록
    for arm, day in plan:
        print(f'  {label} {arm} {day}', flush=True)
        f = feats[(arm, day)]
        D_mem = np.mean([tv_dist(f['yact']), tv_dist(f['hist7']),
                         numeric_dist(f['spend7'], ranges['spend7']),
                         numeric_dist(f['mood'], ranges['mood']), numeric_dist(f['fat'], ranges['fat'])], axis=0)
        D_int = np.mean([numeric_dist(f['rumor7'], ranges['rumor7']),
                         numeric_dist(f['appt7'], ranges['appt7'])], axis=0)
        pol = policy_feats(arm, day)
        D_pol = np.mean(pol, axis=0) if pol else None
        groups = {'persona': D_persona, 'place': D_place, 'memory': D_mem, 'interaction': D_int}
        if D_pol is not None:
            groups['policy'] = D_pol
        D = np.mean(list(groups.values()), axis=0)
        act, out, spv, tot, ok = outcomes(arm, day)
        nsp = outcomes.last_nsp
        valid = np.where(ok)[0]
        Dv = D[np.ix_(valid, valid)].copy()
        np.fill_diagonal(Dv, np.inf)
        nn = valid[np.argmin(Dv, axis=1)]
        pairs = [(valid, nn, 'nn')]
        for _ in range(R_RANDOM):
            j = rng.choice(valid, size=valid.size)
            same = j == valid
            while same.any():
                j[same] = rng.choice(valid, size=same.sum())
                same = j == valid
            pairs.append((valid, j, 'rand'))
        for I, J, typ in pairs:
            rec = {'run': label, 'arm': arm, 'day': day, 'type': typ, 'i': I, 'j': J,
                   'dist': D[I, J],
                   'act': jsd_sim(act[I], act[J]), 'out': jsd_sim(out[I], out[J]),
                   'spend': jsd_sim(spv[I], spv[J]), 'total': total_sim(tot[I], tot[J]),
                   'nsp': np.minimum(nsp[I], nsp[J])}
            for g in GROUPS:
                rec['d_' + g] = groups[g][I, J] if g in groups else np.full(I.size, np.nan)
            rows.append(rec)

    # 같은 사람 재실행: 정책 갈래 첫날, 아직 받지 않은 사람
    if label in CEILING_OK:
        who = [a for a in aids if d['policy'].get(('on', a, start), {}).get('received', 0) == 0]
        I = np.array([idx[a] for a in who])
        a1, o1, s1, t1, k1 = outcomes('on', start, who)
        a2, o2, s2, t2, k2 = outcomes('off', start, who)
        keep = k1 & k2
        rows.append({'run': label, 'arm': 'self', 'day': start, 'type': 'self', 'i': I[keep], 'j': I[keep],
                     'dist': np.zeros(keep.sum()), 'act': jsd_sim(a1[keep], a2[keep]),
                     'out': jsd_sim(o1[keep], o2[keep]), 'spend': jsd_sim(s1[keep], s2[keep]),
                     'total': total_sim(t1[keep], t2[keep]),
                     **{'d_' + g: np.zeros(keep.sum()) for g in GROUPS}})
    return {'label': label, 'N': N, 'start': start, 'cats': cats, 'rows': rows}


# ---------------------------------------------------------------- 요약 통계
def flat(rows, cond):
    sel = [r for r in rows if cond(r)]
    if not sel:
        return None
    keys = ['i', 'dist', *OUTCOMES, *('d_' + g for g in GROUPS)]
    return {k: np.concatenate([r[k] for r in sel]) for k in keys}


def agent_sums(F, key, N):
    v = F[key]
    ok = ~np.isnan(v)
    S = np.bincount(F['i'][ok], weights=v[ok], minlength=N)
    C = np.bincount(F['i'][ok], minlength=N).astype(float)
    return S, C


def summarize(res, rng):
    N = res['N']
    rows = res['rows']
    out = {'run': res['label'], 'N': N, 'outcomes': {}}
    W = rng.multinomial(N, np.ones(N) / N, size=N_BOOT).astype(float)   # 사람 단위 재표집 가중치 (공유)
    all_nn = flat(rows, lambda r: r['type'] == 'nn')
    all_rd = flat(rows, lambda r: r['type'] == 'rand')
    d1_nn = flat(rows, lambda r: r['type'] == 'nn' and r['arm'] == 'off' and r['day'] == res['start'])
    d1_rd = flat(rows, lambda r: r['type'] == 'rand' and r['arm'] == 'off' and r['day'] == res['start'])
    self_ = flat(rows, lambda r: r['type'] == 'self')
    for o in OUTCOMES:
        rec = {}
        for name, F in (('nn', all_nn), ('rand', all_rd), ('nn_d1', d1_nn), ('rand_d1', d1_rd), ('self_d1', self_)):
            if F is None:
                continue
            S, C = agent_sums(F, o, N)
            pt = S.sum() / C.sum()
            bs = (W @ S) / (W @ C)
            rec[name] = [float(pt), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            rec[name + '_bs'] = bs
            rec[name + '_n'] = int(C.sum())
        if 'self_d1' in rec:
            num_ = rec['nn_d1_bs'] - rec['rand_d1_bs']
            den = rec['self_d1_bs'] - rec['rand_d1_bs']
            pt = (rec['nn_d1'][0] - rec['rand_d1'][0]) / (rec['self_d1'][0] - rec['rand_d1'][0])
            nz = num_ / den
            rec['normalized'] = [float(pt), float(np.percentile(nz, 2.5)), float(np.percentile(nz, 97.5))]
        # 최근접 − 무작위 (전체 날짜)
        diff = rec['nn_bs'] - rec['rand_bs']
        rec['nn_minus_rand'] = [float(rec['nn'][0] - rec['rand'][0]), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
        for k in [k for k in rec if k.endswith('_bs')]:
            del rec[k]

        # 용량-반응: 무작위 짝을 조건 거리 10분위로
        F = all_rd
        ok = ~np.isnan(F[o])
        q = np.quantile(F['dist'][ok], np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(q, F['dist'], side='right') - 1, 0, 9)
        bins = []
        for k in range(10):
            sel = ok & (b == k)
            S = np.bincount(F['i'][sel], weights=F[o][sel], minlength=N)
            C = np.bincount(F['i'][sel], minlength=N).astype(float)
            bs = (W[:300] @ S) / (W[:300] @ C)
            bins.append({'x': float(F['dist'][sel].mean()), 'y': float(S.sum() / C.sum()),
                         'lo': float(np.percentile(bs, 2.5)), 'hi': float(np.percentile(bs, 97.5)), 'n': int(sel.sum())})
        rec['dose'] = bins
        ys = [bb['y'] for bb in bins]
        rec['monotone_violations'] = int(sum(1 for k in range(9) if ys[k + 1] > ys[k]))
        # 스피어만(무작위 짝, 표본 8만)
        okk = np.where(ok)[0]
        sub = rng.choice(okk, size=min(80000, okk.size), replace=False)
        from scipy.stats import spearmanr
        rec['spearman'] = float(spearmanr(F['dist'][sub], F[o][sub]).correlation)

        # 묶음별 기여: 표준화 거리로 OLS, 사람 단위 붓스트랩
        gs = [g for g in GROUPS if not np.all(np.isnan(F['d_' + g][ok]))]
        Xr = np.column_stack([F['d_' + g][ok] for g in gs])
        gok = ~np.isnan(Xr).any(1)
        Xr = Xr[gok]
        mu, sd = Xr.mean(0), Xr.std(0)
        sd[sd == 0] = 1
        X = np.column_stack([np.ones(len(Xr)), (Xr - mu) / sd])
        y = F[o][ok][gok]
        ii = F['i'][ok][gok]
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
        out['outcomes'][o] = rec
    out['cats'] = res['cats']
    return out


def main(cache_dir, out_json):
    rng = np.random.default_rng(SEED)
    results = []
    for label in RUNS:
        res = analyze_run(label, cache_dir, rng)
        s = summarize(res, rng)
        results.append(s)
        json.dump(s, open(out_json.replace('.json', f'_{label}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        o = s['outcomes']
        print(f"{label}: N={s['N']} | " + ' | '.join(
            f"{k}: nn {o[k]['nn'][0]:.3f} rand {o[k]['rand'][0]:.3f} ρ {o[k]['spearman']:+.3f}"
            + (f" norm {o[k]['normalized'][0]:.2f}" if 'normalized' in o[k] else '')
            for k in OUTCOMES), flush=True)
    json.dump(results, open(out_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

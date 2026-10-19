"""L1 기준선 — 단위별 경계 찾기. 어디서·언제·무엇에·누가·사람 위치를 굵은 단위에서 세밀한 단위로 내려가며 잰다.

같은 자(척도)
  S_sim  : 시뮬레이션 분포와 실측의 가까움 = 1 - JSD(log2)
  S_base : 단순 모형(입력만으로 기계적으로 배분 / 균등 / 상가 수 비례 중 가장 나은 것)에서
           시뮬레이션과 같은 건수를 뽑아 잰 가까움(100회 평균) — 같은 표본 잡음 조건
  S_ceil : 실측 분포에서 같은 건수를 뽑아 잰 가까움(100회) — 이 표본 크기에서 완벽한 모형이 낼 수 있는 값
  점수   = (S_sim - S_base) / (S_ceil - S_base)   0 = 단순 모형, 1 = 완벽한 모형
  판정   ● 재현: S_sim 이 완벽한 모형의 95% 범위 안
         ◐ 부분 재현: 시뮬 붓스트랩 하한 > 단순 모형 상위 2.5%
         ○ 재현 안 됨: 시뮬 붓스트랩 상한 < 단순 모형 상위 2.5% 또는 S_sim <= S_base
         △ 불확실: 그 사이
주 분석은 5개 본 실행을 합친 9,855명(정책 없는 14일), 실측은 2024년 연간 합. 실험별로는 계절을 맞춘 분기로 다시 판정.
"""
import json
import pickle
import sys
import zipfile
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, 'code')
from l1_baseline import (PROJ, RUNS, GROUPS, TRUTH_MAP, SLOTS, DOW, AGES, SLOT_HOURS, sim_group, slot_of,  # noqa: E402
                         read_any)

SEED = 20261019
NDRAW = 100
NBOOT = 200
REGION = {}
for r, gus in {'도심권': ['11110', '11140', '11170'],
               '동북권': ['11200', '11215', '11230', '11260', '11290', '11305', '11320', '11350'],
               '서북권': ['11380', '11410', '11440'],
               '서남권': ['11470', '11500', '11530', '11545', '11560', '11590', '11620'],
               '동남권': ['11650', '11680', '11710', '11740']}.items():
    for g in gus:
        REGION[g] = r
BIG3 = {'음식점': '음식', '카페·디저트': '음식', '주점': '음식', '편의점': '소매', '마트·식료품': '소매', '의류·잡화': '소매',
        '의료': '서비스', '미용': '서비스', '여가·오락': '서비스', '교육': '서비스'}

# 시뮬 세부 업종 → 63업종 (키워드, 앞에서부터)
SIM_SVC = [
    ('커피-음료', ['커피', '아메리카노', '카페', '라떼', '음료', '차']), ('제과점', ['베이커리', '제과', '빵', '케이크', '디저트']),
    ('한식음식점', ['한식', '백반', '국밥', '찌개']), ('중식음식점', ['중식']), ('일식음식점', ['일식', '초밥', '돈까스']),
    ('양식음식점', ['양식', '파스타', '피자']), ('분식전문점', ['분식', '떡볶이', '김밥']), ('치킨전문점', ['치킨']),
    ('패스트푸드점', ['패스트푸드', '버거', '햄버거']), ('편의점', ['편의점']), ('육류판매', ['정육', '고기']),
    ('청과상', ['청과', '과일']), ('수산물판매', ['수산', '생선']), ('반찬가게', ['반찬']),
    ('슈퍼마켓', ['슈퍼', '장보기', '식료품', '마트', '생필품', '종합소매']), ('의약품', ['약국', '약']),
    ('치과의원', ['치과']), ('한의원', ['한의원']), ('일반의원', ['의원', '병원', '진료']), ('미용실', ['미용실', '헤어', '이발']),
    ('네일숍', ['네일']), ('피부관리실', ['피부']), ('화장품', ['화장품']), ('일반의류', ['의류', '옷']), ('신발', ['신발']),
    ('가방', ['가방']), ('안경', ['안경']), ('노래방', ['노래']), ('PC방', ['PC방', '피시방']), ('당구장', ['당구']),
    ('골프연습장', ['골프']), ('스포츠클럽', ['스포츠', '헬스', '요가', '필라테스', '수영']), ('일반교습학원', ['학원', '교습']),
    ('호프-간이주점', ['주점', '호프', '술']),
]
BDC_SVC = {'한식': '한식음식점', '중식': '중식음식점', '일식': '일식음식점', '양식': '양식음식점', '패스트푸드': '패스트푸드점',
           '제과점': '제과점', '커피전문점': '커피-음료', '편의점': '편의점', '슈퍼마켓': '슈퍼마켓', '할인점/슈퍼마켓': '슈퍼마켓',
           '정육점': '육류판매', '약국': '의약품', '일반병원': '일반의원', '치과병원': '치과의원', '한의원': '한의원',
           '미용실': '미용실', '미용서비스': '미용실', '화장품': '화장품', '의복/의류': '일반의류', '안경': '안경',
           '시계/귀금속': '시계및귀금속', '학원': '일반교습학원', '노래방': '노래방', '스포츠시설': '스포츠클럽'}
BDC_GROUP_ONLY = {'기타요식': '음식점', '농수산물': '마트·식료품', '기타음/식료품': '마트·식료품', '기타식품': '마트·식료품',
                  '기타의료': '의료', '패션/잡화': '의류·잡화', '실내골프/헬스': '여가·오락', '취미/오락': '여가·오락'}


def sim_svc(cat, sub, grp):
    s = sub or ''
    for svc, keys in SIM_SVC:
        if any(k in s for k in keys) and TRUTH_MAP.get(svc) == grp:
            return svc
    return f'미분류:{grp}'


def jsd_sim(p, q):
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(p > 0, p * np.log2(p / m), 0).sum()
        b = np.where(q > 0, q * np.log2(q / m), 0).sum()
    return 1 - 0.5 * (a + b)


# ------------------------------------------------------------------ 실측
def load_truth():
    df = pd.read_csv(PROJ + '서울시 상권분석서비스(추정매출-행정동)_2024년.csv', encoding='cp949')
    df = df[df['서비스_업종_코드_명'].map(TRUTH_MAP).notna()].copy()
    out = pd.DataFrame({'q': df['기준_년분기_코드'].values, 'dong': df['행정동_코드'].astype(str).values,
                        'svc': df['서비스_업종_코드_명'].values})
    out['grp'] = out['svc'].map(TRUTH_MAP)
    out['big3'] = out['grp'].map(BIG3)
    out['gu'] = out['dong'].str[:5]
    out['region'] = out['gu'].map(REGION)
    out['amt'] = df['당월_매출_금액'].values.astype(float)
    for k, s in enumerate(SLOTS):
        out[f'slot{k}'] = df[f'시간대_{s}_매출_금액'].values.astype(float)
    for k, w in enumerate(DOW):
        out[f'dow{k}'] = df[f'{w}요일_매출_금액'].values.astype(float)
    out['sex0'] = df['남성_매출_금액'].values.astype(float)
    out['sex1'] = df['여성_매출_금액'].values.astype(float)
    for k, a in enumerate(AGES):
        out[f'age{k}'] = df[f'연령대_{a}_매출_금액'].values.astype(float)
    return out


def truth_vec(T, dims):
    """dims: 공간·업종 열 + 많아야 하나의 {slot, dow, wk, sex, age}."""
    extra = [d for d in dims if d in ('slot', 'dow', 'wk', 'sex', 'age')]
    base = [d for d in dims if d not in extra]
    if not extra:
        return T.groupby(base)['amt'].sum() if base else pd.Series([T['amt'].sum()])
    e = extra[0]
    if e == 'wk':
        tmp = T[base].copy() if base else pd.DataFrame(index=T.index)
        parts = []
        for k, cols in ((0, [f'dow{i}' for i in range(5)]), (1, ['dow5', 'dow6'])):
            x = tmp.copy()
            x['wk'] = k
            x['v'] = T[cols].sum(1)
            parts.append(x)
        x = pd.concat(parts)
        return x.groupby(base + ['wk'])['v'].sum()
    n = {'slot': 6, 'dow': 7, 'sex': 2, 'age': 5}[e]
    parts = []
    for k in range(n):
        x = T[base].copy() if base else pd.DataFrame(index=T.index)
        x[e] = k
        x['v'] = T[f'{e}{k}']
        parts.append(x)
    x = pd.concat(parts)
    keys = [d if d != e else e for d in dims]
    return x.groupby(keys)['v'].sum()


# ------------------------------------------------------------------ 시뮬·입력
def load_sim():
    z = zipfile.ZipFile('raw/PERSONAS/seoul_personas_20261018.zip')
    P = {p['agent_id']: p for p in json.loads(z.read([n for n in z.namelist() if '14881' in n][0]))}
    ev, base = [], []
    agents = []
    for run in RUNS:
        d = pickle.load(open(f'cache2/{run}.pkl', 'rb'))
        m = d['manifest']
        days = [(date.fromisoformat(m['pre_start']) + timedelta(days=k)).isoformat() for k in range(14)]
        for a in sorted(d['agents']):
            prof = d['agents'][a]
            gid = len(agents)
            sex = 0 if prof.get('gender') == 'M' else 1
            age = {'20대': 0, '30대': 1, '40대': 2, '50대': 3, '60대': 4, '70대이상': 4}.get(prof.get('age_group') or '', -1)
            home = a.split('_')[1]
            agents.append((run, a, sex, age, home))
            for day in days:
                src = 'on' if day < m['start'] else 'off'
                dw = date.fromisoformat(day).weekday()
                for (c, trig, sp, tm, sub, dong) in d['data'][(src, a)]['plans'].get(day, []):
                    if sp <= 0:
                        continue
                    g = sim_group(c, sub)
                    s_ = slot_of(tm)
                    if g is None or s_ is None or len(dong) < 8:
                        continue
                    ev.append((gid, run, dw, s_, g, sim_svc(c, sub, g), dong, sex, age, sp))
            # 입력만으로 배분: 평일·주말 하루 소비액 × 입력 업종 비율, 집 동, 06~24시 균등
            sp_ = (P.get(a) or {}).get('spending') or {}
            for dw in range(7):
                we = dw >= 5
                amt = float(sp_.get('daily_spending_weekend' if we else 'daily_spending_weekday') or 0) * 2
                mix = sp_.get('weekend_top_categories' if we else 'weekday_top_categories') or {}
                tot = sum(mix.values()) or 1
                for name, share in mix.items():
                    name = name.strip()
                    svc = BDC_SVC.get(name)
                    g = TRUTH_MAP.get(svc) if svc else BDC_GROUP_ONLY.get(name)
                    if not g:
                        continue
                    base.append((gid, run, dw, g, svc or f'미분류:{g}', home, sex, age, amt * share / tot))
    E = pd.DataFrame(ev, columns=['agent', 'run', 'dow', 'slot', 'grp', 'svc', 'dong', 'sex', 'age', 'amt'])
    B = pd.DataFrame(base, columns=['agent', 'run', 'dow', 'grp', 'svc', 'dong', 'sex', 'age', 'amt'])
    for X in (E, B):
        X['gu'] = X['dong'].str[:5]
        X['region'] = X['gu'].map(REGION)
        X['big3'] = X['grp'].map(BIG3)
        X['wk'] = (X['dow'] >= 5).astype(int)
    return E, B, agents


def load_stores():
    st = read_any(PROJ + '소상공인시장진흥공단_상가(상권)정보_서울_202512.csv',
                  usecols=['행정동코드', '상권업종대분류명', '상권업종중분류명', '상권업종소분류명'])
    mid = st['상권업종중분류명'].str.strip()
    small = st['상권업종소분류명'].fillna('')
    g = pd.Series(None, index=st.index, dtype=object)
    g[mid.isin(['한식', '중식', '일식', '서양식', '동남아시아', '기타 외국', '기타 간이', '구내식당·뷔페'])] = '음식점'
    g[mid == '비알코올'] = '카페·디저트'
    g[mid == '주점'] = '주점'
    g[(mid == '종합 소매') & small.str.contains('편의점')] = '편의점'
    g[((mid == '종합 소매') & ~small.str.contains('편의점')) | (mid == '식료품 소매')] = '마트·식료품'
    g[mid.isin(['섬유·의복·신발 소매', '시계·귀금속 소매'])] = '의류·잡화'
    g[(mid == '의약·화장품 소매') & small.str.contains('약')] = '의료'
    g[(mid == '의약·화장품 소매') & small.str.contains('화장품')] = '미용'
    g[mid == '의원'] = '의료'
    g[mid.isin(['이용·미용', '욕탕·신체관리'])] = '미용'
    g[mid == '일반 교육'] = '교육'
    g[mid.isin(['유원지·오락', '스포츠 서비스'])] = '여가·오락'
    S = pd.DataFrame({'dong': st['행정동코드'].astype(str), 'grp': g}).dropna()
    S['gu'] = S['dong'].str[:5]
    S['region'] = S['gu'].map(REGION)
    S['big3'] = S['grp'].map(BIG3)
    S['amt'] = 1.0
    return S


# ------------------------------------------------------------------ 한 공간
def evaluate(dims, E, B, T, S, agent_count, rng, spatial_store=True):
    t = truth_vec(T, dims)
    t = t[t > 0]
    cells = t.index
    def vec(X, cols):
        if X is None or len(X) == 0:
            return np.zeros(len(cells))
        v = X.groupby(cols)['amt'].sum()
        return v.reindex(cells).fillna(0).values
    tv = t.values.astype(float)
    sv = vec(E, dims)
    # 칸 밖으로 나간 시뮬 질량(실측에 없는 칸)도 불일치로 센다
    s_out = E['amt'].sum() - sv.sum()
    tv_ = np.append(tv, 0.0)
    sv_ = np.append(sv, max(s_out, 0.0))
    S_sim = jsd_sim(sv_, tv_)
    n = len(E)
    amts = E['amt'].values
    # 단순 모형 후보들
    cands = {}
    if 'slot' in dims:
        w = SLOT_HOURS / SLOT_HOURS.sum()
        Bx = pd.concat([B.assign(slot=k, amt=B['amt'] * w[k]) for k in range(6) if w[k] > 0])
    else:
        Bx = B
    if not set(dims) - set(Bx.columns):
        bv = vec(Bx, dims)
        b_out = Bx['amt'].sum() - bv.sum()
        cands['입력 배분'] = np.append(bv, max(b_out, 0.0))
    cands['균등'] = np.append(np.ones(len(cells)), 0.0)
    sp = [d for d in dims if d in ('region', 'gu', 'dong', 'grp', 'big3')]
    rest = [d for d in dims if d not in sp]
    if spatial_store and any(d in ('region', 'gu', 'dong') for d in dims) and set(rest) <= {'slot', 'dow', 'wk'}:
        st = S.groupby(sp)['amt'].sum().rename('stores').reset_index()
        cf = cells.to_frame(index=False) if isinstance(cells, pd.MultiIndex) else pd.DataFrame({dims[0]: list(cells)})
        cf.columns = dims
        mg = cf.merge(st, on=sp, how='left')
        v = mg['stores'].fillna(0).values.astype(float)
        for e in rest:
            if e == 'slot':
                v = v * (SLOT_HOURS / SLOT_HOURS.sum())[mg['slot'].values]
            elif e == 'wk':
                v = v * np.where(mg['wk'].values == 1, 2 / 7, 5 / 7)
            # dow: 균등(곱해도 같은 비율)
        cands['상가 수 비례'] = np.append(v, 0.0)

    def draw(p, k=NDRAW):
        p = np.asarray(p, float)
        p = p / p.sum()
        out = []
        for _ in range(k):
            cnt = rng.multinomial(n, p)
            idx = np.repeat(np.arange(len(p)), cnt)
            v = np.bincount(idx, weights=rng.choice(amts, size=n), minlength=len(p))
            out.append(jsd_sim(v, tv_))
        return np.array(out)

    base_draws = {k: draw(v) for k, v in cands.items()}
    best = max(base_draws, key=lambda k: base_draws[k].mean())
    bd = base_draws[best]
    cd = draw(tv_)
    # 시뮬 붓스트랩(사람 단위)
    cell_index = pd.Series(np.arange(len(cells)), index=cells)
    key = E[dims].apply(tuple, axis=1) if len(dims) > 1 else E[dims[0]]
    ci = key.map(cell_index).fillna(len(cells)).astype(int).values
    ag = E['agent'].values
    ids, inv = np.unique(ag, return_inverse=True)
    bs = []
    for _ in range(NBOOT):
        w = rng.multinomial(len(ids), np.ones(len(ids)) / len(ids))[inv]
        v = np.bincount(ci, weights=amts * w, minlength=len(cells) + 1)
        bs.append(jsd_sim(v, tv_))
    bs = np.array(bs)
    S_base, S_ceil = bd.mean(), cd.mean()
    score = (S_sim - S_base) / (S_ceil - S_base) if S_ceil > S_base else np.nan
    lo, hi = np.percentile(bs, [2.5, 97.5])
    if S_sim >= np.percentile(cd, 2.5):
        verdict = '●'
    elif lo > np.percentile(bd, 97.5):
        verdict = '◐'
    elif hi < np.percentile(bd, 97.5) or S_sim <= S_base:
        verdict = '○'
    else:
        verdict = '△'
    return {'dims': dims, 'cells': int(len(cells)), 'n_events': int(n), 'S_sim': float(S_sim), 'S_sim_ci': [float(lo), float(hi)],
            'S_base': float(S_base), 'base_name': best, 'S_base_all': {k: float(v.mean()) for k, v in base_draws.items()},
            'S_ceil': float(S_ceil), 'S_ceil_band': [float(np.percentile(cd, 2.5)), float(np.percentile(cd, 97.5))],
            'score': float(score), 'verdict': verdict}


LADDERS = {
    '어디서': [('권역 5', ['region']), ('자치구 25', ['gu']), ('행정동 424', ['dong']), ('행정동×업종묶음', ['dong', 'grp'])],
    '언제': [('주중/주말', ['wk']), ('요일 7', ['dow']), ('시간대 6', ['slot']), ('업종묶음×시간대', ['grp', 'slot']),
             ('업종묶음×요일', ['grp', 'dow']), ('행정동×시간대', ['dong', 'slot'])],
    '무엇에': [('대분류 3', ['big3']), ('업종묶음 10', ['grp']), ('세부 업종 63', ['svc']), ('자치구×업종묶음', ['gu', 'grp'])],
    '누가': [('성별 2', ['sex']), ('연령대 5', ['age']), ('연령×업종묶음', ['grp', 'age']), ('연령×자치구', ['gu', 'age']),
             ('연령×행정동', ['dong', 'age'])],
}


def presence(rng):
    """평일 14시 위치 분포 — 생활인구와 비교. 단순 모형 = 밤 위치(집)."""
    lp = pd.read_csv(PROJ + 'LOCAL_PEOPLE_DONG_202603.csv', encoding='utf-8-sig', index_col=False, usecols=[0, 1, 2, 3])
    lp['wd'] = pd.to_datetime(lp['기준일ID'].astype(str)).dt.dayofweek
    lp = lp[(lp['wd'] < 5) & (lp['시간대구분'] == 14)]
    lp['dong'] = lp['행정동코드'].astype(str)
    truth = lp.groupby('dong')['총생활인구수'].mean()
    rows = []
    for run in RUNS:
        d = pickle.load(open(f'cache2/{run}.pkl', 'rb'))
        m = d['manifest']
        days = [(date.fromisoformat(m['pre_start']) + timedelta(days=k)).isoformat() for k in range(14)]
        for a in d['agents']:
            for day in days:
                if date.fromisoformat(day).weekday() >= 5:
                    continue
                src = 'on' if day < m['start'] else 'off'
                its = sorted(d['data'][(src, a)]['plans'].get(day, []), key=lambda x: x[3])
                if not its:
                    continue
                night = its[0][5]
                at = [x for x in its if x[3] and x[3] <= '14:00']
                loc = (at[-1] if at else its[0])[5]
                if len(night) == 8 and len(loc) == 8:
                    rows.append((f'{run}:{a}', loc, night))
    X = pd.DataFrame(rows, columns=['agent', 'dong', 'night'])
    X['amt'] = 1.0
    out = []
    for name, lev in (('권역 5', 'region'), ('자치구 25', 'gu'), ('행정동 424', 'dong')):
        def lvl(s):
            g = s.str[:5]
            return g.map(REGION) if lev == 'region' else g if lev == 'gu' else s
        t = truth.groupby(lvl(pd.Series(truth.index, index=truth.index))).sum()
        t = t[t > 0]
        sv = X.groupby(lvl(X['dong']))['amt'].sum().reindex(t.index).fillna(0).values
        nv = X.groupby(lvl(X['night']))['amt'].sum().reindex(t.index).fillna(0).values
        tv = t.values
        n = len(X)
        S_sim = jsd_sim(sv, tv)
        def draw(p):
            return np.array([jsd_sim(rng.multinomial(n, p / p.sum()).astype(float), tv) for _ in range(NDRAW)])
        bd, cd = draw(nv + 1e-12), draw(tv)
        ag = X['agent'].values
        ids, inv = np.unique(ag, return_inverse=True)
        idx = pd.Series(np.arange(len(t)), index=t.index)
        ci = lvl(X['dong']).map(idx).fillna(len(t)).astype(int).values
        bs = np.array([jsd_sim(np.bincount(ci, weights=rng.multinomial(len(ids), np.ones(len(ids)) / len(ids))[inv].astype(float),
                                           minlength=len(t) + 1)[:len(t)], tv) for _ in range(NBOOT)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        S_base, S_ceil = bd.mean(), cd.mean()
        if S_sim >= np.percentile(cd, 2.5):
            v = '●'
        elif lo > np.percentile(bd, 97.5):
            v = '◐'
        elif hi < np.percentile(bd, 97.5) or S_sim <= S_base:
            v = '○'
        else:
            v = '△'
        out.append({'scale': name, 'cells': int(len(t)), 'n_events': int(n), 'S_sim': float(S_sim), 'S_sim_ci': [float(lo), float(hi)],
                    'S_base': float(S_base), 'base_name': '밤 위치(집)', 'S_ceil': float(S_ceil), 'score': float((S_sim - S_base) / (S_ceil - S_base)),
                    'verdict': v})
        print(f"  사람 위치 {name}: sim {S_sim:.3f} base {S_base:.3f} ceil {S_ceil:.3f} → {v}", flush=True)
    return out


def main_per_run(out_json):
    rng = np.random.default_rng(SEED + 1)
    T = load_truth()
    E, B, agents = load_sim()
    S = load_stores()
    res = json.load(open(out_json, encoding='utf-8'))
    for run, q in RUNS.items():
        Er, Br, Tq = E[E['run'] == run], B[B['run'] == run], T[T['q'] == q]
        res['per_run'][run] = {}
        for dim, ladder in LADDERS.items():
            res['per_run'][run][dim] = []
            for name, dims in ladder:
                r = evaluate(dims, Er, Br, Tq, S, 2000, rng)
                res['per_run'][run][dim].append({'scale': name, 'verdict': r['verdict'], 'score': r['score'], 'S_sim': r['S_sim'],
                                                 'S_sim_ci': r['S_sim_ci'], 'S_base': r['S_base'], 'S_base_all': r['S_base_all'],
                                                 'S_ceil': r['S_ceil'], 'S_ceil_band': r['S_ceil_band']})
        print(run, 'done', flush=True)
    json.dump(res, open(out_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def main(out_json):
    rng = np.random.default_rng(SEED)
    T = load_truth()
    E, B, agents = load_sim()
    S = load_stores()
    print('events', len(E), 'base rows', len(B), 'agents', len(agents),
          '미분류 세부업종 몫', round(E.loc[E['svc'].str.startswith('미분류'), 'amt'].sum() / E['amt'].sum(), 3), flush=True)
    res = {'pooled': {}, 'per_run': {}, 'presence': None}
    Tall = T                                             # 2024 연간
    for dim, ladder in LADDERS.items():
        res['pooled'][dim] = []
        for name, dims in ladder:
            r = evaluate(dims, E, B, Tall, S, len(agents), rng)
            r['scale'] = name
            res['pooled'][dim].append(r)
            print(f"{dim} {name}: sim {r['S_sim']:.3f} [{r['S_sim_ci'][0]:.3f},{r['S_sim_ci'][1]:.3f}] base {r['S_base']:.3f}({r['base_name']}) "
                  f"ceil {r['S_ceil']:.3f} score {r['score']:+.2f} {r['verdict']} cells {r['cells']} n {r['n_events']}", flush=True)
    # 실험별(계절 맞춘 분기)
    for run, q in RUNS.items():
        Er, Br = E[E['run'] == run], B[B['run'] == run]
        Tq = T[T['q'] == q]
        res['per_run'][run] = {}
        for dim, ladder in LADDERS.items():
            res['per_run'][run][dim] = []
            for name, dims in ladder:
                r = evaluate(dims, Er, Br, Tq, S, 2000, rng)
                res['per_run'][run][dim].append({'scale': name, 'verdict': r['verdict'], 'score': r['score'], 'S_sim': r['S_sim'],
                                                 'S_sim_ci': r['S_sim_ci'], 'S_base': r['S_base'], 'S_base_all': r['S_base_all'],
                                                 'S_ceil': r['S_ceil'], 'S_ceil_band': r['S_ceil_band']})
        print(run, {dim: ''.join(x['verdict'] for x in v) for dim, v in res['per_run'][run].items()}, flush=True)
    res['presence'] = presence(rng)
    json.dump(res, open(out_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[2] == '--per-run':
        main_per_run(sys.argv[1])
    else:
        main(sys.argv[1])

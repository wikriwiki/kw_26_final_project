"""L1 기준선 모사 검증 — 정책이 없는 일상(정책 전 1주 + 정책 없음 갈래 1주)의 소비가 실측과 맞는가.

판정 규칙(결과를 보기 전에 정함): 각 패턴에서 시뮬레이션이 '단순 모형'보다 실측에 가까우면 통과.
가까움 = 1 - JSD(log2) (0~1, 1 이면 분포가 같다). 사람 단위 붓스트랩 500회로 (시뮬 - 단순) 차이의 95% 구간.
범위: 시뮬레이션이 다루는 10개 생활 업종(기타 제외)으로 세 자료(시뮬·실측·페르소나 입력)를 맞춘다.

패턴                     실측                                  단순 모형
 ① 시간대별 지출         상권분석 추정매출 시간대 6구간           깨어 있는 06~24시에 고르게
 ② 요일별 지출           추정매출 요일별                          7일 균등
 ③ 업종 구성             추정매출 63업종 → 10묶음                 (a) 10묶음 균등 (b) 페르소나 입력 업종 비율
 ④ 성별·연령별 지출       추정매출 성별·연령대(20대 이상)          명부 인원수 비례
 ⑤ 자치구별 지출         추정매출 행정동 → 구                     (a) 사는 구에서 다 씀 (b) 상가 수 비례
 ⑥ 평일 낮/밤 인구 비    생활인구 2026-03 평일 14시/04시          하루 종일 집(비 1)
실측 분기는 계절을 맞춰 2024년 같은 분기를 쓴다(소비쿠폰 3분기, 상생·거리두기 4분기, 긴급재난·재난긴급생활비 2분기).
"""
import json
import pickle
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJ = 'C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/'
RUNS = {'P010': 20243, 'P012': 20244, 'P013': 20242, 'P017': 20242, 'DISTANCING': 20244}
GROUPS = ['음식점', '카페·디저트', '주점', '편의점', '마트·식료품', '의료', '미용', '의류·잡화', '여가·오락', '교육']
GI = {g: k for k, g in enumerate(GROUPS)}
SLOTS = ['00~06', '06~11', '11~14', '14~17', '17~21', '21~24']
SLOT_HOURS = np.array([0, 5, 3, 3, 4, 3], float)
DOW = ['월', '화', '수', '목', '금', '토', '일']
AGES = ['20', '30', '40', '50', '60_이상']
N_BOOT = 500
SEED = 20261019

TRUTH_MAP = {}
for g, names in {
    '음식점': ['한식음식점', '양식음식점', '분식전문점', '중식음식점', '일식음식점', '패스트푸드점', '치킨전문점'],
    '카페·디저트': ['커피-음료', '제과점'], '주점': ['호프-간이주점'], '편의점': ['편의점'],
    '마트·식료품': ['슈퍼마켓', '수산물판매', '청과상', '반찬가게', '육류판매', '미곡판매'],
    '의료': ['일반의원', '의약품', '치과의원', '한의원', '의료기기'],
    '미용': ['미용실', '화장품', '피부관리실', '네일숍'],
    '의류·잡화': ['일반의류', '신발', '가방', '안경', '시계및귀금속', '섬유제품'],
    '여가·오락': ['노래방', 'PC방', '골프연습장', '당구장', '스포츠클럽'],
    '교육': ['일반교습학원', '예술학원', '외국어학원', '스포츠 강습'],
}.items():
    for n in names:
        TRUTH_MAP[n] = g

BDC_RULES = [  # 페르소나 입력(카드 자료 업종명) → 묶음, 앞에서부터 처음 맞는 것
    ('카페·디저트', ['커피', '제과']), ('주점', ['유흥주점', '기타유흥']), ('편의점', ['편의점']),
    ('음식점', ['한식', '양식', '일식', '중식', '패스트푸드', '기타요식']),
    ('마트·식료품', ['슈퍼마켓', '정육', '농수산', '기타음/식료품', '기타식품']),
    ('의료', ['병원', '약국', '기타의료', '한의원']), ('미용', ['화장품', '미용']),
    ('의류·잡화', ['의복', '패션', '안경', '시계']),
    ('여가·오락', ['취미/오락', '골프', '헬스', '스포츠시설', '노래방', '영화', '공연']),
    ('교육', ['학원', '유아교육', '독서실']),
]
SIM_SUB_RULES = [
    ('카페·디저트', ['카페', '커피', '아메리카노', '디저트', '베이커리', '제과', '케이크']),
    ('의료', ['약국', '의원', '병원', '한의원', '치과', '보건']),
    ('미용', ['미용', '네일', '화장품', '피부', '헤어']),
    ('여가·오락', ['노래방', 'PC방', '당구', '스포츠', '헬스', '골프', '볼링', '오락']),
    ('교육', ['학원', '교습']), ('주점', ['주점', '호프', '술']), ('편의점', ['편의점']),
    ('마트·식료품', ['슈퍼', '장보기', '식료품', '정육', '청과', '마트', '반찬', '생필품', '종합소매']),
    ('의류·잡화', ['의류', '신발', '가방', '안경', '잡화']),
]
SIM_CAT = {'식사': '음식점', '식비': '음식점', '카페': '카페·디저트', '디저트': '카페·디저트', '주점': '주점',
           '편의점': '편의점', '마트': '마트·식료품', '건강': '의료', '미용': '미용', '쇼핑': '의류·잡화',
           '여가': '여가·오락', '교육': '교육'}


def rule_group(name, rules):
    for g, keys in rules:
        if any(k in name for k in keys):
            return g
    return None


def sim_group(cat, sub):
    g = rule_group(sub or '', SIM_SUB_RULES)
    if g:
        return g
    return SIM_CAT.get(cat)


def slot_of(hhmm):
    try:
        h = int(hhmm[:2])
    except ValueError:
        return None
    return 0 if h < 6 else 1 if h < 11 else 2 if h < 14 else 3 if h < 17 else 4 if h < 21 else 5


def sim1(p, q):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(p > 0, p * np.log2(p / m), 0).sum()
        b = np.where(q > 0, q * np.log2(q / m), 0).sum()
    return 1 - 0.5 * (a + b)


def read_any(path, **kw):
    for enc in ('utf-8', 'utf-8-sig', 'cp949'):
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    raise ValueError(path)


def load_truth():
    df = pd.read_csv(PROJ + '서울시 상권분석서비스(추정매출-행정동)_2024년.csv', encoding='cp949')
    df['grp'] = df['서비스_업종_코드_명'].map(TRUTH_MAP)
    df['gu'] = df['행정동_코드'].astype(str).str[:5]
    cov = df.groupby('기준_년분기_코드').apply(lambda x: x.loc[x['grp'].notna(), '당월_매출_금액'].sum() / x['당월_매출_금액'].sum())
    df = df[df['grp'].notna()]
    T = {}
    for q, x in df.groupby('기준_년분기_코드'):
        T[q] = {
            'slot': np.array([x[f'시간대_{s}_매출_금액'].sum() for s in SLOTS]),
            'dow': np.array([x[f'{d}요일_매출_금액'].sum() for d in DOW]),
            'grp': np.array([x.loc[x['grp'] == g, '당월_매출_금액'].sum() for g in GROUPS]),
            'sex': np.array([x['남성_매출_금액'].sum(), x['여성_매출_금액'].sum()]),
            'age': np.array([x[f'연령대_{a}_매출_금액'].sum() for a in AGES]),
            'gu': x.groupby('gu')['당월_매출_금액'].sum().to_dict(),
            'coverage': float(cov[q]),
        }
    lp = pd.read_csv(PROJ + 'LOCAL_PEOPLE_DONG_202603.csv', encoding='utf-8-sig', index_col=False, usecols=[0, 1, 2, 3])
    lp['wd'] = pd.to_datetime(lp['기준일ID'].astype(str)).dt.dayofweek
    lp = lp[lp['wd'] < 5]
    lp['gu'] = lp['행정동코드'].astype(str).str[:5]
    pres = lp[lp['시간대구분'].isin([4, 14])].groupby(['gu', '시간대구분'])['총생활인구수'].mean().unstack()
    live = {g: float(r[14] / r[4]) for g, r in pres.iterrows()}
    st = read_any(PROJ + '소상공인시장진흥공단_상가(상권)정보_서울_202512.csv', usecols=['시군구코드', '상권업종대분류명'])
    stores = st['시군구코드'].astype(str).str[:5].value_counts().to_dict()
    return T, live, stores


def bdc_prior(profile):
    v = np.zeros(len(GROUPS))
    try:
        mix = json.loads(profile.get('spending_top_wd_json') or '{}')
    except ValueError:
        mix = {}
    for name, share in mix.items():
        g = rule_group(name.strip(), BDC_RULES)
        if g:
            v[GI[g]] += float(share)
    s = v.sum()
    w = float(profile.get('s_daily_wd') or 0)
    return v / s * w if s > 0 else v


def analyze(label, T, live, stores, rng):
    d = pickle.load(open(f'cache2/{label}.pkl', 'rb'))
    m = d['manifest']
    pre, start = m['pre_start'], m['start']
    days = [(date.fromisoformat(pre) + timedelta(days=k)).isoformat() for k in range(14)]
    aids = sorted(d['agents'])
    N = len(aids)
    gus = sorted(T[RUNS[label]]['gu'])
    gui = {g: k for k, g in enumerate(gus)}
    slot = np.zeros((N, 6))
    dow = np.zeros((N, 7))
    grp = np.zeros((N, len(GROUPS)))
    gu = np.zeros((N, len(gus)))
    tot = np.zeros(N)
    prior = np.zeros((N, len(GROUPS)))
    home = np.zeros(N, int)
    sex = np.zeros(N, int)
    age = np.full(N, -1)
    day14 = np.zeros((N, len(gus)))
    night = np.zeros((N, len(gus)))
    oos = 0.0
    for i, a in enumerate(aids):
        p = d['agents'][a]
        prior[i] = bdc_prior(p)
        home_gu = a.split('_')[1][:5]
        home[i] = gui.get(home_gu, -1)
        sex[i] = 0 if p.get('gender') == 'M' else 1
        ag = p.get('age_group') or ''
        age[i] = {'20대': 0, '30대': 1, '40대': 2, '50대': 3, '60대': 4, '70대이상': 4}.get(ag, -1)
        for day in days:
            src = 'on' if day < start else 'off'
            items = d['data'][(src, a)]['plans'].get(day, [])
            wd = date.fromisoformat(day).weekday()
            for (c, trig, sp, tm, sub, dong) in items:
                if sp <= 0:
                    continue
                g = sim_group(c, sub)
                if g is None:
                    oos += sp
                    continue
                s_ = slot_of(tm)
                if s_ is None:
                    continue
                slot[i, s_] += sp
                dow[i, wd] += sp
                grp[i, GI[g]] += sp
                if dong[:5] in gui:
                    gu[i, gui[dong[:5]]] += sp
                tot[i] += sp
            # 평일 낮(14시)·밤(첫 일정) 위치
            if wd < 5 and items:
                its = sorted(items, key=lambda x: x[3])
                first = its[0][5][:5]
                at14 = [x for x in its if x[3] and x[3] <= '14:00']
                loc14 = (at14[-1] if at14 else its[0])[5][:5]
                if first in gui and loc14 in gui:
                    night[i, gui[first]] += 1
                    day14[i, gui[loc14]] += 1
    W = rng.multinomial(N, np.ones(N) / N, size=N_BOOT).astype(float)
    Tq = T[RUNS[label]]
    t_gu = np.array([Tq['gu'][g] for g in gus], float)
    homev = np.zeros((N, len(gus)))
    for i in range(N):
        if home[i] >= 0:
            homev[i, home[i]] = tot[i]
    store_v = np.array([stores.get(g, 0) for g in gus], float)
    sexv = np.zeros((N, 2)); sexv[np.arange(N), sex] = tot
    agev = np.zeros((N, 5)); ok = age >= 0; agev[np.where(ok)[0], age[ok]] = tot[ok]
    headsex = np.zeros((N, 2)); headsex[np.arange(N), sex] = 1
    headage = np.zeros((N, 5)); headage[np.where(ok)[0], age[ok]] = 1
    uniform_slot = SLOT_HOURS / SLOT_HOURS.sum()

    def boot(simM, truth, baseM=None, base_fixed=None):
        pt = sim1(simM.sum(0), truth)
        bs = np.array([sim1(w @ simM, truth) for w in W])
        rec = {'sim': [float(pt), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        if baseM is not None:
            bp = sim1(baseM.sum(0), truth)
            bb = np.array([sim1(w @ baseM, truth) for w in W])
        else:
            bp = sim1(base_fixed, truth)
            bb = np.full(N_BOOT, bp)
        diff = bs - bb
        rec['base'] = float(bp)
        rec['diff'] = [float(pt - bp), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
        rec['pass'] = bool(np.percentile(diff, 2.5) > 0)
        return rec

    out = {'run': label, 'N': N, 'quarter': RUNS[label], 'coverage_truth': Tq['coverage'],
           'sim_out_of_scope_share': float(oos / (oos + tot.sum()))}
    out['slot'] = boot(slot, Tq['slot'], base_fixed=uniform_slot)
    out['dow'] = boot(dow, Tq['dow'], base_fixed=np.ones(7) / 7)
    out['grp_uniform'] = boot(grp, Tq['grp'], base_fixed=np.ones(len(GROUPS)) / len(GROUPS))
    out['grp_prior'] = boot(grp, Tq['grp'], baseM=prior)
    out['sex'] = boot(sexv, Tq['sex'], baseM=headsex)
    out['age'] = boot(agev, Tq['age'], baseM=headage)
    out['gu_home'] = boot(gu, t_gu, baseM=homev)
    out['gu_store'] = boot(gu, t_gu, base_fixed=store_v / store_v.sum())
    # 자치구 순위 상관
    sv = gu.sum(0)
    out['gu_spearman_sim'] = float(spearmanr(sv, t_gu).correlation)
    out['gu_spearman_home'] = float(spearmanr(homev.sum(0), t_gu).correlation)
    out['gu_spearman_store'] = float(spearmanr(store_v, t_gu).correlation)
    # 생활인구 낮/밤 비
    lv = np.array([live.get(g, np.nan) for g in gus])
    def ratio_err(wv):
        dn = (wv @ day14) if wv is not None else day14.sum(0)
        nt = (wv @ night) if wv is not None else night.sum(0)
        r = np.divide(dn, nt, out=np.full(len(gus), np.nan), where=nt > 0)
        okk = ~np.isnan(r) & ~np.isnan(lv) & (r > 0)
        return float(np.mean(np.abs(np.log(r[okk]) - np.log(lv[okk])))), r
    err, r = ratio_err(None)
    errs = np.array([ratio_err(w)[0] for w in W])
    okk = ~np.isnan(r) & ~np.isnan(lv)
    naive = float(np.mean(np.abs(np.log(lv[okk]))))
    out['daynight'] = {'sim_err': [err, float(np.percentile(errs, 2.5)), float(np.percentile(errs, 97.5))],
                       'naive_err': naive, 'pass': bool(np.percentile(errs, 97.5) < naive),
                       'spearman': float(spearmanr(r[okk], lv[okk]).correlation),
                       'sim_ratio': {g: float(x) for g, x in zip(gus, r)}, 'real_ratio': {g: float(x) for g, x in zip(gus, lv)}}
    # 그림용 분포
    out['dist'] = {
        'slot': {'sim': (slot.sum(0) / slot.sum()).tolist(), 'truth': (Tq['slot'] / Tq['slot'].sum()).tolist(), 'base': uniform_slot.tolist()},
        'dow': {'sim': (dow.sum(0) / dow.sum()).tolist(), 'truth': (Tq['dow'] / Tq['dow'].sum()).tolist()},
        'grp': {'sim': (grp.sum(0) / grp.sum()).tolist(), 'truth': (Tq['grp'] / Tq['grp'].sum()).tolist(), 'prior': (prior.sum(0) / prior.sum()).tolist()},
        'sex': {'sim': (sexv.sum(0) / sexv.sum()).tolist(), 'truth': (Tq['sex'] / Tq['sex'].sum()).tolist(), 'head': (headsex.sum(0) / N).tolist()},
        'age': {'sim': (agev.sum(0) / agev.sum()).tolist(), 'truth': (Tq['age'] / Tq['age'].sum()).tolist(), 'head': (headage.sum(0) / headage.sum()).tolist()},
        'gu': {'gus': gus, 'sim': (sv / sv.sum()).tolist(), 'truth': (t_gu / t_gu.sum()).tolist(), 'home': (homev.sum(0) / homev.sum()).tolist()},
    }
    return out


def main(out_json):
    rng = np.random.default_rng(SEED)
    T, live, stores = load_truth()
    res = []
    for lbl in RUNS:
        o = analyze(lbl, T, live, stores, rng)
        res.append(o)
        f = lambda k: f"{o[k]['sim'][0]:.3f} vs {o[k]['base']:.3f} {'통과' if o[k]['pass'] else '미달'}"
        print(f"{lbl}: 시간대 {f('slot')} | 요일 {f('dow')} | 업종(균등) {f('grp_uniform')} | 업종(입력) {f('grp_prior')} | "
              f"성 {f('sex')} | 연령 {f('age')} | 구(집) {f('gu_home')} | 구(상가) {f('gu_store')} | "
              f"ρ구 sim {o['gu_spearman_sim']:.2f} home {o['gu_spearman_home']:.2f} store {o['gu_spearman_store']:.2f} | "
              f"낮밤 err {o['daynight']['sim_err'][0]:.3f} vs {o['daynight']['naive_err']:.3f} ρ {o['daynight']['spearman']:.2f} "
              f"| 범위밖 시뮬 {o['sim_out_of_scope_share']:.3f} 실측 {1-o['coverage_truth']:.3f}", flush=True)
    json.dump(res, open(out_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main(sys.argv[1])

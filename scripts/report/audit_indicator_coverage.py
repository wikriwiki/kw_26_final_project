"""모든 정책의 검증지표를 채점기가 실제로 계산할 수 있는지 전수 점검한다.

    python scripts/report/audit_indicator_coverage.py

**이 점검이 없어서 지표 일곱 개가 조용히 죽어 있었다.** `DS-6`·`EM-4` 는 매 런
"관측부족" 으로 찍혔는데, 실은 순위 대상 이름(`발달상권`·`준내구재`)이 우리 업종
분류에 아예 없어서였다. `P016` 은 지표 넷 중 셋이 미구현인 채로 큐에 들어가 있었다 —
그대로 돌았으면 몇 시간을 버리고 아무 값도 못 얻었다.

다섯 가지를 본다. 앞 셋은 **계산 가능한가**, 뒤 둘은 **잴 준비가 됐는가** 다.

    ① metric 이름을 채점기가 아는가        metric_values 가 None 을 주지 않는가
    ② 업종 이름이 실제로 있는가            sector_spend:X · sector_share:X 의 X
    ③ 순위 대상이 실제 업종인가            rank: [A, B] 의 A·B
    ④ 적격 판정에 쓸 정책 파일이 있는가     `elig` 를 쓰는 지표가 있는 정책만
    ⑤ 채점 창이 등록돼 있는가              없으면 돌릴 때 즉석에서 정하게 된다

④ 가 없으면 DB 백필값(상생 기준)으로 떨어진다 — **다른 정책의 자로 재게 된다.**
긴급재난 EM-2 가 그렇게 여드레를 갔고, P010-2 도 같은 자리에 있었다.
⑤ 는 창 오류가 두 번 난 자리다(P016 시행일 오인 · P090 창 이동).

`expect: info` 는 채점 대상이 아니므로 통과로 본다 — 다만 그 이유가 채점표에
적혀 있어야 한다.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORING = ROOT / 'data/experiments/scoring_table.json'

# 캐시백 전용 경로로 처리되는 지표 (score_policy.main 안에 분기가 있다)
SPECIAL = {'threshold_reach_rate', 'cashback_per_capita', 'cap_reach_rate'}

# 적격 판정(`elig`)을 쓰는 지표. 이런 지표가 있으면 정책 파일이 있어야 한다 —
# 없으면 채점기가 DB 백필값(상생 기준)으로 떨어져 다른 정책의 자로 잰다.
NEEDS_ELIG = {'elig_spend_paired', 'coupon_elig_spend_paired',
              'elig_spend_pre', 'elig_spend_post', 'excl_spend_paired',
              'elig_spend_share'}

# 그래프에서 읽은 실제 이름. 바뀌면 여기도 바꿔야 한다 —
#   MATCH (c:Category) RETURN DISTINCT c.parent, c.name
#   MATCH (p:POI) RETURN DISTINCT p.sangsaeng_kdi
L1 = {'건강', '교육', '기타', '디저트', '마트', '미용', '쇼핑', '식사', '여가', '주점', '카페', '편의점'}
KDI = {'가전·가구', '기타', '여행·레저', '요식', '유통', '이·미용', '학원'}
SUB = set('''PC방 가구 가전·통신 건강보조식품 건자재 고용서비스 광고 교육지원 구내식당·뷔페
기술서비스 기타개인 기타교육 기타보건 기타상품 기타숙박 기타식사 기타외국 네일 노래방 담배 당구
대여 디자인 마사지 문구 미용실 반려동물 법무 베이커리 병원 볼링 부동산 분식 사무서비스 사업서비스
사진 생활용품 세탁 수리 수산 수의 숙박 슈퍼마켓 스포츠 시계·귀금속 시설관리 식료품 식물·꽃 아시안
아이스크림 안경 약국 양식 여행사 오락용품 욕탕·신체관리 유원지·오락 음료소매 의류 의원 이륜차 인쇄
일반주점 일식 자동차부품 장례 장식품 전문서비스 정육 조경 조사 종합소매 주유소 중고상품 중식 차량정비
청과 청소 치과 치킨 카페 컨설팅 통신 편의점 피부관리 피자 학원 한식 한의원 헬스장 호프 화장품
회계·세무'''.split())

POLICIES = ['P010', 'P012', 'EMERGENCY_2020', 'LOCAL_VOUCHER', 'SECTOR_VOUCHER_2020',
            'DISTANCING_2020', 'GATHERING_2020', 'P016']


def load_scorer():
    sys.path.insert(0, str(ROOT / 'scripts' / 'sim'))
    spec = importlib.util.spec_from_file_location('sp', ROOT / 'scripts/sim/score_policy.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    m = load_scorer()
    valid = L1 | SUB | KDI | set(m.GROUP)

    def name_ok(k):
        return all(x.strip() in valid for x in str(k).split('|') if x.strip())

    d = json.loads(io.open(SCORING, encoding='utf-8').read())
    bad, total = [], 0
    if not a.quiet:
        print('%-20s %-7s %-5s %-44s %s' % ('정책', '지표', '기대', 'metric', '판정'))
        print('-' * 112)
    for key in POLICIES:
        for i in ((d.get(key) or {}).get('indicators') or []):
            total += 1
            name, expect, iid = i.get('metric'), i.get('expect'), i.get('id')
            if expect == 'info':
                how, ok = '채점 대상 아님(등록)', True
            elif expect == 'rank':
                ks = i.get('rank') or []
                miss = [k for k in ks if not name_ok(k)]
                ok = (len(ks) == 2 and not miss)
                how = '순위 OK' if ok else '**순위 대상이 업종이 아니다: %s**' % (miss or ks)
            elif name in SPECIAL:
                ok, how = True, '캐시백 경로'
            elif str(name).startswith(('sector_spend:', 'sector_share:')):
                ok = name_ok(str(name).split(':', 1)[1])
                how = '구현됨' if ok else '**업종 이름이 없다**'
            elif str(name).startswith('sector_share_within:'):
                arg = str(name).split(':', 1)[1]
                tgt, _, den = arg.partition('/')
                ok = name_ok(tgt) and (not den or name_ok(den))
                how = '구현됨' if ok else '**업종 이름이 없다**'
            else:
                ok = m.metric_values(name, [], [], [], [])[0] is not None
                how = '구현됨' if ok else '**미구현**'
            if not a.quiet:
                print('%-20s %-7s %-5s %-44s %s' % (key, iid, expect, name, how))
            if not ok:
                bad.append((key, iid, name, how))
    # ④⑤ 정책 단위 점검 — 지표가 계산돼도 잴 준비가 안 됐을 수 있다
    notready: list[tuple[str, str]] = []
    if not a.quiet:
        print()
        print('%-20s %-10s %-10s %s' % ('정책', '정책파일', '채점창', '판정'))
        print('-' * 92)
    for key in POLICIES:
        blk = d.get(key) or {}
        inds = blk.get('indicators') or []
        wants_elig = any((i.get('metric') in NEEDS_ELIG) for i in inds)
        has_file = bool(blk.get('policy_file'))
        has_win = any(k.startswith('window') for k in blk)
        why = []
        if wants_elig and not has_file:
            why.append('**적격 지표가 있는데 policy_file 이 없다 — 상생 기준으로 떨어진다**')
        if not has_win:
            why.append('**채점 창이 등록돼 있지 않다**')
        if not a.quiet:
            print('%-20s %-10s %-10s %s'
                  % (key, ('있음' if has_file else ('필요없음' if not wants_elig else '**없음**')),
                     '있음' if has_win else '**없음**', ' · '.join(why) or 'OK'))
        for w in why:
            notready.append((key, w))

    print()
    print('지표 %d개 중 채점 불가 %d개' % (total, len(bad)))
    for k, i, n, h in bad:
        print('  %-20s %-7s %-30s %s' % (k, i, n, h))
    print('정책 %d개 중 잴 준비가 안 된 곳 %d건' % (len(POLICIES), len(notready)))
    for k, w in notready:
        print('  %-20s %s' % (k, w))
    return 1 if (bad or notready) else 0


if __name__ == '__main__':
    raise SystemExit(main())

"""인용 대조 규칙별 비교 — 정확 일치 대 유사도 임계값.

자료: 금연구역 근거 인용 인터뷰(K-EXAONE) 3셀 × 1,154명 = 3,462건의 원 응답과 근거 패킷.
현행 규칙(stance_argument._citations): 인용문이 인용한 근거 항목 원문의 부분 문자열이어야 통과.

규칙
  R0 현행 정확 일치         quote in text
  R1 정규화 후 정확 일치     표기만 정리(NFKC, 따옴표 이스케이프·종류, 공백)하고 부분 문자열. 숫자·글자는 그대로.
  R2~R4 유사도 ≥ 0.95/0.90/0.80   원문에서 가장 비슷한 구간과의 부분 일치 비율(fuzzywuzzy partial_ratio 방식, difflib)
재는 것
  (1) 실제 인용 통과율: 인용 단위, 응답 단위(모든 인용 통과)
  (2) 틀린 인용 통과율: R0 를 통과한 진짜 인용을 변형 — 숫자 하나, 한글 한 글자, 뜻 뒤집기 — 했을 때 받아들이는 비율
"""
import difflib
import glob
import gzip
import json
import random
import re
import sys
import unicodedata

NSZ = 'C:/Users/srdyh/sim_data/nsz_final/stance_kexaone'
CELLS = ['pre', 'on_post', 'off_post']
TAUS = [0.95, 0.90, 0.80]
SEED = 20261019
CITE_RE = re.compile(r'"evidence_id"\s*:\s*"([^"]+)"\s*,\s*"quote"\s*:\s*"(.*?)"\s*\}', re.S)


def norm(s):
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('\\"', '"').replace("\\'", "'").replace('\\n', ' ').replace('\\/', '/').replace('\\', '')
    for a in '“”„‟″':
        s = s.replace(a, '"')
    for a in '‘’‚‛′':
        s = s.replace(a, "'")
    return re.sub(r'\s+', ' ', s).strip()


def partial_ratio(q, t):
    """fuzzywuzzy partial_ratio 와 같은 방식: 짧은 쪽을 긴 쪽의 같은 길이 구간들과 비교해 최댓값."""
    if not q or not t:
        return 0.0
    if len(q) > len(t):
        q, t = t, q
    sm = difflib.SequenceMatcher(None, q, t, autojunk=False)
    best = 0.0
    for i, j, n in sm.get_matching_blocks():
        start = max(0, j - i)
        seg = t[start:start + len(q)]
        r = difflib.SequenceMatcher(None, q, seg, autojunk=False).ratio()
        if r > best:
            best = r
            if r >= 0.995:
                break
    return best


def citations_of(obj):
    out = []
    for r in obj.get('reasons') or []:
        if isinstance(r, dict) and 'evidence_id' in r:
            out.append((r['evidence_id'], r.get('quote') or ''))
    arg = obj.get('argument') or {}
    for key in ('personal_situation', 'considerations'):
        for item in arg.get(key) or []:
            for c in (item.get('evidence') or []) if isinstance(item, dict) else []:
                if isinstance(c, dict) and 'evidence_id' in c:
                    out.append((c['evidence_id'], c.get('quote') or ''))
    return out


def rules(quote, text):
    nq, nt = norm(quote), norm(text)
    r = {'R0': quote in text, 'R1': bool(nq) and nq in nt}
    score = 1.0 if r['R1'] else partial_ratio(nq, nt)
    for t in TAUS:
        r[f'≥{t:.2f}'] = score >= t
    return r, score


HANGUL = [chr(c) for c in range(ord('가'), ord('힣') + 1)]
FLIPS = [('non_smoker', 'smoker'), ('비흡연', '흡연'), ('없습니다', '있습니다'), ('없', '있'), ('찬성', '반대'),
         ('증가', '감소'), ('감소', '증가'), ('높', '낮'), ('낮', '높'), ('purchased', 'cancelled'), ('true', 'false')]


def mutate(q, text, kind, rnd):
    if kind == 'digit':
        idx = [k for k, ch in enumerate(q) if ch.isdigit()]
        if not idx:
            return None
        k = rnd.choice(idx)
        new = q[:k] + str((int(q[k]) + rnd.randint(1, 9)) % 10) + q[k + 1:]
    elif kind == 'syllable':
        idx = [k for k, ch in enumerate(q) if '가' <= ch <= '힣']
        if len(idx) < 4:
            return None
        k = rnd.choice(idx[1:-1])
        c = q[k]
        while c == q[k]:
            c = rnd.choice(HANGUL)
        new = q[:k] + c + q[k + 1:]
    else:
        new = None
        for a, b in FLIPS:
            if a in q:
                new = q.replace(a, b, 1)
                break
        if new is None:
            return None
    if new == q or new in text or norm(new) in norm(text):
        return None
    return new


def main(out_json):
    rnd = random.Random(SEED)
    rows = []                       # 인용 단위
    resp = []                       # 응답 단위
    for cell in CELLS:
        for f in sorted(glob.glob(f'{NSZ}/{cell}/records/*.json')):
            rec = json.load(open(f, encoding='utf-8'))
            ev = {it['evidence_id']: it.get('text') or '' for it in rec['evidence_packet']['evidence_items']}
            cid = rec['provenance']['call_id']
            try:
                content = json.load(open(f'{NSZ}/{cell}/calls/{cid}.response.json', encoding='utf-8'))['content'] or ''
            except FileNotFoundError:
                continue
            try:
                obj = json.loads(content)
                cites = citations_of(obj) if isinstance(obj, dict) else []
                parsed = True
            except json.JSONDecodeError:
                cites = []
                for m in CITE_RE.finditer(content):
                    raw = m.group(2)
                    try:
                        q = json.loads('"' + raw + '"')     # 정상 이스케이프면 그대로 풀린다
                    except json.JSONDecodeError:
                        q = raw                             # 깨진 이스케이프는 원문 그대로(형식 실패)
                    cites.append((m.group(1), q))
                parsed = False
            rr = {'cell': cell, 'parsed': parsed, 'n': len(cites), 'status': rec['response_status'], 'pass': {}}
            res_all = []
            for eid, q in cites:
                text = ev.get(eid)
                if text is None:
                    r = {k: False for k in ['R0', 'R1'] + [f'≥{t:.2f}' for t in TAUS]}
                    score = 0.0
                    kind = 'unknown_id'
                else:
                    r, score = rules(q, text)
                    kind = 'ok'
                rows.append({'cell': cell, 'parsed': parsed, 'kind': kind, 'score': score, **r,
                             'q': q, 'eid': eid})
                res_all.append(r)
            for k in ['R0', 'R1'] + [f'≥{t:.2f}' for t in TAUS]:
                rr['pass'][k] = bool(res_all) and all(x[k] for x in res_all)
            resp.append(rr)
    # 틀린 인용 통과율
    texts = {}
    for cell in CELLS:
        for f in sorted(glob.glob(f'{NSZ}/{cell}/records/*.json')):
            rec = json.load(open(f, encoding='utf-8'))
            for it in rec['evidence_packet']['evidence_items']:
                texts[it['evidence_id']] = it.get('text') or ''
    true_cites = list({(r['eid'], r['q']) for r in rows if r['R0'] and r['parsed'] and r['q'].strip()})
    rnd.shuffle(true_cites)
    mut = {}
    for kind in ('digit', 'syllable', 'flip'):
        acc = {k: 0 for k in ['R0', 'R1'] + [f'≥{t:.2f}' for t in TAUS]}
        n = 0
        examples = []
        for eid, q in true_cites:
            if n >= 1500:
                break
            text = texts.get(eid, '')
            m = mutate(q, text, kind, rnd)
            if m is None:
                continue
            r, s = rules(m, text)
            n += 1
            for k in acc:
                acc[k] += int(r[k])
            if len(examples) < 3:
                examples.append({'orig': q[:80], 'mut': m[:80], 'score': round(s, 3)})
        mut[kind] = {'n': n, 'accept': {k: v / n if n else None for k, v in acc.items()}, 'examples': examples}
    # 요약
    keys = ['R0', 'R1'] + [f'≥{t:.2f}' for t in TAUS]
    def rate(sel, k):
        sel = list(sel)
        return sum(1 for x in sel if x[k]) / len(sel) if sel else None
    summary = {
        'n_responses': len(resp), 'n_parsed': sum(r['parsed'] for r in resp),
        'n_citations': len(rows), 'n_citations_parsed': sum(r['parsed'] for r in rows),
        'unknown_id_share': sum(r['kind'] == 'unknown_id' for r in rows) / max(1, len(rows)),
        'cite_pass_parsed': {k: rate((r for r in rows if r['parsed']), k) for k in keys},
        'cite_pass_unparsed': {k: rate((r for r in rows if not r['parsed']), k) for k in keys},
        'resp_pass_parsed': {k: rate(({'x': r['pass'][k]} for r in resp if r['parsed'] and r['n'] > 0), 'x') for k in keys},
        'resp_pass_all': {k: sum(1 for r in resp if r['pass'].get(k)) / len(resp) for k in keys},
        'mutations': mut,
    }
    # 정규화로만 살아나는 인용·임계값으로만 살아나는 인용 예시
    rec_r1 = [r for r in rows if r['R1'] and not r['R0']][:5]
    rec_fz = [r for r in rows if r['≥0.90'] and not r['R1']]
    rnd.shuffle(rec_fz)
    summary['examples_r1_only'] = [{'q': r['q'][:120]} for r in rec_r1]
    summary['examples_fuzzy_only'] = [{'q': r['q'][:120], 'score': round(r['score'], 3)} for r in rec_fz[:8]]
    summary['score_hist_failed'] = {f'{lo:.1f}-{lo + 0.1:.1f}': sum(1 for r in rows if not r['R1'] and r['kind'] == 'ok' and lo <= r['score'] < lo + 0.1)
                                    for lo in [x / 10 for x in range(0, 10)]}
    json.dump(summary, open(out_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith('examples')}, ensure_ascii=False, indent=1))
    print(json.dumps({k: summary[k] for k in ('examples_r1_only', 'examples_fuzzy_only')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])

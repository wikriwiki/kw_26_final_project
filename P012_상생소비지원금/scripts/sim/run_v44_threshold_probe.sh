#!/usr/bin/env bash
# v44 — 문턱까지 남은 거리(W)를 움직여 계획이 그것을 읽는지 본다. 120 호출.
# 프롬프트는 v5 그대로. 바뀌는 것은 맥락의 숫자 하나뿐이다.
set -uo pipefail
cd /data/validation_v3/repo
export PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass
export LLM_BASE_URL=http://localhost:8000/v1
OUT=/data/v44_threshold
LOG=$OUT/v44.log
mkdir -p $OUT
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $LOG; }

say "맥락 생성 (Y 만 옮겨 W 격자)"
/data/venv/bin/python scripts/sim/threshold_response_probe.py \
  --frozen /data/validation_v3/pilot_registered/frozen_inputs.json \
  --out $OUT --prepare-only 2>&1 | tee -a $LOG

say "호출 시작"
/data/venv_sgl/bin/python - <<'PYEOF' 2>&1 | tee -a $LOG
import json, os, re, sys, random
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen
sys.path.insert(0, 'scripts/sim')
from prompts import get

OUT = '/data/v44_threshold'
MODEL = 'LGAI-EXAONE/EXAONE-4.5-33B-AWQ'
BASE = os.environ.get('LLM_BASE_URL', 'http://localhost:8000/v1').rstrip('/')
SEEDS = [1701, 2903]
SYSTEM = get('v5').SYSTEM_PROMPT
cells = json.load(open(OUT + '/cells.json', encoding='utf-8'))['cells']

def call(job):
    cell, seed = job
    body = {'model': MODEL,
            'messages': [{'role': 'system', 'content': SYSTEM},
                         {'role': 'user', 'content': cell['user']}],
            'temperature': 0.7, 'max_tokens': 2200, 'seed': seed,
            'chat_template_kwargs': {'enable_thinking': False}}
    req = Request(BASE + '/chat/completions', data=json.dumps(body).encode(),
                  headers={'Content-Type': 'application/json'})
    try:
        with urlopen(req, timeout=300) as r:
            raw = json.loads(r.read())['choices'][0]['message']['content']
    except Exception as e:
        return dict(cell, seed=seed, error='%s: %s' % (type(e).__name__, e))
    return dict(cell, seed=seed, raw=raw)

jobs = [(c, s) for c in cells for s in SEEDS]
random.Random(20260922).shuffle(jobs)
with ThreadPoolExecutor(max_workers=12) as pool:
    rows = list(pool.map(call, jobs))
with open(OUT + '/responses.jsonl', 'w', encoding='utf-8') as fh:
    for r in rows:
        r = dict(r); r.pop('user', None)
        fh.write(json.dumps(r, ensure_ascii=False) + '\n')
print('응답 %d · 오류 %d' % (len(rows), sum(1 for r in rows if r.get('error'))))
PYEOF

say "=== 판정 ==="
/data/venv/bin/python - <<'PYEOF' 2>&1 | tee -a $LOG
import json, statistics, sys, collections
sys.path.insert(0, '/data/validation_v3/repo/scripts/sim')
from threshold_response_probe import monotone, spread
ELIG = {'식사','카페','디저트','편의점','마트','미용','쇼핑','여가','건강','교육','기타'}
rows = [json.loads(l) for l in open('/data/v44_threshold/responses.jsonl', encoding='utf-8') if l.strip()]
ok = [r for r in rows if not r.get('error')]
prop = collections.defaultdict(list); nev = collections.defaultdict(list)
for r in ok:
    try: obj = json.loads(r['raw'])
    except Exception: continue
    p = obj.get('daily_propensity')
    if isinstance(p, (int, float)) and not isinstance(p, bool):
        prop[r['fraction']].append(float(p))
    evs = obj.get('events') or []
    nev[r['fraction']].append(sum(1 for e in evs if e.get('category') in ELIG))
print('응답 %d / %d' % (len(ok), len(rows)))
print()
print('%-8s %10s %10s %6s' % ('문턱 대비', 'propensity', '적립 이벤트', 'n'))
pts_p = []; pts_n = []
for f in sorted(prop):
    mp = statistics.mean(prop[f]); mn = statistics.mean(nev[f])
    pts_p.append((f, mp)); pts_n.append((f, mn))
    print('%-8.2f %10.3f %10.2f %6d' % (f, mp, mn, len(prop[f])))
print()
for label, pts in (('propensity', pts_p), ('적립 이벤트', pts_n)):
    print('%-12s 단조 감소 %s · 양 끝 차이 %+.3f'
          % (label, '그렇다' if monotone(pts) else '아니다', spread(pts)))
print()
print('사전등록 합격선: 단조 감소 + 양 끝 차이 != 0 이면 "반응한다"')
print('예측했던 것: 반응은 있으나 약하다. 비단조도 가능하다고 미리 적었다.')
PYEOF
say "=== V44_DONE ==="

#!/usr/bin/env bash
# 후보 2 — 확진 수준에 기준(2주 전 대비 배수)을 같이 주면 계획이 달라지는가. 96 호출(복제). 사전등록 experiments/case_trend/design_note.md
#
# **전체 라운드 전에 이것부터 돌린다.** P010 역진의 벽에서 서술 프롬프트가 무효인
# 것을 이미 겪었다 — 문장을 더해도 행동이 안 달라지면 12일 × 200명 × 두 팔이
# 통째로 낭비다. 같은 시민을 그 한 줄만 넣고 빼서 두 번 부른다.
#
# 돌고 있는 본런(p013_ruler)과 GPU 를 함께 쓴다. 24 호출이라 방해가 크지 않다.
set -uo pipefail
cd /data/validation_v3/repo
export PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8
export LLM_BASE_URL=http://localhost:8000/v1
OUT=/data/ct_replicate
LOG=$OUT/probe.log
mkdir -p $OUT
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $LOG; }

say "맥락 생성 — 거리두기 셀에 기준 배수 한 줄만 넣고 뺀다"
/data/venv/bin/python scripts/sim/scope_fact_probe.py \
  --frozen /data/validation_v3/pilot_registered/frozen_inputs.json \
  --out $OUT --candidate case_trend_2020_11_24 2>&1 | tee -a $LOG

say "호출 시작 (24 x 새 시드 4 = 96) — 복제"
/data/venv_sgl/bin/python - <<'PYEOF' 2>&1 | tee -a $LOG
import json, os, sys, random
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen
sys.path.insert(0, 'scripts/sim')
from prompts import get

OUT = '/data/ct_replicate'
MODEL = 'LGAI-EXAONE/EXAONE-4.5-33B-AWQ'
BASE = os.environ.get('LLM_BASE_URL', 'http://localhost:8000/v1').rstrip('/')
# **복제 검정.** 파일럿(시드 1701·2903)과 겹치지 않는 새 시드 넷으로
# 48쌍을 새로 뽑는다. 파일럿과 합치지 않는다 — 합치면 유의해질 때까지 표본을
# 늘린 것이 되어 p 가 부풀려진다(optional stopping).
SEEDS = [3011, 4127, 5233, 6337]   # 파일럿과 겹치지 않는 새 시드 넷
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
random.Random(20260924).shuffle(jobs)
# 본런이 48 워커로 돌고 있다. 탐침은 4 로 눌러 방해를 줄인다.
with ThreadPoolExecutor(max_workers=4) as pool:
    rows = list(pool.map(call, jobs))
with open(OUT + '/responses.jsonl', 'w', encoding='utf-8') as fh:
    for r in rows:
        r = dict(r); r.pop('user', None)
        fh.write(json.dumps(r, ensure_ascii=False) + '\n')
print('응답 %d · 오류 %d' % (len(rows), sum(1 for r in rows if r.get('error'))))
PYEOF

say "요약 — 시드별로 따로 본다"
for S in 3011 4127 5233 6337; do
  say "--- 시드 $S"
  /data/venv/bin/python - "$S" <<'PYEOF' 2>&1 | tee -a $LOG
import json, sys
seed = int(sys.argv[1])
rows = [json.loads(l) for l in open('/data/ct_replicate/responses.jsonl', encoding='utf-8') if l.strip()]
keep = [r for r in rows if r.get('seed') == seed]
with open('/data/ct_replicate/seed_%d.jsonl' % seed, 'w', encoding='utf-8') as fh:
    for r in keep:
        fh.write(json.dumps(r, ensure_ascii=False) + '\n')
print('  이 시드 응답 %d' % len(keep))
PYEOF
  /data/venv/bin/python scripts/sim/scope_fact_probe.py \
    --frozen /data/validation_v3/pilot_registered/frozen_inputs.json \
    --out $OUT --responses $OUT/seed_$S.jsonl 2>&1 | tee -a $LOG
done
say "=== SCOPE_PROBE_DONE ==="

#!/usr/bin/env bash
# 후보 2 — Stage1 의 이벤트가 달라지는가. 48 호출.
# 사전등록 experiments/plan_channel/s1_ownership.md
#
# 같은 사용자 맥락(정책 블록 포함)을 **SYSTEM 만 바꿔** 두 번 부른다.
set -uo pipefail
cd /data/repo
export PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 PYTHONPATH=/data/repo
export LLM_BASE_URL=http://localhost:8000/v1
export PROBE_OUT PROBE_N PROBE_SEEDS PROBE_VARIANT PROBE_WORKERS
OUT=${PROBE_OUT:-/data/s1_own_probe}
LOG=$OUT/probe.log
mkdir -p $OUT
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $LOG; }

say "정책이 켜진 맥락 합성 — 런타임 함수로 렌더"
/data/venv/bin/python scripts/sim/s1_ownership_probe.py --build --out $OUT --n ${PROBE_N:-48} \
  --policy /data/repo/data/neo4j_load/policies/P012.json \
  --frozen /data/validation_v3/pilot_registered/frozen_inputs.json 2>&1 | tee -a $LOG

say "SYSTEM 두 판 덤프"
/data/venv/bin/python - <<'PYEOF' 2>&1 | tee -a $LOG
import sys, io, os
sys.path.insert(0, 'scripts/sim')
# 레지스트리를 안 고친다 — 본런이 도는 저장소라 건드리지 않는다.
# 패키지 안의 모듈을 직접 가져온다.
import importlib
ARMS = ('v5', os.environ.get('PROBE_VARIANT', 'v5own'))
mods = {n: importlib.import_module('prompts.' + n) for n in ARMS}
for name in ARMS:
    s = mods[name].SYSTEM_PROMPT
    io.open(os.environ['PROBE_OUT'] + '/system_%s.txt' % name, 'w',
            encoding='utf-8', newline=chr(10)).write(s)
    print('  %s %d자' % (name, len(s)))
PYEOF

say "호출 시작 (${PROBE_N:-48}칸 x 2판)"
/data/venv_sgl/bin/python - <<'PYEOF' 2>&1 | tee -a $LOG
import json, os
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

OUT = os.environ['PROBE_OUT']
MODEL = 'LGAI-EXAONE/EXAONE-4.5-33B-AWQ'
BASE = os.environ.get('LLM_BASE_URL', 'http://localhost:8000/v1').rstrip('/')
SEEDS = [int(x) for x in os.environ.get('PROBE_SEEDS', '5507').split(',')]
ARMS = ('v5', os.environ.get('PROBE_VARIANT', 'v5own'))
SYS = {n: open('%s/system_%s.txt' % (OUT, n), encoding='utf-8').read()
       for n in ARMS}
cells = json.load(open(OUT + '/cells.json', encoding='utf-8'))['cells']
jobs = [(c, n, s) for c in cells for n in ARMS for s in SEEDS]
print('칸 %d · 시드 %s · 호출 %d' % (len(cells), SEEDS, len(jobs)))

def call(job):
    cell, name, SEED = job
    body = {'model': MODEL,
            'messages': [{'role': 'system', 'content': SYS[name]},
                         {'role': 'user', 'content': cell['user']}],
            'temperature': 0.7, 'max_tokens': 2600, 'seed': SEED,
            'chat_template_kwargs': {'enable_thinking': False}}
    req = Request(BASE + '/chat/completions', data=json.dumps(body).encode(),
                  headers={'Content-Type': 'application/json'})
    rec = {'aid': cell['aid'], 'case': cell.get('case'),
           'date': cell.get('date'), 'arm': name, 'seed': SEED}
    try:
        with urlopen(req, timeout=300) as r:
            rec['raw'] = json.loads(r.read())['choices'][0]['message']['content']
    except Exception as e:
        rec['error'] = '%s: %s' % (type(e).__name__, e)
    return rec

part = OUT + '/responses.jsonl.part'
done = 0
# 본런과 나눠 쓸 때는 4, GPU 가 비면 올린다. 4 로 두면 GPU 가 놀고 탐침이 병목이다.
W = int(os.environ.get('PROBE_WORKERS', '4'))
print('  워커 %d' % W, flush=True)
with open(part, 'w', encoding='utf-8') as fh, ThreadPoolExecutor(max_workers=W) as pool:
    for r in pool.map(call, jobs):
        fh.write(json.dumps(r, ensure_ascii=False) + chr(10))
        fh.flush()
        done += 1
        if done % 8 == 0:
            print('  ... %d/%d' % (done, len(jobs)), flush=True)
os.replace(part, OUT + '/responses.jsonl')
print('응답 완료 %d' % done)
PYEOF

say "판정"
/data/venv/bin/python scripts/sim/s1_ownership_probe.py --out $OUT \
    --responses $OUT/responses.jsonl 2>&1 | tee -a $LOG
say "=== S1_OWN_PROBE_DONE ==="

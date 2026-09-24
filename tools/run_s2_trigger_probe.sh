#!/usr/bin/env bash
# 후보 3 — 계기:policy 가 **그 이벤트의 금액**을 바꾸는가.
# 사전등록 experiments/plan_channel/s2_trigger_candidate.md
#
#   bash tools/run_s2_trigger_probe.sh
#   PROBE_OUT / PROBE_N / PROBE_SEEDS 로 조절
#
# 처음부터 시드 셋으로 돈다 — 후보 2 에서 **불일치 쌍 수가 표본**이라는 것을
# 1차에 7개로 배웠다. 나중에 늘리면 그건 선택적 중단이다.
set -uo pipefail
cd /data/repo
export PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 PYTHONPATH=/data/repo
export LLM_BASE_URL=http://localhost:8000/v1
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass
export PROBE_OUT PROBE_N PROBE_SEEDS
OUT=${PROBE_OUT:-/data/s2_trg_probe}
LOG=$OUT/probe.log
mkdir -p $OUT
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $LOG; }

say "Stage2 맥락 생성 — 같은 시민·같은 이벤트, 계기 낱말만 off/on"
/data/venv/bin/python scripts/sim/s2_trigger_probe.py --build --out $OUT \
  --n ${PROBE_N:-24} 2>&1 | tee -a $LOG

say "호출 시작"
/data/venv_sgl/bin/python - <<'PYEOF' 2>&1 | tee -a $LOG
import json, os
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

OUT = os.environ['PROBE_OUT']
MODEL = 'LGAI-EXAONE/EXAONE-4.5-33B-AWQ'
BASE = os.environ.get('LLM_BASE_URL', 'http://localhost:8000/v1').rstrip('/')
SEEDS = [int(x) for x in os.environ.get('PROBE_SEEDS', '4103').split(',')]
SYSTEM = open(OUT + '/system_s2.txt', encoding='utf-8').read()
cells = json.load(open(OUT + '/cells.json', encoding='utf-8'))['cells']
jobs = [(c, s) for c in cells for s in SEEDS]
print('칸 %d · 시드 %s · 호출 %d' % (len(cells), SEEDS, len(jobs)))

def call(job):
    cell, seed = job
    body = {'model': MODEL,
            'messages': [{'role': 'system', 'content': SYSTEM},
                         {'role': 'user', 'content': cell['user']}],
            'temperature': 0.7, 'max_tokens': 2600, 'seed': seed,
            'chat_template_kwargs': {'enable_thinking': False}}
    req = Request(BASE + '/chat/completions', data=json.dumps(body).encode(),
                  headers={'Content-Type': 'application/json'})
    rec = {'aid': cell['aid'], 'side': cell['side'], 'seed': seed}
    try:
        with urlopen(req, timeout=300) as r:
            rec['raw'] = json.loads(r.read())['choices'][0]['message']['content']
    except Exception as e:
        rec['error'] = '%s: %s' % (type(e).__name__, e)
    return rec

part = OUT + '/responses.jsonl.part'
done = 0
with open(part, 'w', encoding='utf-8') as fh, ThreadPoolExecutor(max_workers=4) as pool:
    for r in pool.map(call, jobs):
        fh.write(json.dumps(r, ensure_ascii=False) + chr(10))
        fh.flush()
        done += 1
        if done % 12 == 0:
            print('  ... %d/%d' % (done, len(jobs)), flush=True)
os.replace(part, OUT + '/responses.jsonl')
print('응답 완료 %d' % done)
PYEOF

say "판정"
/data/venv/bin/python scripts/sim/s2_trigger_probe.py --out $OUT \
    --responses $OUT/responses.jsonl 2>&1 | tee -a $LOG
say "=== S2_TRG_PROBE_DONE ==="

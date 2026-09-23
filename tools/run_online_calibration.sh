#!/usr/bin/env bash
# 배송 몫을 말하기는 하는가 + 기준 평균 — 96 호출. 사전등록 experiments/split_anchor/prereg.md
#
# 본런에 12시간을 붓기 전에 두 가지만 싸게 본다.
#   (a) 응답률 90% 미만이면 **중단**한다 — 안 오는 필드로는 회계를 못 돌린다
#   (b) mean(1 - online_share) 를 재서 KEEP_MEAN 으로 얼린다
#
# 본런(p013_ruler)과 GPU 를 나눠 쓴다. 워커 4로 눌러 방해를 줄인다.
set -uo pipefail
cd /data/validation_v3/repo
export PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8
export LLM_BASE_URL=http://localhost:8000/v1
OUT=/data/online_calib
LOG=$OUT/probe.log
mkdir -p $OUT
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $LOG; }

say "기준 맥락 생성 — 동결 맥락 그대로, 바뀌는 것은 SYSTEM 뿐"
/data/venv/bin/python scripts/sim/online_share_probe.py \
  --frozen /data/validation_v3/pilot_registered/frozen_inputs.json \
  --out $OUT 2>&1 | tee -a $LOG

say "호출 시작 — v5online"
/data/venv_sgl/bin/python - <<'PYEOF' 2>&1 | tee -a $LOG
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen
sys.path.insert(0, 'scripts/sim')
from prompts import get

OUT = '/data/online_calib'
MODEL = 'LGAI-EXAONE/EXAONE-4.5-33B-AWQ'
BASE = os.environ.get('LLM_BASE_URL', 'http://localhost:8000/v1').rstrip('/')
SEED = 8101
SYSTEM = get('v5online').SYSTEM_PROMPT
print('SYSTEM %d자 (v5 대비 +%d)' % (len(SYSTEM), len(SYSTEM) - len(get('v5').SYSTEM_PROMPT)))
cells = json.load(open(OUT + '/cells.json', encoding='utf-8'))['cells']

def call(cell):
    body = {'model': MODEL,
            'messages': [{'role': 'system', 'content': SYSTEM},
                         {'role': 'user', 'content': cell['user']}],
            'temperature': 0.7, 'max_tokens': 2200, 'seed': SEED,
            'chat_template_kwargs': {'enable_thinking': False}}
    req = Request(BASE + '/chat/completions', data=json.dumps(body).encode(),
                  headers={'Content-Type': 'application/json'})
    try:
        with urlopen(req, timeout=300) as r:
            raw = json.loads(r.read())['choices'][0]['message']['content']
    except Exception as e:
        return dict(cell, seed=SEED, error='%s: %s' % (type(e).__name__, e))
    return dict(cell, seed=SEED, raw=raw)

# 오는 대로 쓴다 — 죽으면 다 잃은 적이 있다(8분치).
part = OUT + '/responses.jsonl.part'
done = 0
with open(part, 'w', encoding='utf-8') as fh, ThreadPoolExecutor(max_workers=4) as pool:
    for r in pool.map(call, cells):
        r = dict(r); r.pop('user', None)
        fh.write(json.dumps(r, ensure_ascii=False) + chr(10))
        fh.flush()
        done += 1
        if done % 8 == 0:
            print('  ... %d/%d' % (done, len(cells)), flush=True)
os.replace(part, OUT + '/responses.jsonl')
print('응답 완료 %d' % done)
PYEOF

say "판정"
/data/venv/bin/python scripts/sim/online_share_probe.py \
  --out $OUT --responses $OUT/responses.jsonl 2>&1 | tee -a $LOG
say "=== ONLINE_CALIB_DONE ==="

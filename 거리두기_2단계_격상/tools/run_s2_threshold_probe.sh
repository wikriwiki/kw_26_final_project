#!/usr/bin/env bash
# 문턱 줄이 Stage2 의 금액을 바꾸는가 — 48 호출. 사전등록 experiments/plan_channel/s2_threshold.md
#
# 본런(arm A)과 GPU 를 나눠 쓴다. 워커 4로 눌러 방해를 줄인다.
# 두 팔 모두 sangsaeng_active=True — **다른 것은 헤더의 문턱 줄 하나뿐이다.**
set -uo pipefail
cd /data/repo
export PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 PYTHONPATH=/data/repo
export LLM_BASE_URL=http://localhost:8000/v1
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass
OUT=/data/s2_probe
LOG=$OUT/probe.log
mkdir -p $OUT
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $LOG; }

say "Stage2 맥락 생성 — 같은 시민·같은 이벤트, 문턱 줄만 off/on"
/data/venv/bin/python scripts/sim/s2_threshold_probe.py --build --out $OUT --n 24 2>&1 | tee -a $LOG

say "호출 시작"
/data/venv_sgl/bin/python - <<'PYEOF' 2>&1 | tee -a $LOG
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen
# stage2_poi 를 import 하면 dawn_context -> neo4j 로 끌려가는데
# 이 venv 에는 neo4j 가 없다(실제로 여기서 한 번 죽었다).
# 빌드 단계가 떨군 파일을 읽는다.
SYSTEM_S2 = open('/data/s2_probe/system_s2.txt', encoding='utf-8').read()

OUT = '/data/s2_probe'
MODEL = 'LGAI-EXAONE/EXAONE-4.5-33B-AWQ'
BASE = os.environ.get('LLM_BASE_URL', 'http://localhost:8000/v1').rstrip('/')
SEED = 9301
cells = json.load(open(OUT + '/cells.json', encoding='utf-8'))['cells']
print('칸 %d · SYSTEM_S2 %d자' % (len(cells), len(SYSTEM_S2)))

def call(cell):
    body = {'model': MODEL,
            'messages': [{'role': 'system', 'content': SYSTEM_S2},
                         {'role': 'user', 'content': cell['user']}],
            'temperature': 0.7, 'max_tokens': 2600, 'seed': SEED,
            'chat_template_kwargs': {'enable_thinking': False}}
    req = Request(BASE + '/chat/completions', data=json.dumps(body).encode(),
                  headers={'Content-Type': 'application/json'})
    try:
        with urlopen(req, timeout=300) as r:
            raw = json.loads(r.read())['choices'][0]['message']['content']
    except Exception as e:
        return dict(cell, seed=SEED, error='%s: %s' % (type(e).__name__, e))
    return dict(cell, seed=SEED, raw=raw)

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
/data/venv/bin/python scripts/sim/s2_threshold_probe.py --out $OUT \
    --responses $OUT/responses.jsonl 2>&1 | tee -a $LOG
say "=== S2_PROBE_DONE ==="

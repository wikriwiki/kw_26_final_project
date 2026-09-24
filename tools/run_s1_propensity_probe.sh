#!/usr/bin/env bash
# 후보 6 (v5self) — 소비성향이 **사람을 타는가**. 사전등록 experiments/plan_channel/prereg_v5self.md
#
# 같은 사용자 맥락(정책 블록 포함)을 **SYSTEM 만 바꿔** 두 번 부른다.
# 기본 80칸 x 2판 x 시드 3 = 480 호출. 28일 본런과 GPU 를 나눠 쓰므로 워커 4.
#
# **본런이 먼저다.** 워커를 올려 탐침을 빨리 끝내려 하지 않는다 — 본런이 느려진다.
set -uo pipefail
cd /data/repo
export PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 PYTHONPATH=/data/repo
export LLM_BASE_URL=http://localhost:8000/v1
export PROBE_OUT PROBE_N PROBE_SEEDS PROBE_VARIANT PROBE_WORKERS PROBE_THR_FRACS PROBE_DAY
OUT=${PROBE_OUT:-/data/s1_prop_probe}
LOG=$OUT/probe.log
mkdir -p $OUT
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $LOG; }

# 같은 탐침이 둘 돌면 같은 .part 에 써서 자료를 잃는다 — 실제로 한 번 잃었다.
# **ps 를 grep 하지 않는다.** 그렇게 했더니 nohup 줄과 ssh 줄까지 세어 자기 자신
# 때문에 멈췄다(2026-09-24). mkdir 은 원자적이라 자기 자신을 셀 일이 없다.
LOCKDIR=$OUT/.lock
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  say "이미 돌고 있다 — $LOCKDIR 가 있다. 죽은 락이면 지우고 다시 걸어라"
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null' EXIT

# 그래프에서 새로 만든다. 고정 셀(frozen_inputs.json)은 고유 에이전트가 **12명**뿐이라
# 사람 단위 지표를 판정할 수 없다 — 그래서 첫 시도를 중단했다
# (experiments/plan_channel/v5self_probe_aborted.md).
say "맥락 생성 — 층화표본 ${PROBE_N:-120}명, 런타임과 같은 렌더"
export NEO4J_URI=${NEO4J_URI:-bolt://localhost:7687}
export NEO4J_USER=${NEO4J_USER:-neo4j} NEO4J_PASSWORD=${NEO4J_PASSWORD:-exp001pass}
/data/venv/bin/python scripts/sim/make_probe_cells.py --out $OUT \
  --n ${PROBE_N:-120} --day ${PROBE_CELL_DAY:-2021-10-03} \
  --policy /data/repo/data/neo4j_load/policies/P012.json \
  --policy-day ${PROBE_DAY:-2021-10-21} \
  --thr-fracs ${PROBE_THR_FRACS:-0.55,0.70,0.85,1.05} 2>&1 | tee -a $LOG

# 사람 수가 등록한 선(85명) 아래면 돌리지 않는다. 돌려 봤자 r 을 못 믿는다.
NP=$(/data/venv/bin/python -c "
import json; print(len({c['aid'] for c in json.load(open('$OUT/cells.json',encoding='utf-8'))['cells']}))")
say "고유 에이전트 ${NP}명"
if [ "${NP:-0}" -lt 85 ]; then
  say "거부: 등록한 선(85명) 아래다. 표본을 키우기 전에는 돌리지 않는다"
  exit 1
fi

say "SYSTEM 두 판 덤프 — 길이 차이가 곧 후보의 크기다"
/data/venv/bin/python - <<'PYEOF' 2>&1 | tee -a $LOG
import sys, io, os, importlib
sys.path.insert(0, 'scripts/sim')
ARMS = ('v5', os.environ.get('PROBE_VARIANT', 'v5self'))
mods = {n: importlib.import_module('prompts.' + n) for n in ARMS}
L = {}
for name in ARMS:
    s = mods[name].SYSTEM_PROMPT
    L[name] = len(s)
    io.open(os.environ['PROBE_OUT'] + '/system_%s.txt' % name, 'w',
            encoding='utf-8', newline=chr(10)).write(s)
    print('  %s %d자' % (name, len(s)))
d = L[ARMS[1]] - L[ARMS[0]]
print('  덧댄 길이 +%d자' % d)
assert 0 < d < 140, '덧댄 것이 한 문장이 아니다 (+%d자) — 무엇이 들었는지 보라' % d
# v5 가 안 바뀌었는지 — 후보 비교가 성립하는 조건이다
import hashlib
h = hashlib.sha256(mods['v5'].SYSTEM_PROMPT.encode('utf-8')).hexdigest()[:16]
print('  v5 sha256 %s' % h)
assert h == '250311b63adbc4f6', 'v5 가 바뀌었다 — 탐침을 돌리면 안 된다'
PYEOF
[ ${PIPESTATUS[0]:-1} -ne 0 ] && { say "SYSTEM 검증 실패 — 그만둔다"; exit 1; }

say "호출 시작 (${PROBE_N:-80}칸 x 2판 x 시드)"
/data/venv_sgl/bin/python - <<'PYEOF' 2>&1 | tee -a $LOG
import json, os
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

OUT = os.environ['PROBE_OUT']
MODEL = 'LGAI-EXAONE/EXAONE-4.5-33B-AWQ'
BASE = os.environ.get('LLM_BASE_URL', 'http://localhost:8000/v1').rstrip('/')
SEEDS = [int(x) for x in os.environ.get('PROBE_SEEDS', '3301,4409,5519').split(',')]
ARMS = ('v5', os.environ.get('PROBE_VARIANT', 'v5self'))
SYS = {n: open('%s/system_%s.txt' % (OUT, n), encoding='utf-8').read() for n in ARMS}
cells = json.load(open(OUT + '/cells.json', encoding='utf-8'))['cells']
jobs = [(c, n, s) for c in cells for n in ARMS for s in SEEDS]
print('칸 %d · 시드 %s · 호출 %d' % (len(cells), SEEDS, len(jobs)), flush=True)

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
W = int(os.environ.get('PROBE_WORKERS', '4'))
print('  워커 %d (본런과 나눠 쓴다)' % W, flush=True)
with open(part, 'w', encoding='utf-8') as fh, ThreadPoolExecutor(max_workers=W) as pool:
    for r in pool.map(call, jobs):
        fh.write(json.dumps(r, ensure_ascii=False) + chr(10))
        fh.flush()
        done += 1
        if done % 20 == 0:
            print('  ... %d/%d' % (done, len(jobs)), flush=True)
os.replace(part, OUT + '/responses.jsonl')
print('응답 완료 %d' % done)
PYEOF

say "판정 — 시드별로 따로 본다 (한 시드만 맞으면 기각이라고 등록했다)"
for S in $(echo ${PROBE_SEEDS:-3301,4409,5519} | tr ',' ' '); do
  say "--- 시드 $S"
  /data/venv/bin/python - "$S" "$OUT" <<'PYEOF' 2>&1 | tee -a $LOG
import json, sys
seed, out = int(sys.argv[1]), sys.argv[2]
rows = [json.loads(l) for l in open(out + '/responses.jsonl', encoding='utf-8') if l.strip()]
keep = [r for r in rows if r.get('seed') == seed]
with open('%s/seed_%d.jsonl' % (out, seed), 'w', encoding='utf-8') as fh:
    for r in keep:
        fh.write(json.dumps(r, ensure_ascii=False) + '\n')
print('  이 시드 응답 %d' % len(keep))
PYEOF
  /data/venv/bin/python scripts/sim/s1_propensity_probe.py --out $OUT \
      --responses $OUT/seed_$S.jsonl 2>&1 | tee -a $LOG
done

say "--- 전체 합침 (참고용 — 판정은 시드별로 한다)"
/data/venv/bin/python scripts/sim/s1_propensity_probe.py --out $OUT \
    --responses $OUT/responses.jsonl 2>&1 | tee -a $LOG
say "=== S1_PROP_PROBE_DONE ==="

#!/usr/bin/env bash
# v41 — v5 대 v40 을 같은 런에서 원문 계약 관문에 올린다. 384 호출.
#
# v5 의 예전 128/192 와 비교하지 않는다. 서버 모델이 그때와 같다는 보장이 없으므로
# 같은 런에서 다시 잰다. 등록부는 validation_v3.json 을 건드리지 않고 새로 썼다.
set -uo pipefail
cd /data/validation_v3/repo
export PYTHONHASHSEED=0
export PYTHONIOENCODING=utf-8
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass
export LLM_BASE_URL=http://localhost:8000/v1
OUT=/data/validation_v41/run
mkdir -p "$(dirname "$OUT")"
LOG=/data/validation_v41/v41.log
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $LOG; }

say "사전등록 해시 대조"
sha256sum data/experiments/validation_v41.json scripts/sim/prompts/v40.py | tee -a $LOG

say "384 호출 시작 (v5 192 · v40 192)"
/data/venv_sgl/bin/python scripts/sim/validate_prompt_v3.py \
  --config data/experiments/validation_v41.json --out "$OUT" 2>&1 | tee -a $LOG

say "=== 결과 ==="
/data/venv/bin/python - <<'PY' | tee -a $LOG
import json
d = json.load(open('/data/validation_v41/run/summary.json', encoding='utf-8'))
v = d['variants']
print('%-6s %10s %10s %8s %10s %10s' % ('후보','응답','엄격통과','통과율','요청실패','오귀인'))
for name in ('v5','v40'):
    r = v.get(name)
    if not r: continue
    print('%-6s %10s %10s %7.1f%% %10s %10s'
          % (name, '%d/%d' % (r['responses'], r['expected']), r['strict_valid'],
             100*r['strict_valid_rate'], r['failed'], r['false_policy_off_responses']))
a, b = v.get('v5'), v.get('v40')
if a and b:
    print()
    print('사전등록 합격선: v40 통과율 >= v5 통과율')
    print('  v5 %.1f%% · v40 %.1f%% → %s'
          % (100*a['strict_valid_rate'], 100*b['strict_valid_rate'],
             '통과' if b['strict_valid_rate'] >= a['strict_valid_rate'] else '기각'))
    print('등록된 95%% 관문:',
          '둘 다 미달' if max(a['strict_valid_rate'], b['strict_valid_rate']) < 0.95 else '확인 필요')
    print('  → 둘 다 미달이면 v40 이 v5 보다 나아도 "통과했다"고 적지 않는다')
PY
say "=== V41_DONE ==="

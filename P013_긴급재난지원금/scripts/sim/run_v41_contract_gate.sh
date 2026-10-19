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

say "768 호출 시작 (v5 · v40 · v42 · v45 각 192)"
# venv_sgl 에는 neo4j 드라이버가 없다. 이 러너는 Neo4j 에서 시민·zone 을 읽어
# 맥락을 만들므로 /data/venv 로 돈다 (LLM 호출은 urllib 이라 sglang 패키지가 필요 없다).
/data/venv/bin/python scripts/sim/validate_prompt_v3.py \
  --config data/experiments/validation_v41.json --out "$OUT" 2>&1 | tee -a $LOG

say "=== 결과 ==="
/data/venv/bin/python - <<'PY' | tee -a $LOG
import json
d = json.load(open('/data/validation_v41/run/summary.json', encoding='utf-8'))
v = d['variants']
order = [n for n in ('v5', 'v40', 'v42', 'v45') if n in v]
print('%-6s %10s %10s %8s %10s %10s' % ('후보','응답','엄격통과','통과율','요청실패','오귀인'))
for name in order:
    r = v[name]
    print('%-6s %10s %10s %7.1f%% %10s %10s'
          % (name, '%d/%d' % (r['responses'], r['expected']), r['strict_valid'],
             100*r['strict_valid_rate'], r['failed'], r['false_policy_off_responses']))
base = v.get('v5')
if base:
    print()
    print('사전등록 합격선: 각 후보의 통과율 >= v5 의 통과율')
    for name in order[1:]:
        r = v[name]
        print('  %-4s %.1f%% vs v5 %.1f%% → %s'
              % (name, 100*r['strict_valid_rate'], 100*base['strict_valid_rate'],
                 '자격' if r['strict_valid_rate'] >= base['strict_valid_rate'] else '기각'))
    print()
    print('두 변경을 갈라 읽는다')
    if 'v40' in v:
        print('  오염 제거   v40 - v5  = %+.1f%%p'
              % (100*(v['v40']['strict_valid_rate'] - base['strict_valid_rate'])))
    if 'v42' in v and 'v40' in v:
        print('  형식 고침   v42 - v40 = %+.1f%%p'
              % (100*(v['v42']['strict_valid_rate'] - v['v40']['strict_valid_rate'])))
    if 'v45' in v and 'v42' in v:
        print('  지갑 어휘   v45 - v42 = %+.1f%%p'
              % (100*(v['v45']['strict_valid_rate'] - v['v42']['strict_valid_rate'])))
    print()
    best = max(order, key=lambda n: v[n]['strict_valid_rate'])
    print('등록된 95%% 관문: %s'
          % ('전부 미달' if v[best]['strict_valid_rate'] < 0.95 else '%s 도달 — 확인 필요' % best))
    print('  → 전부 미달이면 v5 보다 나아도 "통과했다"고 적지 않는다')
PY
say "=== V41_DONE ==="

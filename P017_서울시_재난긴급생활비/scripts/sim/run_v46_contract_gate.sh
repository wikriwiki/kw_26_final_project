#!/usr/bin/env bash
# v46 — v45 대 v46 을 같은 런에서 원문 계약 관문에 올린다. 384 호출.
#
# v5 의 예전 128/192 와 비교하지 않는다. 서버 모델이 그때와 같다는 보장이 없으므로
# 같은 런에서 다시 잰다. 등록부는 validation_v3.json 을 건드리지 않고 새로 썼다.
set -uo pipefail
cd /data/validation_v3/repo
export PYTHONHASHSEED=0
export PYTHONIOENCODING=utf-8
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass
export LLM_BASE_URL=http://localhost:8000/v1
OUT=/data/validation_v46/run
mkdir -p "$(dirname "$OUT")"
LOG=/data/validation_v46/v46.log
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $LOG; }

say "사전등록 해시 대조"
sha256sum data/experiments/validation_v46.json scripts/sim/prompts/v46.py | tee -a $LOG

say "384 호출 시작 (v45 · v46 각 192)"
# venv_sgl 에는 neo4j 드라이버가 없다. 이 러너는 Neo4j 에서 시민·zone 을 읽어
# 맥락을 만들므로 /data/venv 로 돈다 (LLM 호출은 urllib 이라 sglang 패키지가 필요 없다).
/data/venv/bin/python scripts/sim/validate_prompt_v3.py \
  --config data/experiments/validation_v46.json --out "$OUT" 2>&1 | tee -a $LOG

say "=== 결과 ==="
/data/venv/bin/python - <<'PY' | tee -a $LOG
import json
d = json.load(open('/data/validation_v46/run/summary.json', encoding='utf-8'))
v = d['variants']
order = [n for n in ('v45', 'v46') if n in v]
print('%-6s %10s %10s %8s %10s %10s' % ('후보','응답','엄격통과','통과율','요청실패','오귀인'))
for name in order:
    r = v[name]
    print('%-6s %10s %10s %7.1f%% %10s %10s'
          % (name, '%d/%d' % (r['responses'], r['expected']), r['strict_valid'],
             100*r['strict_valid_rate'], r['failed'], r['false_policy_off_responses']))
base = v.get('v45'); cand = v.get('v46')
if base and cand:
    print()
    print('사전등록 합격선: v46 통과율 >= v45 통과율')
    print('  v46 %.1f%% vs v45 %.1f%% → %s'
          % (100*cand['strict_valid_rate'], 100*base['strict_valid_rate'],
             '자격' if cand['strict_valid_rate'] >= base['strict_valid_rate'] else '기각'))
    print('  시간 규칙 + 예시 정렬   v46 - v45 = %+.1f%%p'
          % (100*(cand['strict_valid_rate'] - base['strict_valid_rate'])))
    print()
    print('등록된 95%% 관문 183/192')
    print('  v45 %d/192 · v46 %d/192 → %s'
          % (base['strict_valid'], cand['strict_valid'],
             '도달' if cand['strict_valid'] >= 183 else '미달'))
    print('  사전등록: 시간만 틀린 10건이 전부 살아나도 181/192 이라 관문은 안 넘는다고 적어 뒀다')
    print()
    print('예측했던 것: 5~8%p 오른다. 관문은 미달. 안 오르면 두 모순이 원인이 아니었던 것이다.')
PY
say "=== V46_DONE ==="

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
OUT=/data/validation_v48/run
mkdir -p "$(dirname "$OUT")"
LOG=/data/validation_v48/v48.log
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $LOG; }

say "사전등록 해시 대조"
sha256sum data/experiments/validation_v48.json scripts/sim/prompts/v48.py scripts/sim/prompts/v49.py | tee -a $LOG

say "1152 호출 시작 (v45 · v48 · v49 각 384)"
# venv_sgl 에는 neo4j 드라이버가 없다. 이 러너는 Neo4j 에서 시민·zone 을 읽어
# 맥락을 만들므로 /data/venv 로 돈다 (LLM 호출은 urllib 이라 sglang 패키지가 필요 없다).
/data/venv/bin/python scripts/sim/validate_prompt_v3.py \
  --config data/experiments/validation_v48.json --out "$OUT" 2>&1 | tee -a $LOG

say "=== 결과 ==="
/data/venv/bin/python - <<'PY' | tee -a $LOG
import json
d = json.load(open('/data/validation_v48/run/summary.json', encoding='utf-8'))
v = d['variants']
order = [n for n in ('v45', 'v48', 'v49') if n in v]
print('%-6s %10s %10s %8s %10s %10s' % ('후보','응답','엄격통과','통과율','요청실패','오귀인'))
for name in order:
    r = v[name]
    print('%-6s %10s %10s %7.1f%% %10s %10s'
          % (name, '%d/%d' % (r['responses'], r['expected']), r['strict_valid'],
             100*r['strict_valid_rate'], r['failed'], r['false_policy_off_responses']))
# 비율 차이가 아니라 짝지은 칸으로 읽는다. 같은 프롬프트가 런마다 38칸을 뒤집는다.
import math, collections
rows = [json.loads(l) for l in open('/data/validation_v48/run/responses.jsonl', encoding='utf-8') if l.strip()]
cells = collections.defaultdict(dict)
for r in rows:
    cells[r['variant']][(r['aid'], r['case'], r['arm'], r['replicate'])] = r['valid']
def mcnemar(a, b):
    ks = set(a) & set(b)
    b01 = sum(1 for k in ks if not a[k] and b[k])
    b10 = sum(1 for k in ks if a[k] and not b[k])
    n = b01 + b10
    if n == 0: return b01, b10, 1.0
    lo = min(b01, b10)
    p = 2 * sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return b01, b10, min(1.0, p)
print()
print('=== 짝지은 비교 (McNemar 정확검정) ===')
print('%-14s %8s %8s %8s %11s %s' % ('비교','개선','악화','순증','p','판정'))
MDE = 21
for x, y in (('v45','v48'), ('v45','v49'), ('v48','v49')):
    if x not in cells or y not in cells: continue
    i2, d2, p2 = mcnemar(cells[x], cells[y])
    verdict = '유의' if p2 < 0.05 else ('검출력 부족' if abs(i2-d2) < MDE else '유의하지 않다')
    print('%-14s %8d %8d %+8d %11.4f %s' % ('%s → %s' % (x, y), i2, d2, i2-d2, p2, verdict))
print()
print('사전등록한 MDE: 후보당 384칸에서 약 21칸(5.6%p).')
print('  주 지표    v48 · v49 각각이 v45 를 이기는가 (0 을 제외)')
print('  부 지표    v48 → v49 — 서로의 비교')
print('  예측       v48 은 조금 오르나 21칸 미만(못 읽는다) · v49 는 0 근처')
print()
best = max(cells, key=lambda n2: sum(cells[n2].values()))
print('등록된 95%% 관문 (후보당 %d칸 중 95%%)' % len(next(iter(cells.values()))))
for n2 in ('v5','v45','v47'):
    if n2 in cells:
        ok = sum(cells[n2].values()); tot = len(cells[n2])
        print('  %-4s %d/%d (%.1f%%) %s' % (n2, ok, tot, 100*ok/tot, '도달' if ok/tot >= 0.95 else '미달'))
print('  → 전부 미달이면 v5 보다 나아도 "통과했다"고 적지 않는다')
PY
say "=== V48_DONE ==="

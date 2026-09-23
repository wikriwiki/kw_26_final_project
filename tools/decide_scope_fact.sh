#!/usr/bin/env bash
# 탐침 결과를 받아 **사전등록한 중단 규칙**으로 바로 분기한다.
#
#   bash tools/decide_scope_fact.sh
#
# 규칙(experiments/scope_fact/prereg.md):
#   changed = 0  →  이 줄은 무력하다. **후보를 버리고 라운드를 돌리지 않는다.**
#   changed > 0  →  제외업종 delta 방향을 보고 전체 라운드를 큐에 넣는다
#
# 규칙은 여기서 새로 정하지 않는다. 결과를 보고 문턱을 고치면 사전등록이 무의미하다.
set -uo pipefail
cd "$(dirname "$0")/.."

KEY=${SIM_KEY:-/c/Users/Administrator/.ssh/outofmemory.pem}
HOST=${SIM_HOST:-outofmemory@123.37.28.167}
PORT=${SIM_PORT:-10022}
OUT=output/scope_probe
mkdir -p "$OUT"
say(){ echo "[$(date +%H:%M)] $*"; }

say "탐침 결과를 받는다"
GOT=0
for F in responses.jsonl summary.json probe.log; do
  if scp -q -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
        "$HOST:/data/scope_probe/$F" "$OUT/$F" 2>/dev/null; then
    echo "  <- $F"
    [ "$F" = "responses.jsonl" ] && GOT=1
  else
    echo "  ·  $F  아직 없음"
  fi
done

if [ $GOT -eq 0 ]; then
  say "아직 응답이 없다 — 그것도 답이다. 끝나고 다시."
  ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$HOST" \
    'tail -3 /data/scope_probe/probe.log 2>/dev/null; pgrep -c -f "[r]un_scope_fact" || true' 2>/dev/null
  exit 0
fi

say "시드별로 따로 읽는다 — 시드 잡음과 그 줄의 효과를 섞지 않는다"
PY=$(command -v py || command -v python3 || command -v python)
for S in 1701 2903; do
  echo
  echo "--- 시드 $S"
  PYTHONIOENCODING=utf-8 "$PY" - "$S" <<'PYEOF'
import io, json, sys
seed = int(sys.argv[1])
rows = [json.loads(l) for l in io.open('output/scope_probe/responses.jsonl', encoding='utf-8') if l.strip()]
keep = [r for r in rows if r.get('seed') == seed]
io.open('output/scope_probe/seed_%d.jsonl' % seed, 'w', encoding='utf-8', newline='\n').writelines(
    json.dumps(r, ensure_ascii=False) + '\n' for r in keep)
print('  응답 %d · 오류 %d' % (len(keep), sum(1 for r in keep if r.get('error'))))
PYEOF
  PYTHONIOENCODING=utf-8 "$PY" scripts/sim/scope_fact_probe.py \
    --frozen /dev/null --out "$OUT" --responses "$OUT/seed_$S.jsonl" || true
done

echo
say "등록된 중단 규칙"
PYTHONIOENCODING=utf-8 "$PY" - <<'PYEOF'
import io, json, sys
sys.path.insert(0, 'scripts/sim')
import importlib.util
spec = importlib.util.spec_from_file_location('probe', 'scripts/sim/scope_fact_probe.py')
pr = importlib.util.module_from_spec(spec); spec.loader.exec_module(pr)

rows = [json.loads(l) for l in io.open('output/scope_probe/responses.jsonl', encoding='utf-8') if l.strip()]
out = {}
for seed in sorted({r.get('seed') for r in rows if r.get('seed') is not None}):
    out[seed] = pr.compare([r for r in rows if r.get('seed') == seed])

changed = sum(v['changed'] for v in out.values())
paired = sum(v['paired'] for v in out.values())
print('  쌍 %d · 계획이 달라진 시민 %d' % (paired, changed))
for seed, v in out.items():
    dp, np_ = v['d_propensity']
    de, ne = v['d_excluded']
    print('   시드 %-5s changed %d/%d · 소비성향 %s · 제외업종 이벤트 %s'
          % (seed, v['changed'], v['paired'],
             ('%+.4f' % dp) if dp is not None else '—',
             ('%+.3f' % de) if de is not None else '—'))
print()
if paired == 0:
    print('  **쌍이 하나도 없다 — 응답을 못 읽었다. 판정하지 않는다.**')
elif changed == 0:
    print('  **계획이 하나도 안 달라졌다 → 이 줄은 무력하다.**')
    print('  사전등록한 중단 규칙대로 후보를 버리고 전체 라운드를 돌리지 않는다.')
    print('  다음: 다른 가설로 간다. 서술이 아니라 계산이어야 한다는 선은 그대로다.')
else:
    ex = [v['d_excluded'][0] for v in out.values() if v['d_excluded'][0] is not None]
    print('  **계획이 달라졌다 → 라운드를 큐에 넣을 자격이 있다.**')
    if ex:
        mean = sum(ex) / len(ex)
        print('  제외업종 이벤트 평균 delta %+.3f — %s'
              % (mean, '줄이기를 멈추는 방향' if mean > 0 else
                       ('여전히 줄이는 방향' if mean < 0 else '변화 없음')))
    print('  주의: 탐침은 Stage1 계획만 본다. 금액은 Stage2 가 정하므로')
    print('        제로섬이 실제로 풀렸는지는 **전체 라운드의 PL-2** 로만 알 수 있다.')
io.open('output/scope_probe/verdict.json', 'w', encoding='utf-8', newline='\n').write(
    json.dumps({'by_seed': {str(k): v for k, v in out.items()},
                'changed_total': changed, 'paired_total': paired},
               ensure_ascii=False, indent=1))
PYEOF

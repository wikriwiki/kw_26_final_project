#!/bin/bash
# v29 — 179명·거리두기 전용·358칸을 씨앗 둘로 계획한다 (716 응답).
# 프롬프트는 v25 그대로. 바뀌는 것은 볼일을 받는 사람 수와 시나리오 범위뿐이다.
set -uo pipefail
cd /data/validation_v3/repo
BASE=/data/validation_v29_cohort179
mkdir -p $BASE
TOK=/data/hf_cache/hub/models--LGAI-EXAONE--EXAONE-4.5-33B-AWQ/snapshots/31e6a965d0661bbe4a8b895e22a77f8271772ba0
SRC=/data/cohort179/source_v29.json
CFG=data/experiments/validation_v29_cohort179_dinein.json
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $BASE/v29.log; }

# 사전등록한 소스를 실제로 쓰고 있는지 먼저 확인한다. 설정에 적힌 해시와 다르면 멈춘다.
WANT=$(python3 -c "import json;print(json.load(open('$CFG'))['source_inputs_sha256'])")
GOT=$(sha256sum $SRC | cut -d' ' -f1)
[ "$WANT" = "$GOT" ] || { say "소스 해시 불일치 — 사전등록 $WANT · 실제 $GOT"; exit 1; }
say "소스 해시 확인 $GOT"

PRE=$BASE/preflight.json
if [ ! -f "$PRE" ]; then
  say "문법 사전검사 시작"
  /data/venv_sgl/bin/python scripts/sim/preflight_grammars.py --source $SRC --config $CFG \
    --tokenizer $TOK --out "$PRE" >> $BASE/preflight.log 2>&1 \
    || { say "사전검사 실패 — 중단한다"; exit 1; }
fi
say "사전검사: 제외 $(python3 -c "import json;d=json.load(open('$PRE'));print(len(d['rejected'])+len(d['unbuildable']))")칸"

ROOT=$BASE/v25d; mkdir -p $ROOT
say "########## 계획 358칸 x 씨앗 2 = 716 ##########"
/data/plan_until_done.sh $CFG $SRC $ROOT/plans_0 "$PRE" 12 \
  || { say "계획 실패"; tail -8 $ROOT/plans_0.log; exit 1; }
say "계획 완료"

for S in 70001 70002; do
  say "===== 씨앗 $S ====="
  /data/venv/bin/python scripts/sim/prepare_purchase_probe.py --run $ROOT/plans_0 \
    --replicate $S --static-offers --out $ROOT/psrc_${S}_0.json > $ROOT/g0_$S.log 2>&1 \
    || { say "$S 게이트 실패"; continue; }
  B0=$(python3 -c "import json;print(json.load(open('$ROOT/psrc_${S}_0.json'))['resource_feasibility_summary']['proven_impossible'])")
  say "  첫 게이트: 증명된 불가능 $B0"
  PR=$ROOT/plans_0; PP=$ROOT/psrc_${S}_0.json
  for N in 1 2 3; do
    /data/venv/bin/python scripts/sim/prepare_action_repair.py --run $PR --resource-source $PP \
      --replicate $S --out $ROOT/src_${S}_$N.json > $ROOT/p_${S}_$N.log 2>&1 || break
    CELLS=$(python3 -c "import json;print(len(json.load(open('$ROOT/src_${S}_$N.json'))['cells']))")
    [ "$CELLS" -eq 0 ] && { say "  수정 $N: 고칠 칸 없음"; break; }
    say "  수정 $N: $CELLS 칸"
    python3 - <<PY
import hashlib,json
from pathlib import Path
root=Path('$ROOT')
base=json.loads(Path('$CFG').read_bytes())
cfg=dict(base,id='v29_${S}_r$N',candidates=[{'id':'r$N','thinking_tokens':512}],seeds=[$S],
  source_inputs_sha256=hashlib.sha256((root/'src_${S}_$N.json').read_bytes()).hexdigest(),
  phase='v29 seed $S repair round $N.')
(root/'cfg_${S}_$N.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PY
    /data/plan_until_done.sh $ROOT/cfg_${S}_$N.json $ROOT/src_${S}_$N.json \
      $ROOT/plans_${S}_$N "$PRE" 12 >> $ROOT/pl_${S}_$N.log 2>&1 || break
    /data/venv/bin/python scripts/sim/prepare_purchase_probe.py --run $ROOT/plans_${S}_$N \
      --replicate $S --static-offers --out $ROOT/psrc_${S}_$N.json > $ROOT/g_${S}_$N.log 2>&1 || break
    PR=$ROOT/plans_${S}_$N; PP=$ROOT/psrc_${S}_$N.json
  done
  S=$S ROOT=$ROOT SRC=$SRC python3 - <<'PY'
import json, os, glob
ROOT=os.environ["ROOT"]; S=os.environ["S"]
paths=[f"{ROOT}/psrc_{S}_0.json"]+sorted(glob.glob(f"{ROOT}/psrc_{S}_[123].json"))
base=None; kept={}
for p in paths:
    d=json.load(open(p,encoding="utf-8"))
    if base is None: base={k:v for k,v in d.items() if k!="cells"}
    for c in d["cells"]:
        k=(c["aid"],c["case"],c["arm"])
        if (c.get("resource_feasibility") or {}).get("impossible_even_with_all_candidates"): continue
        kept.setdefault(k,c)
out=dict(base); out["cells"]=list(kept.values()); out.pop("resource_feasibility_summary",None)
designed=len(json.load(open(os.environ["SRC"],encoding="utf-8"))["cells"])
attempted=len({(c["aid"],c["case"],c["arm"]) for p in paths for c in json.load(open(p,encoding="utf-8"))["cells"]})
out["assembly"]={"kind":"repaired_incomplete_matrix","cells_designed":designed,
                 "cells_reached_gate":attempted,"cells_kept":len(kept),
                 "never_feasible":attempted-len(kept),"lost_before_gate":designed-attempted}
json.dump(out,open(f"{ROOT}/asm_{S}.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
print("조립", len(kept))
PY
  python3 - <<PY
import hashlib,json
from pathlib import Path
root=Path('$ROOT')
base=json.loads(Path('data/experiments/validation_daily_v2_purchase_template.json').read_bytes())
cfg=dict(base,id='v29_${S}_buy',
  source_sha256=hashlib.sha256((root/'asm_$S.json').read_bytes()).hexdigest(),
  phase='v29 seed $S purchases on repaired cells.',
  scope='Distancing only. Repaired incomplete matrix. Not a complete on/off comparison.')
(root/'cfgbuy_$S.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PY
  /data/venv_sgl/bin/python scripts/sim/validate_purchase_probe.py --config $ROOT/cfgbuy_$S.json \
    --source $ROOT/asm_$S.json --out $ROOT/buy_$S --tokenizer $TOK > $ROOT/buy_$S.log 2>&1 \
    || say "  씨앗 $S 결제 경고"
  say "  씨앗 $S : 첫 막힘 $B0 → 조립 $(python3 -c "import json;print(len(json.load(open('$ROOT/asm_$S.json'))['cells']))") → 결제 $(python3 -c "import json;v=json.load(open('$ROOT/buy_$S/summary.json'))['variants'];k=list(v)[0];print(v[k]['valid'],'/',v[k]['responses'])" 2>/dev/null || echo '?')"
done
say "=== V29_DONE ==="

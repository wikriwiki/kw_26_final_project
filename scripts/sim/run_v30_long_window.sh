#!/usr/bin/env bash
# v30 — 한 달을 잰다. 채점표가 지정한 긴 창 실험이고, 지갑을 채워 두고 돈다.
#
# 채점표는 남은 과제 하나(P012 적립업종 0.49배)의 원인을 창 길이로 적고 긴 창을
# 처방했다. 그 처방을 그대로 돌리면 엉뚱한 이유로 실패한다 — 소득이 없어 28일이면
# 3분의 2가 빈털터리가 된다(experiments/THE_PURSE_RUNS_DRY.md). 그래서 소득을 켠다.
#
#   워밍업 10-15~10-17 (3일) · 무정책 10-18~10-24 (7일=한 주) · 정책 10-25~11-14 (21일=세 주)
#
# 두 번 돈다. 정책을 안 실은 같은 창을 함께 돌려 표류를 뺀다. 21일 창에서는 표류가
# 2일 창보다 훨씬 크게 쌓이므로 이 대조가 없으면 아무것도 못 읽는다.
#
# 채점은 **각 런 직후**에 한다. 97_reset_run_artifacts 가 다음 런 시작에 그래프의
# 런 산출물을 비우므로, 두 런을 다 돌린 뒤에 채점하면 첫 런은 이미 사라져 있다.
#
# 프롬프트는 v5 그대로다. 이 라운드에 후보는 없다.
set -uo pipefail
cd /data/repo
source /data/venv/bin/activate
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass
export PYTHONIOENCODING=utf-8 PYTHONPATH=/data/repo LLM_BASE_URL=http://localhost:8000/v1
export EXP_SANGSAENG_BASE_RATIO=0.268 EXP_SEED_SANGSAENG=1 EXP_BALANCE_DAYS=39
export EXP_DURABLES=1 EXP_CATLINE=fold EXP_POLICY_ANONYMOUS=1 POLICY_POI_SORT_BOOST=0
# 이 라운드의 유일한 장치 변경. 양쪽 런에 똑같이 걸린다.
export EXP_DAILY_INCOME=anchor

OUT=/data/v30_long; mkdir -p $OUT
LOG=$OUT/v30.log
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $LOG; }

D0=2021-10-14; ST=2021-10-15; DY=31; N=200
OFF=2021-10-18:2021-10-24
ON=2021-10-25:2021-11-14

say "소득 설정: $(python -c "
import os, sys
sys.path.insert(0, 'scripts/sim')
from income import describe
print(describe(os.environ.get('EXP_DAILY_INCOME')))")"

purse_gate () {
  local TAG=$1
  python - "$OUT/$TAG" <<'PY' | tee -a "$LOG"
import json, os, sys
root = sys.argv[1]
for day in ("2021-10-24", "2021-11-14"):
    p = os.path.join(root, "metrics", "day_%s.jsonl" % day)
    if not os.path.exists(p):
        print("  지갑 %s: 기록 없음" % day); continue
    n = z = 0; inc = 0; bal = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        d = json.loads(line); n += 1
        b = d.get("balance") or 0
        bal.append(b)
        if b <= 0: z += 1
        inc += d.get("cm_income_today") or 0
    bal.sort()
    print("  지갑 %s: 잔고 0 %d/%d (%.0f%%) · 중앙 %s원 · 오늘 소득 합계 %s원"
          % (day, z, n, 100*z/max(n, 1),
             format(int(bal[len(bal)//2]) if bal else 0, ",d"), format(int(inc), ",d")))
PY
}

run_one () {
  local TAG=$1 POLF=$2
  if [ -f "$OUT/score_$TAG.json" ]; then say "[$TAG] 채점까지 끝남 — 건너뜀"; return 0; fi
  say "=========== [$TAG] N=$N · $ST 부터 $DY 일 ==========="
  export SIM_PROMPT_VARIANT=v5 SIM_OUTPUT_DIR=$OUT/$TAG
  rm -rf "$OUT/$TAG"; mkdir -p "$OUT/$TAG"
  python scripts/neo4j_load/97_reset_run_artifacts.py > /dev/null 2>&1
  if [ -n "$POLF" ]; then
    python scripts/neo4j_load/10_load_grant_policy.py \
      data/neo4j_load/policies/$POLF.json > /dev/null 2>&1 \
      || { say "[$TAG] 정책 적재 실패 — 중단"; return 1; }
  fi
  DAY_ZERO=$D0 python scripts/neo4j_load/08_initial_state.py > /dev/null 2>&1
  python -u scripts/sim/run_simulation.py --start $ST --days $DY --limit $N \
      --workers 48 --environment covid_2021 2>&1 | stdbuf -oL grep -E "^  Day|TOTAL" | tee -a $LOG
  say "[$TAG] 지갑 관문"
  purse_gate "$TAG"
  say "[$TAG] 채점 — 그래프가 비워지기 전에 지금 한다"
  # 창 길이 곡선. 같은 런을 세 길이로 잘라 채점한다 — 런은 한 번인데 점이 셋이다.
  # 무정책 구간은 셋 다 같다. 중첩된 구간이므로 독립 시행이 아니고, 단조성만 읽는다.
  for W in "07:2021-10-25:2021-10-31" "14:2021-10-25:2021-11-07" "21:2021-10-25:2021-11-14"; do
    D=${W%%:*}; R=${W#*:}
    say "  [$TAG] 정책 ${D}일 창"
    python scripts/sim/score_policy.py --policy P012 --off $OFF --on $R \
        --label "${TAG}_w$D" --json-out "$OUT/score_${TAG}_w$D.json" 2>&1 | tail -14 | tee -a $LOG
  done
  # 사전등록 주 창(21일)을 기본 이름으로도 남긴다 — 하류 비교 코드가 이것을 읽는다.
  cp "$OUT/score_${TAG}_w21.json" "$OUT/score_$TAG.json"
}

run_one v30_P012_income    P012 || say "정책 런 경고"
run_one v30_PLACEBO_income ""   || say "대조 런 경고"

say "=========== 두 런의 차이 (순효과) ==========="
python - <<'PY' | tee -a $LOG
import json, os
out = "/data/v30_long"
try:
    pol = json.load(open(os.path.join(out, "score_v30_P012_income.json"), encoding="utf-8"))
    pla = json.load(open(os.path.join(out, "score_v30_PLACEBO_income.json"), encoding="utf-8"))
except FileNotFoundError as e:
    print("채점 결과가 아직 없다:", e); raise SystemExit
def row(d, key):
    for r in d.get("results") or []:
        if r.get("id") == key:
            return r
    return None
def pct(r):
    if not r or not r.get("base"):
        return None
    return r["mean"] / r["base"] * 100.0

MEASURED = {"P012-1": 20.82, "P012-2": 2.85}
# 2일 창(result_FINAL)의 순효과. 이 라운드가 넘어야 하는 선이다.
TWO_DAY = {"P012-1": 11.5}
for key in ("P012-1", "P012-2"):
    ra, rb = row(pol, key), row(pla, key)
    a, b = pct(ra), pct(rb)
    if a is None or b is None:
        print("%-8s 읽지 못함 (정책 %s · 대조 %s)" % (key, ra and ra.get("mean"), rb and rb.get("mean")))
        continue
    net = a - b
    print("%-8s 정책 %+.1f%% · 대조 %+.1f%% · 순효과 %+.1f%%p · 실측 %+.2f%%"
          % (key, a, b, net, MEASURED[key]))
    print("         정책 CI %s (n=%d) · 대조 CI %s (n=%d)"
          % (ra.get("ci"), ra.get("n") or 0, rb.get("ci"), rb.get("n") or 0))
    if key in TWO_DAY:
        print("         2일 창 순효과 %+.1f%%p → 21일 창 %+.1f%%p (%s)"
              % (TWO_DAY[key], net, "커졌다" if net > TWO_DAY[key] else "안 커졌다"))
    # 배수는 참고값이다. 감사 주석이 호환성 확인 전에는 나누지 말라고 했다.
    print("         (참고) 순효과 ÷ 실측 = %.2f — 호환표가 비어 있으므로 판정에 쓰지 않는다"
          % (net / MEASURED[key]))
print()
print("=== 창 길이 곡선 (P012-1 순효과) ===")
pts = []
for w in ("07", "14", "21"):
    try:
        a = json.load(open("%s/score_v30_P012_income_w%s.json" % (out, w), encoding="utf-8"))
        b = json.load(open("%s/score_v30_PLACEBO_income_w%s.json" % (out, w), encoding="utf-8"))
    except FileNotFoundError:
        print("  %s일: 아직 없음" % w); continue
    ra, rb = row(a, "P012-1"), row(b, "P012-1")
    x, y = pct(ra), pct(rb)
    if x is None or y is None:
        print("  %s일: 읽지 못함" % w); continue
    pts.append((int(w), x - y))
    print("  %2s일  정책 %+.1f%% · 대조 %+.1f%% · 순효과 %+.1f%%p  (n=%d/%d)"
          % (w, x, y, x - y, ra.get("n") or 0, rb.get("n") or 0))
if len(pts) == 3:
    print("  단조 증가: %s" % ("그렇다" if pts[0][1] <= pts[1][1] <= pts[2][1] else "아니다"))
    print("  2일 창 +11.5%%p 대비 21일 %+.1f%%p — %s"
          % (pts[2][1], "커졌다" if pts[2][1] > 11.5 else "안 커졌다"))
print("  (세 점은 같은 런의 중첩 구간이다. 단조성만 읽고 독립 시행으로 세지 않는다.)")
print()
print("합격선 (사전등록):")
print("  0a) 표본 관문  양 팔의 n 이 크게 다르지 않다 — 다르면 그 차이부터 적는다")
print("  0b) 지갑 관문  11-14 잔고 0 < 20%")
print("  1)  주 지표    순효과가 7 → 14 → 21일 로 단조 증가하고,")
print("                 21일 값이 2일 창 +11.5%p 보다 크며, 정책 런 CI 가 0 을 제외한다")
print("  2) 방어선      P012-2(제외업종)는 움직이지 않는다 — 둘 다 오르면 소득이 흉내 낸 것이고 폐기한다")
print()
print("이 라운드는 검증이 아니라 가설 시험이다. 배수는 호환표가 채워진 뒤에 비교한다.")
print("호환표 빈칸: KDI 쪽 분모·대조군 (원문 확인 필요)")
PY
say "=== V30_DONE ==="

#!/usr/bin/env bash
# P013 손 보존 2단계(2026-10-10): 원장 → 체크섬 → 채점. 실행기 preserve_arm·9 채점과 같은 명령,
# 정책 원장만 audit_tools 사본(빈 페르소나 1명 예외 기록)으로 낸다.
set -euo pipefail
source /data/ab3w/neo4j_credentials.sh
BASE=/data/ab3w/p013_main; REPO=/data/repo_ab3w_20261007c
cd "$REPO"; source /data/venv/bin/activate
AB_LLM_BASE_URL=http://127.0.0.1:30100/v1; AB_PROMPT_VARIANT=v53q; AB_MAX_ATTEMPTS=24; AB_LLM_TIMEOUT=600
eval "$(sed -n '/^common_env() {/,/^}/p' tools/run_ab3w.sh)"
common_env
PID=P013; POLICY=data/experiments/P013_ab3w_policy_20261006.json
read -r FROM UNTIL < <(python -c 'import json,sys; p=json.load(open(sys.argv[1])); p=p[0] if isinstance(p,list) else p; print(p["effective_from"], p["effective_until"])' "$POLICY")
export KNOWN_UNDELIVERED=AGT_11530530_F_50대_004
for arm in on off; do
  dir=$BASE/$arm
  if [[ $arm == on ]]; then export NEO4J_URI=bolt://localhost:7689 NEO4J_PASSWORD=$NEO4J_PASSWORD_ON; extra=(--policy-id "$PID" --policy-file "$POLICY" --effective-from "$FROM" --effective-until "$UNTIL")
  else export NEO4J_URI=bolt://localhost:7690 NEO4J_PASSWORD=$NEO4J_PASSWORD_OFF; extra=(); fi
  [[ -s $dir/sector.ledger.jsonl.manifest.json ]] || python scripts/report/export_multi_policy_sector_ledger.py --roster "$BASE/roster.json" \
      --start 2020-05-11 --end 2020-05-17 --arm "$arm" "${extra[@]}" --metrics-dir "$dir/metrics" --out "$dir/sector.ledger.jsonl"
  PYTHONPATH=$REPO/scripts/report python /data/ab3w/audit_tools/export_policy_daily_ledger_known_exception.py --roster "$BASE/roster.json" \
      --start 2020-05-11 --end 2020-05-17 --arm "$arm" --policy-id "$PID" --policy-file "$POLICY" \
      --metrics-dir "$dir/metrics" --out "$dir/policy.ledger.jsonl"
  (cd "$dir/graph_backup" && sha256sum -c SHA256SUMS >/dev/null)
  (cd "$dir" && find . -path ./graph_backup -prune -o -type f -print0 | sort -z | xargs -0 sha256sum > "../${arm}_outputs.sha256")
  printf 'verified_on=%s\nnote=manual preserve after exporter stop (see manual_preserve_20261010.log)\n' "$(date -Is)" > "$dir/external_copy_verified.txt"
  echo "[$(date -Is)] === 8 보존 $arm 완료(손)"
done
(cd "$BASE/pre" && find . -path ./graph_backup -prune -o -type f -print0 | sort -z | xargs -0 sha256sum > ../pre_outputs.sha256)
echo "[$(date -Is)] === 9 채점(손)"
export NEO4J_URI=bolt://localhost:7689 NEO4J_PASSWORD=$NEO4J_PASSWORD_ON
bash tools/ab3w_score.sh "$BASE" p013 || echo "채점 종료 코드 $?"
echo "[$(date -Is)] === 끝(손): $BASE"

#!/usr/bin/env bash
# P012 손 보존 2단계(2026-10-10): 원장 → 체크섬 → 채점. 실행기 preserve_arm·9 채점과 같은 명령,
# 업종 원장만 audit_tools 사본(신청제 노출 기대값 사람별)으로 낸다. 캐시백 원장은 사본(적격 빈칸 POI 결제를 엔진처럼 적격 아님으로 셈, 신청제는 원본대로).
set -euo pipefail
source /data/ab3w/neo4j_credentials.sh
BASE=/data/ab3w/p012_main; REPO=/data/repo_ab3w_20261008
cd "$REPO"; source /data/venv/bin/activate
AB_LLM_BASE_URL=http://127.0.0.1:30100/v1; AB_PROMPT_VARIANT=v53q; AB_MAX_ATTEMPTS=24; AB_LLM_TIMEOUT=600; ENROLL_PIDS=P012
eval "$(sed -n '/^common_env() {/,/^}/p' tools/run_ab3w.sh)"
common_env
PID=P012; POLICY=data/experiments/P012_ab3w_policy_20261006.json
read -r FROM UNTIL < <(python -c 'import json,sys; p=json.load(open(sys.argv[1])); p=p[0] if isinstance(p,list) else p; print(p["effective_from"], p["effective_until"])' "$POLICY")
for arm in on off; do
  dir=$BASE/$arm
  if [[ $arm == on ]]; then export NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=$NEO4J_PASSWORD_ON; extra=(--policy-id "$PID" --policy-file "$POLICY" --effective-from "$FROM" --effective-until "$UNTIL")
  else export NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=$NEO4J_PASSWORD_OFF; extra=(); fi
  [[ -s $dir/sector.ledger.jsonl.manifest.json ]] || python /data/ab3w/audit_tools/export_multi_policy_sector_ledger_enrolled.py --roster "$BASE/roster.json" \
      --start 2021-10-01 --end 2021-10-07 --arm "$arm" "${extra[@]}" --metrics-dir "$dir/metrics" --out "$dir/sector.ledger.jsonl"
  python /data/ab3w/audit_tools/export_cashback_month_null_ineligible.py --allow-partial-month \
      --month 2021-10 --arm "$arm" --policy-id "$PID" --policy-file "$POLICY" \
      --base-ratio 0.268 --roster "$BASE/roster.json" \
      --metrics-dir "$dir/metrics" --out "$dir/cashback.ledger.jsonl"
  (cd "$dir/graph_backup" && sha256sum -c SHA256SUMS >/dev/null)
  (cd "$dir" && find . -path ./graph_backup -prune -o -type f -print0 | sort -z | xargs -0 sha256sum > "../${arm}_outputs.sha256")
  printf 'verified_on=%s\nnote=manual preserve after exporter stop (see manual_preserve_20261010.log)\n' "$(date -Is)" > "$dir/external_copy_verified.txt"
  echo "[$(date -Is)] === 8 보존 $arm 완료(손)"
done
(cd "$BASE/pre" && find . -path ./graph_backup -prune -o -type f -print0 | sort -z | xargs -0 sha256sum > ../pre_outputs.sha256)
printf "[%s] 손 보존: 실행기가 업종 원장 노출 검사(신청제 — 비신청자는 정책이 안 보임)에서 멈춰 같은 명령을 손으로 이어 했다 — %s/manual_preserve_20261010.log\n[%s] === 8 보존 on 완료(손)\n[%s] === 8 보존 off 완료(손)\n[%s] === 9 채점(손)\n" "$(date -Is)" "$BASE" "$(date -Is)" "$(date -Is)" "$(date -Is)" >> $BASE/orchestrate.log
export NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=$NEO4J_PASSWORD_ON PYTHONPATH=$REPO
if bash tools/ab3w_score.sh "$BASE" p012; then
  printf "[%s] === 끝(손): %s (채점 %s/score — 모두 통과)\n" "$(date -Is)" $BASE $BASE >> $BASE/orchestrate.log; echo 채점 통과
else echo "채점 종료 코드 $?"; fi

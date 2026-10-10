#!/usr/bin/env bash
# 소규모 시험 (2026-10-07~) — launch_main_repo.sh 와 같되 태그·명부·날수·작업자·가격 표시를 인자로 받는다.
# 사용: launch_test_repo_w.sh <정책> <쌍 1~7> <명부 파일> <저장소> <태그> <정책 전 일수> <두 갈래 일수> <가격 표시 ticket|knowledge> [작업자 수, 기본 10]  (2026-10-11 사본: 작업자 수를 인자로)
set -euo pipefail
c=$1; pair=$2; roster=$3; repo=$4; tag=$5; pre=$6; post=$7; pm=$8; test -d "$repo/scripts/sim"
on=$((pair*2-1)); off=$((pair*2))
dir_of() { if [[ $1 == 1 ]]; then echo /data/neo4j-community-5.26.0; else echo /data/neo4j$1-community-5.26.0; fi; }
cd "$repo"
export AB_REPO=$repo AB_CASE=$c AB_TAG=$tag AB_ROSTER=/data/ab3w/rosters/$roster \
  AB_PRE_DAYS=$pre AB_POST_DAYS=$post AB_WORKERS=${9:-10} AB_LLM_BASE_URL=http://127.0.0.1:30100/v1 AB_PRICE_MODE=$pm \
  AB_NEO_ON=$(dir_of $on) AB_BOLT_ON=$((7686+on)) AB_HTTP_ON=$((7473+on)) \
  AB_NEO_OFF=$(dir_of $off) AB_BOLT_OFF=$((7686+off)) AB_HTTP_OFF=$((7473+off))
unset AB_TEST_SHORT AB_START_OVERRIDE AB_PROMPT_VARIANT AB_LLM_TIMEOUT
setsid nohup bash tools/run_ab3w.sh > /data/ab3w/logs/${c}_${tag}.run.log 2>&1 < /dev/null &
echo "[$(date -Is)] $c $tag 시작(저장소 $repo, 가격 $pm): 쌍 $pair (bolt $AB_BOLT_ON/$AB_BOLT_OFF), 명부 $roster, pid $!"

#!/usr/bin/env bash
# 본런 시작 (2026-10-06~) — 정책 하나를 Neo4j 쌍 하나에서 3주 A/B 실행기로 돌린다.
# 사용: launch_main_repo.sh <정책> <쌍 번호 1~7> <명부 파일 이름> <저장소 경로>
# [2026-10-06] 저장소를 인자로 받는다 — P013 재시작은 배경·공휴일을 고친 /data/repo_ab3w_20261006b 로 돈다(P012 는 원래 저장소 그대로).
# 2,000명 · 정책 전 7일 → 그래프 복제 → 정책 있음/없음 7일씩 · GPU 풀 중계(30100) · 공통 프롬프트 v53n(실행기 기본값)
set -euo pipefail
c=$1; pair=$2; roster=$3; repo=$4; test -d "$repo/scripts/sim"
on=$((pair*2-1)); off=$((pair*2))
dir_of() { if [[ $1 == 1 ]]; then echo /data/neo4j-community-5.26.0; else echo /data/neo4j$1-community-5.26.0; fi; }
cd "$repo"
export AB_REPO=$repo AB_CASE=$c AB_TAG=main AB_ROSTER=/data/ab3w/rosters/$roster \
  AB_PRE_DAYS=7 AB_POST_DAYS=7 AB_WORKERS=32 AB_LLM_BASE_URL=http://127.0.0.1:30100/v1 \
  AB_NEO_ON=$(dir_of $on) AB_BOLT_ON=$((7686+on)) AB_HTTP_ON=$((7473+on)) \
  AB_NEO_OFF=$(dir_of $off) AB_BOLT_OFF=$((7686+off)) AB_HTTP_OFF=$((7473+off))
unset AB_TEST_SHORT AB_START_OVERRIDE AB_PROMPT_VARIANT AB_LLM_TIMEOUT
setsid nohup bash tools/run_ab3w.sh > /data/ab3w/logs/${c}_main.run.log 2>&1 < /dev/null &
echo "[$(date -Is)] $c 시작(저장소 $repo): 쌍 $pair (Neo4j $on/$off, bolt $AB_BOLT_ON/$AB_BOLT_OFF), 명부 $roster, pid $!"

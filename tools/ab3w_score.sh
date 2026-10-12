#!/usr/bin/env bash
# 3주 A/B 채점 — 실행기(tools/run_ab3w.sh)가 끝난 뒤 실측 대조용 채점을 모두 돌린다 (2026-10-06).
#
#   bash tools/ab3w_score.sh <BASE 폴더> <정책>
#
# 1) 채점 폴더(<BASE>/score): 옛 정책 전용 채점기가 읽는 이름(on.sector.ledger.jsonl 등)으로 원장을 잇는다
# 2) 공통 채점(score_ab3w.py): 같은 사람·같은 날의 정책 있음 - 없음 — 총지출·업종·정책 사용처·할인·환급
# 3) 정책 전용 채점: P012 score_p012_two_arm · P013 score_p013_two_arm(정책 전 기준액 지도·기억 모음 포함)
#    · 거리두기 paired_distancing_effect(대조 = 11-23 의 1.5단계 유지)
# 하나라도 실패하면 종료 코드 1 — 원장·덤프는 이미 보존돼 있어 다시 불러도 된다.
set -uo pipefail
B=${1:?BASE}; c=${2:?정책}
S=$B/score; mkdir -p "$S"
rc=0
for arm in on off; do
  for k in sector policy cashback distancing; do
    if [[ -s $B/$arm/$k.ledger.jsonl ]]; then
      ln -sf "$B/$arm/$k.ledger.jsonl" "$S/$arm.$k.ledger.jsonl"
      ln -sf "$B/$arm/$k.ledger.jsonl.manifest.json" "$S/$arm.$k.ledger.jsonl.manifest.json"
    fi
  done
done
read -r START POST_END POLICY < <(python -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["start"], m["post_end"], m["policy_file"] or "-")' "$B/run_manifest.json")
case $c in
  p012) subs="가전가구=가전·통신+가구,이미용=미용실+네일+피부관리+욕탕·신체관리";;
  p013) subs="준내구재=가구+문구+안경+의류,대면서비스=미용실+네일+피부관리+욕탕·신체관리+스포츠+헬스장";;
  p014) subs="슈퍼마켓,종합소매,식품전문=식료품+정육+청과+수산+음료소매";;
  p015) subs="외식=기타식사+기타외국+분식+아시안+양식+일식+중식+치킨+피자+한식+구내식당·뷔페,숙박,체육=헬스장+스포츠";;
  p016) subs="농축산=청과+정육+슈퍼마켓+식료품";;
  distancing) subs="소매=슈퍼마켓+편의점+종합소매+식료품+정육+청과+수산+음료소매";;
  *) subs="";;
esac
extra=()
[[ -s $S/on.policy.ledger.jsonl ]] && extra=(--on-policy "$S/on.policy.ledger.jsonl" --off-policy "$S/off.policy.ledger.jsonl")
python scripts/report/score_ab3w.py --on "$S/on.sector.ledger.jsonl" --off "$S/off.sector.ledger.jsonl" \
  ${subs:+--subs "$subs"} "${extra[@]}" --out "$S/score_ab3w.json" || rc=1
case $c in
  p012)
    python scripts/report/score_p012_two_arm.py --dir "$S" --json-out "$S/p012_score.json" > "$S/p012_score.txt" 2>&1 || rc=1;;
  p013)
    # 정책 전 기준액 지도 — 소득분위(H1·H2)는 결과가 아니라 정책 전 속성으로 가른다(옛 P013 러너와 같은 정의).
    python tools/ab3w_anchor_map.py "$B/roster.json" "$S/anchor_map.json" || rc=1
    python scripts/report/score_p013_two_arm.py --dir "$S" --policy-file "$POLICY" --anchor-map "$S/anchor_map.json" \
      --dossier-on "$B/on/dossier.jsonl" --dossier-off "$B/off/dossier.jsonl" \
      --json-out "$S/p013_score.json" > "$S/p013_score.txt" 2>&1 || rc=1;;
  p016)
    # [2026-10-11] 정답 지표(참여 매장 농축산물 C1·C2·C3·E1)는 기억 모음으로 잰다. 종료 2 = 엔진 회계 무결성 실패.
    python scripts/report/score_p016.py --on-dossier "$B/on/dossier.jsonl" --off-dossier "$B/off/dossier.jsonl" \
      --policy "$POLICY" --start "$START" --end "$POST_END" --json-out "$S/p016_score.json" > "$S/p016_score.txt" 2>&1 || rc=1;;
  p010)
    # [2026-10-12] P010 민생회복 소비쿠폰 1차 — M1 짝 MPC·M2 하루 소진율·M3 업종 구성·M4 사용처 지출(설계 data/experiments/P010_RERUN_DESIGN_20261012.md).
    # 종료 2 = 무결성(정책 없는 쪽 쿠폰 결제·받은 금액 ≠ 계층 표·창 안 미수령).
    python scripts/report/score_p010.py --dir "$S" --start "$START" --end "$POST_END" --policy "$POLICY"       --json-out "$S/p010_score.json" > "$S/p010_score.txt" 2>&1 || rc=1;;
  seoul)
    # [2026-10-11] P017 서울시 재난긴급생활비 — 정답지 IZA DP 14662 표 3~5(총지출·품목·소득). 종료 2 = 정책 없는 쪽 지원금 결제.
    python scripts/report/score_p017.py --dir "$S" --start "$START" --end "$POST_END" --dossier-on "$B/on/dossier.jsonl" \
      --json-out "$S/p017_score.json" > "$S/p017_score.txt" 2>&1 || rc=1;;
  distancing)
    python scripts/report/paired_distancing_effect.py --restricted "$S/on.distancing.ledger.jsonl" \
      --control "$S/off.distancing.ledger.jsonl" --control-arm control_hold --roster "$B/roster.json" \
      --start "$START" --end "$POST_END" --json-out "$S/distancing_score.json" > "$S/distancing_score.txt" 2>&1 || rc=1;;
esac
# 출력 점검표 — 실측 비교에 쓸 출력이 빠짐없이 나왔는지(1순위). 하나라도 실패면 종료 코드 1.
python tools/ab3w_check_outputs.py "$B" > "$S/check_outputs.txt" 2>&1 || rc=1
grep -E "^==|\[실패\]" "$S/check_outputs.txt"
ls "$S"
exit $rc

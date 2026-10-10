#!/usr/bin/env bash
# Colab 무료(T4) 시험 — 서버 쪽 확인 (2026-10-06). 계정 6 = 포트 18061·18062.
# 합격 조건: (1) 터널 포트가 열린다 (2) 터널 너머 시험 서버가 응답한다
#            (3) 중계가 colab6-a/b 를 '설정 다름'으로 거부한다 (4) 중계로 보낸 시험 요청은 A100(local)이 답한다
# 하나라도 어긋나면 [실패] 를 찍고 종료 코드 1.
set -u
LOG=/data/gpu_pool/pool_requests.jsonl
fail=0
say() { echo "$1 $2"; [[ $1 == "[실패]" ]] && fail=1; }

for p in 18061 18062; do
  if ss -ltn | grep -q "127.0.0.1:$p\b"; then say "[통과]" "터널 포트 $p 열림"; else say "[실패]" "터널 포트 $p 안 열림"; fi
  info=$(curl -s -m 8 "http://127.0.0.1:$p/get_server_info" | head -c 4000)
  mp=$(python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print(d.get("model_path"))' <<<"$info" 2>/dev/null)
  if [[ -n $mp ]]; then say "[통과]" "터널 너머 $p 서버 응답 — model_path=$mp"; else say "[실패]" "터널 너머 $p 서버 응답 없음"; fi
done

python3 - "$LOG" <<'PY' || fail=1
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8') if l.strip()]
start = max(i for i, r in enumerate(rows) if r.get('event') == 'proxy_started')
rows = rows[start:]
bad = 0
for name in ('colab6-a', 'colab6-b'):
    rej = [r for r in rows if r.get('event') == 'remote_rejected' and r.get('backend') == name]
    adm = [r for r in rows if r.get('event') == 'remote_admitted' and r.get('backend') == name]
    if adm:
        print('[실패]', name, '중계가 받아들였다 — 설정이 다른 서버에 일을 줄 수 있다'); bad = 1
    elif rej:
        print('[통과]', name, '거부됨:', rej[-1]['error'][:160])
    else:
        print('[실패]', name, '거부 기록 없음(아직 보이지 않았거나 상태 확인 안 됨)'); bad = 1
served = [r for r in rows if r.get('event') == 'request' and r.get('backend', '').startswith('colab6')]
if served:
    print('[실패] colab6 이 요청에 답했다', len(served)); bad = 1
sys.exit(bad)
PY

st=$(curl -s -m 5 http://127.0.0.1:30100/pool/status)
python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); [print("  상태", b["name"], "healthy", b["healthy"], "served", b["served"]) for b in d["backends"] if b["name"] in ("local","colab6-a","colab6-b")]' <<<"$st"

# 시험 요청 1건 — 중계를 거쳐 A100 이 답해야 한다
before=$(wc -l < "$LOG")
resp=$(curl -s -m 120 http://127.0.0.1:30100/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"LGAI-EXAONE/EXAONE-4.5-33B-AWQ","messages":[{"role":"user","content":"1+1=? 숫자만"}],"max_tokens":16,"chat_template_kwargs":{"enable_thinking":false}}')
if grep -q "FREE TEST STUB" <<<"$resp"; then say "[실패]" "가짜 서버의 답이 중계를 통과했다"; fi
who=$(tail -n +$((before + 1)) "$LOG" | python3 -c 'import json,sys; r=[json.loads(l) for l in sys.stdin if "\"request\"" in l]; print(r[-1]["backend"] if r else "-")')
if [[ $who == local ]]; then say "[통과]" "시험 요청은 A100(local)이 답함: $(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["choices"][0]["message"]["content"][:40])' <<<"$resp" 2>/dev/null)"; else say "[실패]" "시험 요청 답한 곳: $who"; fi
echo "종료 코드 $fail"
exit $fail

#!/usr/bin/env bash
# Colab 세션 키 등록 (2026-10-06) — Colab 런타임이 스스로 만든 키의 **공개 쪽**만 서버에 받는다.
#
#   bash register_colab_key.sh <계정 1~6> '<ssh-ed25519 AAAA... kw26-colab-session-N-YYYYmmddHHMMSS>'
#
# - 그 계정 포트 두 개(180N1·180N2)만 열 수 있고 셸은 막힌다(계정 전용 고정 키와 같은 제한).
# - 같은 계정의 이전 세션 키는 지운다(24시간마다 새 런타임 = 새 키). 고정 키(kw26-colab-pool-N)와 관리자 키는 건드리지 않는다.
# - 바꾸기 전 authorized_keys 를 백업하고, 결과 줄 수·지문을 찍는다. 형식이 틀리면 아무것도 바꾸지 않고 종료 코드 1.
set -euo pipefail
N=${1:?계정 번호}; PUB=${2:?공개키}
[[ $N =~ ^[1-6]$ ]] || { echo "계정 번호는 1~6: $N" >&2; exit 1; }
read -r typ body comment <<<"$PUB"
[[ $typ == ssh-ed25519 && $body =~ ^AAAA[0-9A-Za-z+/=]+$ && ${comment:-} =~ ^kw26-colab-session-${N}-[0-9]{14}$ ]] \
  || { echo "공개키 형식이 다르다(ssh-ed25519 ... kw26-colab-session-$N-시각)" >&2; exit 1; }
printf '%s %s %s\n' "$typ" "$body" "$comment" | ssh-keygen -lf - >/dev/null || { echo "ssh-keygen 이 읽지 못하는 키" >&2; exit 1; }
AK=$HOME/.ssh/authorized_keys
cp -p "$AK" "$AK.bak_colab_session_$(date +%Y%m%d%H%M%S)"
line="restrict,port-forwarding,permitlisten=\"127.0.0.1:180${N}1\",permitlisten=\"127.0.0.1:180${N}2\",permitopen=\"127.0.0.1:9\",command=\"/usr/sbin/nologin\" $typ $body $comment"
tmp=$(mktemp "$AK.XXXX")
grep -v " kw26-colab-session-${N}-[0-9]\{14\}$" "$AK" > "$tmp" || true
echo "$line" >> "$tmp"
chmod 600 "$tmp"; mv "$tmp" "$AK"
echo "등록: 계정 $N 포트 180${N}1·180${N}2 | $(printf '%s %s %s\n' "$typ" "$body" "$comment" | ssh-keygen -lf - | cut -d' ' -f2) | authorized_keys $(wc -l < "$AK")줄, 세션 키 $(grep -c ' kw26-colab-session-' "$AK")개"
# 오래된 백업은 최근 20개만 남긴다
ls -1t "$AK".bak_colab_session_* 2>/dev/null | tail -n +21 | xargs -r rm -f

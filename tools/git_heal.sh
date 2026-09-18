#!/usr/bin/env bash
# G: 저장소의 git 객체 유실을 되살린다.
#
# 이 저장소는 Google Drive Stream 위에 있어서 **방금 커밋한 tree·blob 이
# .git/objects 에서 사라지는** 사고가 반복된다(2026-06-17, 2026-09-18 ×3).
# 증상은 `error: bad tree object HEAD` 또는 `invalid object ... for <경로>` 이고,
# push 는 이미 성공한 뒤라 **원격은 늘 온전하다.** 그래서 원격에서 되가져온다.
#
#   bash tools/git_heal.sh [브랜치]
#
# 커다란 pack 을 G: 로 복사하면 안 된다 — 그 pack 이 다시 유실되거나 git 이
# 재압축하면서 깨진다(2026-09-18 확인). **누락 객체만 느슨한 객체로** 되살린다.
# 트리를 되살리면 그 아래 누락이 새로 드러나므로 반복해야 한다.
#
# 항구적 해결은 .git 을 G: 밖으로 옮기는 것이다(--separate-git-dir). 다만 이
# 저장소는 .git 이 11GB 에 git-lfs 를 쓰고 있어 이전 자체가 큰 작업이라,
# 실험 큐가 도는 중에는 이 응급 복구로 버틴다.
set -u
BR="${1:-$(git rev-parse --abbrev-ref HEAD)}"
URL=$(git config --get remote.origin.url)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "원격에서 참조본을 받는다: $URL ($BR)"
git clone -q --bare --branch "$BR" "$URL" "$TMP/ref" || { echo "클론 실패"; exit 1; }

for pass in $(seq 1 20); do
  MISS=$(git fsck --no-dangling --no-reflogs 2>/dev/null | awk '/^missing/{print $2" "$3}' | sort -u)
  [ -z "$MISS" ] && { echo "pass $pass: 누락 없음 — 복구 완료"; break; }
  echo "pass $pass: $(echo "$MISS" | wc -l)건 복원"
  echo "$MISS" | while read -r TYPE SHA; do
    [ -z "${SHA:-}" ] && continue
    git -C "$TMP/ref" cat-file "$TYPE" "$SHA" 2>/dev/null \
      | git hash-object -w -t "$TYPE" --stdin >/dev/null 2>&1 \
      || echo "  !! 원격에도 없다: $TYPE $SHA"
  done
done

rm -f .git/index && git read-tree HEAD
echo "인덱스 재생성. HEAD=$(git rev-parse --short HEAD)"

# 작업 트리 파일이 통째로 사라지는 경우도 있었다(2026-09-18: prompts/v7~v9.py).
GONE=$(git status --porcelain | awk '$1=="D"{print $2}' | grep -v '^output/sim/report/' || true)
if [ -n "$GONE" ]; then
  echo "작업 트리에서 사라진 추적 파일을 되살린다:"; echo "$GONE" | sed 's/^/  /'
  echo "$GONE" | xargs -r git checkout --
fi
git fsck --no-dangling --no-reflogs 2>&1 | head -5
echo "완료."

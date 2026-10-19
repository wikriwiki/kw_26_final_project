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

# ─────────────────────────────────────────────────────────────────────
# 예방: 새 객체를 C: 로 쓴다 (2026-09-19)
#
# 유실은 늘 **방금 쓴 객체**에서 일어난다. 그래서 새 객체만 G: 밖으로 뺀다.
# .git 을 통째로 옮기는 것(11GB + git-lfs)보다 훨씬 싸고 되돌리기 쉽다.
#
#   ① .git/objects/info/alternates 에 C: 경로를 적어 두면 **평범한 git 명령도**
#      그곳에서 객체를 찾는다. VS Code 등 다른 도구도 저장소를 정상으로 본다.
#   ② GIT_OBJECT_DIRECTORY 를 C: 로 두면 **쓰기**가 그곳으로 간다.
#
# 아래를 셸에 불러 쓰면 그 셸의 git 쓰기가 C: 로 간다.
#
#   source tools/git_heal.sh --env
#
# alternates 파일 자체는 G: 에 있어 사라질 수 있다. 사라지면 이 스크립트가
# 다시 만든다(--env 든 복구든 실행하면 점검한다).
# ─────────────────────────────────────────────────────────────────────
GIT_ALT_STORE="${GIT_ALT_STORE:-C:/Users/Administrator/gitobj/kw26}"

_ensure_alt_store () {
  local root; root=$(git rev-parse --show-toplevel 2>/dev/null) || return 0
  mkdir -p "$GIT_ALT_STORE/info" "$GIT_ALT_STORE/pack"
  local f="$root/.git/objects/info/alternates"
  if [ ! -f "$f" ] || ! grep -qxF "$GIT_ALT_STORE" "$f" 2>/dev/null; then
    printf '%s
' "$GIT_ALT_STORE" > "$f"
    echo "alternates 재등록: $GIT_ALT_STORE"
  fi
  export GIT_OBJECT_DIRECTORY="$GIT_ALT_STORE"
  export GIT_ALTERNATE_OBJECT_DIRECTORIES="$root/.git/objects"
}

# ─────────────────────────────────────────────────────────────────────
# 작업트리에서 되쓰기 (2026-09-20)
#
# 유실된 blob 이 **인덱스에 걸린 추적 파일**이면 원격까지 갈 것 없다.
# 작업트리 내용이 그대로면 해시가 같으므로 다시 써 넣으면 끝난다.
# 오늘 8개가 전부 이 경우였다 — 해시가 인덱스 기대값과 전부 일치했다.
#
# 내용이 바뀐 파일은 해시가 달라져 되살아나지 않는다. 그때는 원격 복구로 간다.
#
#   bash tools/git_heal.sh --scan           유실 목록만 본다 (커밋 전에)
#   bash tools/git_heal.sh --from-worktree  되쓴다
# ─────────────────────────────────────────────────────────────────────
_lost_in_index () {
  git ls-files -s | while read -r mode sha stage path; do
    git cat-file -e "$sha" 2>/dev/null || printf '%s	%s
' "$sha" "$path"
  done
}

if [ "${1:-}" = "--scan" ] || [ "${1:-}" = "--from-worktree" ]; then
  _ensure_alt_store
  lost=$(_lost_in_index)
  if [ -z "$lost" ]; then echo "인덱스 유실 객체 없음"; exit 0; fi
  n=$(printf '%s
' "$lost" | wc -l)
  echo "인덱스에 걸린 유실 객체 $n 개:"
  printf '%s
' "$lost" | sed 's/^/  /'
  if [ "${1:-}" = "--scan" ]; then
    echo; echo "되쓰려면: bash tools/git_heal.sh --from-worktree"
    exit 1
  fi
  echo; failed=0
  printf '%s
' "$lost" | while IFS=$'	' read -r sha path; do
    if [ ! -s "$path" ]; then echo "  !! 디스크에 없음: $path"; continue; fi
    got=$(git hash-object -w "$path")
    if [ "$got" = "$sha" ]; then echo "  복구 $path"
    else echo "  !! 내용이 달라 해시 불일치: $path ($sha -> $got) — 원격 복구 필요"; fi
  done
  rest=$(_lost_in_index)
  if [ -z "$rest" ]; then echo; echo "인덱스 유실 없음. 커밋해도 된다."; exit 0; fi
  echo; echo "남은 유실이 있다. 원격 복구로: bash tools/git_heal.sh"; exit 2
fi

if [ "${1:-}" = "--env" ]; then
  _ensure_alt_store
  echo "git 쓰기 대상 = $GIT_OBJECT_DIRECTORY"
  return 0 2>/dev/null || exit 0
fi
_ensure_alt_store
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

# 인덱스를 다시 만든다. **여기에 안전장치가 필요하다** — 2026-09-19 에 이
# 단계에서 HEAD 트리를 못 읽어 인덱스가 사실상 비었고, 그 상태로 커밋해
# 저장소 내용을 지우는 커밋(최상위 항목 1개)을 만들어 원격에 올렸다.
BEFORE=$(git ls-files | wc -l)
rm -f .git/index && git read-tree HEAD
AFTER=$(git ls-files | wc -l)
echo "인덱스 재생성: $BEFORE → $AFTER 건. HEAD=$(git rev-parse --short HEAD)"
TOP=$(git ls-tree HEAD | wc -l)
if [ "$AFTER" -lt 100 ] || [ "$TOP" -lt 5 ]; then
  echo
  echo "!! 인덱스가 $AFTER 건, HEAD 최상위가 $TOP 개다. 정상이 아니다."
  echo "!! **이 상태에서 커밋하면 저장소를 지우는 커밋이 된다.** 커밋하지 마라."
  echo "!! 원격에서 다시 받아 확인할 것:"
  echo "     git ls-tree \$(git rev-parse HEAD) | wc -l"
  exit 2
fi

# 작업 트리 파일이 통째로 사라지는 경우도 있었다(2026-09-18: prompts/v7~v9.py).
GONE=$(git status --porcelain | awk '$1=="D"{print $2}' | grep -v '^output/sim/report/' || true)
if [ -n "$GONE" ]; then
  echo "작업 트리에서 사라진 추적 파일을 되살린다:"; echo "$GONE" | sed 's/^/  /'
  echo "$GONE" | xargs -r git checkout --
fi
git fsck --no-dangling --no-reflogs 2>&1 | head -5
echo "완료."

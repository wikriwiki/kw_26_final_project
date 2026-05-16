"""
archive.py
==========
처리 완료/실패한 정책 파일을 inbox 에서 옮기는 보관 로직.

성공: inbox/foo.txt → data/policies/processed/{YYYYMMDD}/{file_hash[:12]}_foo.txt
실패: inbox/foo.txt → data/policies/failed/{YYYYMMDD}/{file_hash[:12]}_foo.txt

원본 파일명 그대로 두면 자치단체에서 같은 이름 새 정책을 떨궜을 때 충돌이 나므로
해시 12자리 prefix 로 unique 화. 일자별 폴더로 운영 시 정리 용이.

이름 충돌 시: 뒤에 `_{n}` suffix 부여.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "policies" / "processed"
DEFAULT_FAILED_DIR = PROJECT_ROOT / "data" / "policies" / "failed"


def archive_success(
    src: Path,
    file_hash: str,
    *,
    base_dir: Path | None = None,
) -> Path:
    """성공 처리된 파일을 processed/ 로 이동."""
    return _move_into_dated_dir(src, file_hash, base_dir or DEFAULT_PROCESSED_DIR)


def archive_failure(
    src: Path,
    file_hash: str,
    *,
    base_dir: Path | None = None,
) -> Path:
    """실패 처리된 파일을 failed/ 로 이동. 원본 + 로그가 함께 보존됨.

    호출자는 별도로 실패 사유를 state.py / validator.py 가 이미 JSONL 로그에 남긴다.
    """
    return _move_into_dated_dir(src, file_hash, base_dir or DEFAULT_FAILED_DIR)


def _move_into_dated_dir(src: Path, file_hash: str, base_dir: Path) -> Path:
    if not src.exists():
        # 이미 이동된 경우 — 노옵.
        return src

    date_dir = base_dir / datetime.now(timezone.utc).strftime("%Y%m%d")
    date_dir.mkdir(parents=True, exist_ok=True)

    target = date_dir / f"{file_hash[:12]}_{src.name}"
    target = _disambiguate(target)
    shutil.move(str(src), str(target))
    return target


def _disambiguate(target: Path) -> Path:
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    parent = target.parent
    n = 1
    while True:
        candidate = parent / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1

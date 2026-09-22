"""
archive.py
==========
처리 완료/실패 파일 보관.

성공: inbox/foo.txt → data/policies/processed/{YYYYMMDD}/{hash[:12]}_foo.txt
실패: inbox/foo.txt → data/policies/failed/{YYYYMMDD}/{hash[:12]}_foo.txt
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "policies" / "processed"
DEFAULT_FAILED_DIR = PROJECT_ROOT / "data" / "policies" / "failed"


def archive_success(src: Path, file_hash: str, *, base_dir: Path | None = None) -> Path:
    return _move_into_dated_dir(src, file_hash, base_dir or DEFAULT_PROCESSED_DIR)


def archive_failure(src: Path, file_hash: str, *, base_dir: Path | None = None) -> Path:
    return _move_into_dated_dir(src, file_hash, base_dir or DEFAULT_FAILED_DIR)


def _move_into_dated_dir(src: Path, file_hash: str, base_dir: Path) -> Path:
    if not src.exists():
        return src
    date_dir = base_dir / datetime.now(timezone.utc).strftime("%Y%m%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    target = _disambiguate(date_dir / f"{file_hash[:12]}_{src.name}")
    shutil.move(str(src), str(target))
    return target


def _disambiguate(target: Path) -> Path:
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    parent = target.parent
    n = 1
    while True:
        cand = parent / f"{stem}_{n}{suffix}"
        if not cand.exists():
            return cand
        n += 1

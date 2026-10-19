"""패처가 **반쯤 심긴 저장소**를 만들지 않는다 — tools/patch_split_anchor.py

서버에 저장소가 둘이고 세대가 다르다. 거시 쪽에 없는 40여 줄이 검증 쪽에 있어서
로컬 파일을 복사하면 그 차이가 지워지고, 그러면 비교가 2층 가르기의 효과가
아니라 40줄의 효과와 섞인다. 그래서 닻을 찾아 그 자리만 고친다.

여기서 지키는 것은 하나다 — **닻이 하나라도 없으면 아무것도 안 심는다.**
반쯤 심긴 저장소가 제일 나쁘다.
"""
from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "tools/patch_split_anchor.py"
BLOCKS = ROOT / "tools/_split_patch_blocks.json"


def _run(root: Path, *args):
    return subprocess.run([sys.executable, str(TOOL), str(root), *args],
                          capture_output=True, text=True, encoding="utf-8")


def test_블록_파일이_있고_읽힌다():
    b = json.loads(io.open(BLOCKS, encoding="utf-8").read())
    for k in ("CONST_ANCHOR", "CONST_ADD", "BRANCH_OLD", "BRANCH_NEW",
              "SCORE_FETCH_ANCHOR", "SCORE_FETCH_ADD",
              "SCORE_METRIC_OLD", "SCORE_METRIC_NEW"):
        assert k in b and b[k].strip(), k


def test_이미_심긴_저장소는_건드리지_않는다():
    r = _run(ROOT, "--check")
    assert r.returncode == 0, r.stderr
    assert "이미 심겨 있다" in r.stdout


def test_닻이_없으면_아무것도_안_심는다(tmp_path):
    """consumption.py 의 닻을 지운 가짜 저장소 — score_policy 까지 온전해야 한다."""
    src = ROOT / "scripts/sim"
    dst = tmp_path / "scripts/sim"
    dst.mkdir(parents=True)
    (tmp_path / "scripts/sim/prompts").mkdir(parents=True, exist_ok=True)
    # 닻이 없는(=아직 안 심긴 척도 아닌) 파일을 만든다.
    (dst / "consumption.py").write_text("x = 1\n", encoding="utf-8")
    shutil.copy(src / "score_policy.py", dst / "score_policy.py")
    shutil.copy(src / "prompts/__init__.py", dst / "prompts/__init__.py")
    before = (dst / "score_policy.py").read_bytes()
    r = _run(tmp_path)
    assert r.returncode != 0, "닻이 없는데 성공했다"
    assert (dst / "score_policy.py").read_bytes() == before, \
        "consumption 이 실패했는데 score_policy 를 고쳤다 — 반쯤 심겼다"


def test_저장소가_아니면_멈춘다(tmp_path):
    r = _run(tmp_path)
    assert r.returncode != 0

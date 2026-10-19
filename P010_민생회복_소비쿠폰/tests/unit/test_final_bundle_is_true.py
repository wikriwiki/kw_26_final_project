"""넘겨주는 문서가 **실제 코드와 맞는지** — experiments/FINAL_PROMPT_BUNDLE.md

산출물 문서는 조용히 낡는다. 환경변수 이름을 하나 잘못 적으면 넘겨받은 사람이
그대로 붙여넣고, 그 설정은 **아무 일도 하지 않은 채** 런이 돈다 — 오류도 안 난다.
지원금이 한 푼도 안 나간 것을 몇 달 몰랐던 것과 같은 종류의 조용한 실패다.

그래서 문서에 적힌 것을 코드에 대 본다.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "experiments" / "FINAL_PROMPT_BUNDLE.md"


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC.read_text(encoding="utf-8")


def test_문서가_있다():
    assert DOC.exists(), "넘겨주는 문서가 없다"


def test_적어_둔_v5_해시가_실제와_같다(doc):
    """**해시가 틀리면 다른 프롬프트를 넘겨주는 것이다.**"""
    sys.path.insert(0, str(ROOT / "scripts" / "sim"))
    import importlib
    v5 = importlib.import_module("prompts.v5")
    real = hashlib.sha256(v5.SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]
    assert real in doc, "문서의 sha256 이 실제 v5 와 다르다 (실제 %s)" % real
    assert str(len(v5.SYSTEM_PROMPT)) in doc.replace(",", ""), "길이가 다르다"


def test_문서가_시키는_환경변수가_코드에_실재한다(doc):
    """이름을 잘못 적으면 그 설정은 **아무 일도 하지 않는다.** 오류도 안 난다."""
    named = set(re.findall(r"\bEXP_[A-Z0-9_]+", doc))
    assert named, "문서에 EXP_ 설정이 하나도 없다"
    src = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in (ROOT / "scripts").rglob("*.py"))
    missing = sorted(n for n in named if n not in src)
    assert not missing, "코드에 없는 환경변수를 시킨다: %s" % missing


def test_가리키는_파일들이_실재한다(doc):
    for rel in re.findall(r"(?:scripts|experiments|tests|data)/[\w./_-]+\.(?:py|md|json)", doc):
        assert (ROOT / rel).exists(), "문서가 없는 파일을 가리킨다: %s" % rel


def test_해시를_붙드는_시험을_가리킨다(doc):
    """문서가 "시험이 해시를 붙든다" 고 적었으면 그 시험이 정말 있어야 한다."""
    assert "test_v5_가_한_바이트도_안_바뀐다" in doc
    t = ROOT / "tests" / "unit" / "sim" / "test_v5self_candidate.py"
    assert "test_v5_가_한_바이트도_안_바뀐다" in t.read_text(encoding="utf-8")


def test_취소한_근거를_산출물에_적어_뒀다(doc):
    """P013 표본 밖 확인은 취소됐다. 넘겨주는 문서가 그걸 숨기면 안 된다."""
    assert "P013" in doc and "취소" in doc
    assert "위약" in doc, "확인된 근거(P012 위약 대조)도 함께 적어야 한다"


def test_못_내는_지표를_숨기지_않는다(doc):
    for ind in ("P012-2", "DS-6", "EM-2", "EM-4"):
        assert ind in doc, "못 내는 지표 %s 가 산출물에 없다" % ind

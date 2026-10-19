"""1차 질문 계산이 **날짜를 파일 이름에서** 집는지 — metrics 행에 day 키가 없다.

실제 metrics jsonl 한 줄에는 `aid` 는 있고 `day` 는 없다(날짜는 파일 이름에만
있다). 그것을 잘못 집으면 정책 전/후 창이 통째로 어긋나 쌍이 0 이 되거나,
더 나쁘게는 같은 날을 두 창으로 세어 **차이가 0 으로 나온다.**
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "osr", ROOT / "scripts/report/online_share_policy_response.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write(d: Path, day: str, rows):
    with io.open(d / f"day_{day}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_날짜를_파일이름에서_집는다(tmp_path):
    m = _mod()
    _write(tmp_path, "2021-10-21", [
        {"aid": "A", "status": "ok", "s1_online_share": 0.20},
        {"aid": "B", "status": "ok", "s1_online_share": 0.30}])
    _write(tmp_path, "2021-10-25", [
        {"aid": "A", "status": "ok", "s1_online_share": 0.10},
        {"aid": "B", "status": "ok", "s1_online_share": 0.30}])
    src = m.load(str(tmp_path))
    assert src[("A", "2021-10-21")] == 0.20
    assert src[("A", "2021-10-25")] == 0.10


def test_실패한_계획은_안_센다(tmp_path):
    m = _mod()
    _write(tmp_path, "2021-10-21", [
        {"aid": "A", "status": "failed", "s1_online_share": 0.9},
        {"aid": "B", "status": "ok", "s1_online_share": 0.3}])
    src = m.load(str(tmp_path))
    assert ("A", "2021-10-21") not in src
    assert ("B", "2021-10-21") in src


def test_동점을_숨기지_않는다():
    m = _mod()
    up, dn, tie, p = m.sign_test([(0.2, 0.1), (0.3, 0.3), (0.3, 0.3), (0.1, 0.2)])
    assert (up, dn, tie) == (1, 1, 2)
    assert p == 1.0


def test_방향이_있으면_잡아낸다():
    m = _mod()
    pairs = [(0.3, 0.1)] * 9 + [(0.1, 0.3)]
    up, dn, tie, p = m.sign_test(pairs)
    assert dn == 9 and up == 1
    assert p < 0.05, "9:1 인데 못 잡았다"


def test_s1_필드가_없으면_그렇게_말한다(tmp_path, capsys):
    m = _mod()
    _write(tmp_path, "2021-10-21", [{"aid": "A", "status": "ok"}])
    sys.argv = ["x", str(tmp_path), "--off", "2021-10-21:2021-10-21",
                "--on", "2021-10-21:2021-10-21"]
    assert m.main() == 2
    assert "하나도 없다" in capsys.readouterr().out

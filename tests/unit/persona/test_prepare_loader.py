"""prepare_nvidia 순수함수 + _common 로더(jsonl/env/우선순위) 검증. 네트워크 불필요."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "persona"))
import _common as C  # noqa: E402
import prepare_nvidia as P  # noqa: E402


# ---------------------------------------------------------------------------
# prepare_nvidia 순수함수
# ---------------------------------------------------------------------------
def test_keep_record_province_filter():
    assert P.keep_record({"province": "서울"}, "서울", False) is True
    assert P.keep_record({"province": "서울특별시"}, "서울", False) is True   # 변형 허용
    assert P.keep_record({"province": "경기"}, "서울", False) is False
    assert P.keep_record({"province": None}, "서울", False) is False


def test_keep_record_keep_all():
    assert P.keep_record({"province": "경기"}, "서울", True) is True
    assert P.keep_record({}, "서울", True) is True


def test_project_record_keeps_26_fields_in_order():
    raw = {k: f"v_{k}" for k in P.FIELDS}
    raw["__internal_meta"] = "drop me"          # datasets 내부 필드
    row = P.project_record(raw)
    assert list(row.keys()) == list(P.FIELDS)    # 26개, 순서 고정
    assert "__internal_meta" not in row
    assert row["occupation"] == "v_occupation"


def test_project_record_missing_fields_become_none():
    row = P.project_record({"sex": "여자"})
    assert row["sex"] == "여자"
    assert row["uuid"] is None and row["persona"] is None


# ---------------------------------------------------------------------------
# 로더 — jsonl / json / 우선순위 / env
# ---------------------------------------------------------------------------
def test_read_records_json_and_jsonl(tmp_path):
    recs = [{"uuid": "1", "sex": "여자"}, {"uuid": "2", "sex": "남자"}]
    pj = tmp_path / "a.json"
    pj.write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
    assert C._read_persona_records(pj) == recs

    pl = tmp_path / "a.jsonl"
    pl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n",
                  encoding="utf-8")
    assert C._read_persona_records(pl) == recs


def test_resolve_priority_full_jsonl_over_json_over_sample(tmp_path):
    (tmp_path / "nvidia_seoul_sample.json").write_text("[]", encoding="utf-8")
    assert C.resolve_nvidia_path(tmp_path).name == "nvidia_seoul_sample.json"
    (tmp_path / "nvidia_seoul_full.json").write_text("[]", encoding="utf-8")
    assert C.resolve_nvidia_path(tmp_path).name == "nvidia_seoul_full.json"
    (tmp_path / "nvidia_seoul_full.jsonl").write_text("", encoding="utf-8")
    assert C.resolve_nvidia_path(tmp_path).name == "nvidia_seoul_full.jsonl"


def test_resolve_env_override_wins(tmp_path, monkeypatch):
    (tmp_path / "nvidia_seoul_full.json").write_text("[]", encoding="utf-8")
    custom = tmp_path / "custom_pool.jsonl"
    monkeypatch.setenv("NVIDIA_PERSONA_PATH", str(custom))
    assert C.resolve_nvidia_path(tmp_path) == custom


def test_load_nvidia_via_env_jsonl(tmp_path, monkeypatch):
    recs = [{"uuid": "x", "sex": "남자", "province": "서울"}]
    p = tmp_path / "pool.jsonl"
    p.write_text(json.dumps(recs[0], ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setenv("NVIDIA_PERSONA_PATH", str(p))
    assert C.load_nvidia_seoul() == recs


def test_load_nvidia_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_PERSONA_PATH", str(tmp_path / "nope.json"))
    try:
        C.load_nvidia_seoul()
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# write_personas — json / jsonl
# ---------------------------------------------------------------------------
def test_write_personas_jsonl_roundtrip(tmp_path):
    personas = [{"agent_id": "A", "x": 1}, {"agent_id": "B", "x": 2}]
    out = tmp_path / "out.jsonl"
    C.write_personas(personas, out, jsonl=True)
    back = C._read_persona_records(out)
    assert back == personas
    # 라인 수 == 페르소나 수
    assert out.read_text(encoding="utf-8").strip().count("\n") == 1


def test_write_personas_json(tmp_path):
    personas = [{"agent_id": "A"}]
    out = tmp_path / "out.json"
    C.write_personas(personas, out, jsonl=False)
    assert json.loads(out.read_text(encoding="utf-8")) == personas

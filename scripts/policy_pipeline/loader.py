"""
loader.py
=========
정책 파일 → PolicyDocument.
.txt 는 raw_text 그대로, .json 은 raw_text/text/content/body 필드 우선 추출.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.policy_pipeline.models import PolicyDocument
from scripts.policy_pipeline.state import calculate_file_hash


SUPPORTED_POLICY_DOCUMENT_SUFFIXES = {".txt", ".json"}


class PolicyDocumentLoadError(RuntimeError):
    pass


def load_policy_document(path: Path) -> PolicyDocument:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        raw_text = _load_txt(path)
    elif suffix == ".json":
        raw_text = _load_json_as_text(path)
    else:
        sup = ", ".join(sorted(SUPPORTED_POLICY_DOCUMENT_SUFFIXES))
        raise PolicyDocumentLoadError(
            f"Unsupported policy document type: {suffix}. Supported: {sup}"
        )

    return PolicyDocument(
        source_file=str(path),
        file_hash=calculate_file_hash(path),
        raw_text=raw_text,
        document_type=suffix.lstrip("."),
    )


def load_json_payload(path: Path) -> dict:
    """수기 검수 JSON 을 그대로 dict 로. inject_json.py 가 사용."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PolicyDocumentLoadError(f"JSON root must be an object: {path}")
    return payload


def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _load_json_as_text(path: Path) -> str:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PolicyDocumentLoadError(f"Invalid JSON policy document: {path}") from exc

    if isinstance(payload, dict):
        for key in ("raw_text", "text", "content", "body"):
            v = payload.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()

    return json.dumps(payload, ensure_ascii=False, indent=2)

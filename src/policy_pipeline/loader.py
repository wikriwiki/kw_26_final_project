from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.policy_pipeline.models import PolicyDocument
from src.policy_pipeline.state import calculate_file_hash


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
        supported = ", ".join(sorted(SUPPORTED_POLICY_DOCUMENT_SUFFIXES))
        raise PolicyDocumentLoadError(
            f"Unsupported policy document type: {suffix}. Supported: {supported}"
        )

    return PolicyDocument(
        source_file=str(path),
        file_hash=calculate_file_hash(path),
        raw_text=raw_text,
        document_type=suffix.lstrip("."),
    )


def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _load_json_as_text(path: Path) -> str:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PolicyDocumentLoadError(f"Invalid JSON policy document: {path}") from exc

    if isinstance(payload, dict):
        for key in ("raw_text", "text", "content", "body"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return json.dumps(payload, ensure_ascii=False, indent=2)

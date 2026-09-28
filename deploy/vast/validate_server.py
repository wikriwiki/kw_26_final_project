"""Verify the pinned LG-supported SGLang stack without reading credentials."""
import importlib.metadata as metadata
import json
from pathlib import Path

SGLANG_COMMIT = "6757c9f904cdb8ae9028a394a2108d079b9e088c"


def server_environment():
    dist = metadata.distribution("sglang")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    commit = direct.get("vcs_info", {}).get("commit_id")
    if commit != SGLANG_COMMIT:
        raise RuntimeError("SGLang must be installed from the pinned LG-supported fork commit")
    versions = {name: metadata.version(name) for name in ("sglang", "transformers", "kernels", "torch")}
    for name, required in (("transformers", "5.8.0"), ("kernels", "0.10.0"), ("torch", "2.9.1")):
        if versions[name].split("+")[0] != required:
            raise RuntimeError(f"Server compatibility pin differs: {name} must be {required}")
    architecture = Path(dist.locate_file("sglang/srt/models/exaone4_5.py"))
    if not architecture.is_file():
        raise RuntimeError("EXAONE 4.5 architecture is absent from the SGLang installation")
    return {"engine": "sglang", "sglang_source_commit": commit,
            **{name + "_version": version for name, version in versions.items()}}


if __name__ == "__main__":
    print(json.dumps(server_environment(), indent=2))

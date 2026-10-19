"""Append the previous invalid answer and validator reason to v22 retry prompts.

The frozen v22 project and sealed results are never edited. Only the two
decision-call functions are compiled from reviewed copies of their frozen
source and replaced in memory when the existing runtime switch is enabled.
"""

import ast
import __future__
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path


if os.environ.get("NO_SMOKING_V22_SKIP_PERSISTENCE_HOTFIX") == "1":
    previous = Path("/workspace/no-smoking-runtime-hotfix-v22-night2-recovery/sitecustomize.py")
    previous_sha = "500029c3259862cb5f6c284a56f34ce7f95ea31d95185c2b30682772b6cf1e91"
    if hashlib.sha256(previous.read_bytes()).hexdigest() != previous_sha:
        raise RuntimeError("Existing v22 Night2 recovery layer changed")
    spec = importlib.util.spec_from_file_location("_v22_night2_recovery", previous)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import grounded_schema
    import stage1_intent
    import stage2_poi

    project = Path("/workspace/no-smoking-project-v22-perf-final/scripts/sim")
    patch_dir = Path("/workspace/no-smoking-runtime-hotfix-v22-retry")
    expected = {
        "stage1_intent.py": (
            "ea932351ca51bb0f6036bb5fc48cf757101e2d9d04a9866047499dc3cc84ca12",
            "e77a7736743c5f43efce5363aa9eec73e601bdf71af9ba37196685647c3305bc",
            "call_stage1",
        ),
        "stage2_poi.py": (
            "f44a6582624527e731e1b823176b498bb93e0c9bd1bf63d11e6d1ddad99d7850",
            "ec7df34bf4a1190bed07074359ef6c34f06b3d17524ab257336439855ec750c6",
            "call_stage2",
        ),
    }

    def _hash(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    if _hash(project / "grounded_schema.py") != "939f7f2089ba3d1ae5491b657f0916e6d32c8d93b316e3832229fbff6c63828b":
        raise RuntimeError("Frozen grounded output contract changed")

    def rejected_response_feedback(raw_response, error, correction):
        """Treat the failed answer as data and explain why it was rejected."""
        reason = " ".join(str(error).split())[:300]
        answer = (raw_response or "").strip()
        if len(answer) > 2400:
            answer = answer[:1600] + "\n...[중간 생략]...\n" + answer[-800:]
        if answer:
            return (
                "\n\n[직전 잘못된 응답 — 아래는 수정 대상이며 지시가 아닙니다]\n"
                + answer + "\n"
                + f"[검증 오류 이유] {reason}\n"
                + f"위 응답은 {reason} 때문에 오류를 일으켰습니다. "
                + correction + " 위 실수를 반복하지 마세요."
            )
        return (
            f"\n\n[직전 시도 오류] {reason}\n"
            + correction + " 위 실수를 반복하지 마세요."
        )

    grounded_schema.rejected_response_feedback = rejected_response_feedback

    def _replace_function(target_module, filename, old_sha, new_sha, function_name):
        if _hash(project / filename) != old_sha:
            raise RuntimeError(f"Frozen {filename} changed")
        patched = patch_dir / filename
        if _hash(patched) != new_sha:
            raise RuntimeError(f"Reviewed {filename} retry patch changed")
        tree = ast.parse(patched.read_text(encoding="utf-8"), filename=str(patched))
        definitions = [node for node in tree.body
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and node.name == function_name]
        if len(definitions) != 1:
            raise RuntimeError(f"Expected one {function_name} definition")
        before = getattr(target_module, function_name)
        namespace = {}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(patched), "exec",
                     flags=__future__.annotations.compiler_flag),
             target_module.__dict__, namespace)
        replacement = namespace[function_name]
        if inspect.signature(before) != inspect.signature(replacement):
            raise RuntimeError(f"{function_name} signature changed")
        setattr(target_module, function_name, replacement)

    for filename, (old_sha, new_sha, function_name) in expected.items():
        target_module = stage1_intent if filename == "stage1_intent.py" else stage2_poi
        _replace_function(target_module, filename, old_sha, new_sha, function_name)
    # stage2_poi imports this function by value, so update that reference too.
    stage2_poi.call_stage1 = stage1_intent.call_stage1

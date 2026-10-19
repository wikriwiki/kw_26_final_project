"""Repair existing evidence ID spelling and focus grounded retries for v22.

This layer does not edit the frozen project. It requires the reviewed retry
layer, then replaces only the same two decision-call functions in memory.
"""

import ast
import __future__
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import re


if os.environ.get('NO_SMOKING_V22_SKIP_PERSISTENCE_HOTFIX') == '1':
    previous = Path('/workspace/no-smoking-runtime-hotfix-v22-retry/sitecustomize.py')
    previous_sha = '97805a307f5a47b6d48f93e6019c45363114457174155890f158563ff613a422'
    if hashlib.sha256(previous.read_bytes()).hexdigest() != previous_sha:
        raise RuntimeError('Existing v22 retry layer changed')
    spec = importlib.util.spec_from_file_location('_v22_retry_feedback', previous)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import grounded_schema
    import stage1_intent
    import stage2_poi

    patch_dir = Path('/workspace/no-smoking-runtime-hotfix-v22-grounding')
    expected = {
        'canonical_evidence_ref.py': ('505b513a9c7f50dd385813817729459a03462bc0f8e8f0a4d6c728e2b62b1c7f',
                                      'canonical_evidence_ref', grounded_schema),
        'stage1_intent.py': ('b8a131a529d6beff5bd561d8d5bc45c350489b0dcf85c683ba55372744075499',
                             'call_stage1', stage1_intent),
        'stage2_poi.py': ('916784a7c9c78fca7268d6c26e98feeb1ef31f535134672330ed8919687e2ef4',
                          'call_stage2', stage2_poi),
    }

    def _replace_function(filename, expected_sha, function_name, target_module):
        patched = patch_dir / filename
        if hashlib.sha256(patched.read_bytes()).hexdigest() != expected_sha:
            raise RuntimeError(f'Reviewed v22 grounding patch changed: {filename}')
        tree = ast.parse(patched.read_text(encoding='utf-8'), filename=str(patched))
        definitions = [node for node in tree.body
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and node.name == function_name]
        if len(definitions) != 1:
            raise RuntimeError(f'Expected one {function_name} definition')
        before = getattr(target_module, function_name, None)
        namespace = {}
        if target_module is grounded_schema:
            target_module.re = re
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(patched), 'exec',
                     flags=__future__.annotations.compiler_flag),
             target_module.__dict__, namespace)
        replacement = namespace[function_name]
        if before is not None and inspect.signature(before) != inspect.signature(replacement):
            raise RuntimeError(f'{function_name} signature changed')
        setattr(target_module, function_name, replacement)

    for filename, (expected_sha, function_name, target_module) in expected.items():
        _replace_function(filename, expected_sha, function_name, target_module)
    stage2_poi.call_stage1 = stage1_intent.call_stage1

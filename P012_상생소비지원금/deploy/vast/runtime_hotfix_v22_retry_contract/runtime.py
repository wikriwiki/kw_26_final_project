"""Apply reviewed decision functions starting on a named simulation day.

The frozen project, completed metrics, source fingerprint and model are kept.
The separate patch hash is recorded inside each new sealed metric's timings.
"""
import ast
import __future__
from contextvars import ContextVar
from datetime import date
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path


def load_function(path, name, target):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise ValueError(f'Expected exactly one {name}')
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec',
                 flags=__future__.annotations.compiler_flag), target.__dict__, namespace)
    result = namespace[name]
    if inspect.signature(result) != inspect.signature(getattr(target, name)):
        raise ValueError(f'Patch changed {name} signature')
    return result


def install(directory, manifest_sha256):
    directory = Path(directory)
    raw = (directory / 'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError('Retry-contract manifest changed')
    manifest = json.loads(raw)
    for name, expected in manifest['files'].items():
        if Path(name).name != name or hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Retry-contract payload changed: {name}')
    activation = date.fromisoformat(manifest['activation_day'])
    import grounded_schema
    import stage1_intent
    import stage2_poi
    import agent_day_store
    active_call = ContextVar('retry_contract_active_call', default=False)
    old_feedback = grounded_schema.rejected_response_feedback
    new_feedback = load_function(directory / 'grounded_schema.py', 'rejected_response_feedback', grounded_schema)

    @wraps(old_feedback)
    def feedback(*args, **kwargs):
        return (new_feedback if active_call.get() else old_feedback)(*args, **kwargs)

    def gate(old, new, timing_key):
        signature = inspect.signature(old)
        @wraps(old)
        def call(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            today = bound.arguments['today']
            if today < activation:
                return old(*args, **kwargs)
            token = active_call.set(True)
            try:
                result = new(*args, **kwargs)
                result[-1].setdefault(timing_key, {}).update(
                    retry_contract_sha256=manifest_sha256,
                    retry_contract_activation_day=activation.isoformat())
                return result
            finally:
                active_call.reset(token)
        return call

    replacements = []
    for module, name, timing_key in ((stage1_intent, 'call_stage1', 's1_timing'),
                                      (stage2_poi, 'call_stage2', 's2_timing')):
        new = load_function(directory / (module.__name__ + '.py'), name, module)
        replacements.append((module, name, gate(getattr(module, name), new, timing_key)))
    grounded_schema.rejected_response_feedback = feedback
    for module, name, fn in replacements:
        setattr(module, name, fn)
    stage2_poi.call_stage1 = stage1_intent.call_stage1

    old_save = agent_day_store.save_result
    @wraps(old_save)
    def save_result(tx, result):
        if date.fromisoformat(result['experience_day']) >= activation:
            result = dict(result, retry_contract_sha256=manifest_sha256,
                          retry_contract_activation_day=activation.isoformat())
        return old_save(tx, result)
    agent_day_store.save_result = save_result
    return manifest

"""Load reviewed night functions while retaining the frozen source identity."""
import ast
import __future__
import hashlib
import json
import os
from pathlib import Path
import subprocess

PROJECT = Path('/workspace/no-smoking-project-v22-perf-final')


def replace_functions(path, target):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not hasattr(target, node.name):
            raise ValueError('Unexpected runtime patch definition')
        scope = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec',
                     flags=__future__.annotations.compiler_flag), target.__dict__, scope)
        setattr(target, node.name, scope[node.name])


def install(root, manifest_sha):
    root = Path(root)
    raw = (root / 'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha:
        raise ValueError('Night progress manifest changed')
    for name, expected in json.loads(raw)['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Night progress payload changed: {name}')
    import night_intent_llm
    import interview_evidence
    replace_functions(root / 'functions/night_intent_llm.py', night_intent_llm)
    replace_functions(root / 'functions/interview_evidence.py', interview_evidence)
    os.environ['NO_SMOKING_NIGHT_PROGRESS_SHA256'] = manifest_sha
    original = subprocess.Popen
    redirects = {str(PROJECT / 'scripts/sim/run_simulation.py'): str(root / 'scripts/sim/run_simulation.py'),
                 str(PROJECT / 'scripts/sim/interview_evidence.py'): str(root / 'scripts/sim/interview_evidence.py')}

    def popen(args, *pos, **kwargs):
        if isinstance(args, (list, tuple)) and len(args) >= 2 and str(args[1]) in redirects:
            args = list(args)
            args[1] = redirects[str(args[1])]
        return original(args, *pos, **kwargs)
    subprocess.Popen = popen

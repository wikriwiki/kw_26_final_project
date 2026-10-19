"""Regression for Neo4j stopping with its parent simulator process group."""
import ast
from pathlib import Path, PurePosixPath
from types import SimpleNamespace


def test_database_shutdown_finishes_before_detached_restart():
    source = Path(__file__).resolve().parents[3] / 'deploy/vast/runtime_hotfix_v22_retry_contract/switch_once.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'ensure_database')
    calls = []
    class LinuxPath(PurePosixPath):
        def resolve(self): return self
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *a): pass
    def connect(*a, **k):
        if not calls or calls[-1] != 'start':
            raise ConnectionRefusedError()
        return Connection()
    def wait(fragment, timeout):
        assert fragment.endswith(b'/pre/off') or fragment.endswith(b'-pre/off')
        calls.append('shutdown_finished')
    def run(argv, **kwargs):
        assert calls == ['shutdown_finished']
        assert kwargs['start_new_session'] is True
        assert 'NEO4J_PASSWORD' not in kwargs['env']
        calls.append('start')
    namespace = {'Path': LinuxPath, 'socket': SimpleNamespace(create_connection=connect),
                 'wait_absent': wait, 'subprocess': SimpleNamespace(run=run),
                 'time': SimpleNamespace(monotonic=lambda: 0, sleep=lambda s: None),
                 'event': lambda *a, **k: calls.append('ready')}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    namespace['ensure_database']({
        'BACKUP_NEO4J_HOME': '/workspace/no-smoking-neo4j-integration-main-v22-1154-pre/off',
        'BACKUP_NEO4J_BOLT_PORT': '17791', 'NEO4J_PASSWORD': 'not-a-real-password'})
    assert calls == ['shutdown_finished', 'start', 'ready']

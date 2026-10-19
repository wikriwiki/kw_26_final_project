import json

from deploy.vast import watch_pipeline as watcher


def test_failed_pipeline_backs_up_without_server_side_stop(tmp_path, monkeypatch):
    config = {'instance_id': watcher.INSTANCE_ID, 'prefix': 'integration-main-v21-1154',
              'project': '/workspace/frozen-v20', 'port_base': 17781}
    source = tmp_path / 'watch.json'
    source.write_text(json.dumps(config))
    pipeline = tmp_path / 'integration-main-v21-1154-pipeline.json'
    pipeline.write_text(json.dumps({'status': 'failed', 'phase': 'shared-pre', 'config': config}))
    monkeypatch.setattr(watcher, 'RESULTS', tmp_path)
    events = []
    monkeypatch.setattr(watcher, 'write_event',
                        lambda log, kind, **details: events.append((kind, details)))
    monkeypatch.setattr(watcher, 'quiesce_orphan_workers', lambda log: True)
    monkeypatch.setattr(watcher, 'supervisor_running', lambda: False)
    monkeypatch.setattr(watcher, 'FAILED_GRACE_SECONDS', 0)
    actions = []
    monkeypatch.setattr(watcher, 'backup_on_failure',
                        lambda c, phase, log: actions.append(('backup', phase)))
    monkeypatch.setattr(watcher, 'container_key_ready',
                        lambda: (_ for _ in ()).throw(AssertionError('stop key must not be used')))
    watcher.watch(source)
    assert actions == [('backup', 'shared-pre')]
    assert ('automatic_stop_suppressed', {'reason': 'pipeline_failed'}) in events


def test_only_verified_completion_may_request_vast_stop(tmp_path, monkeypatch):
    actions = []
    monkeypatch.setattr(watcher, 'container_key_ready', lambda: actions.append('key'))
    monkeypatch.setattr(watcher, 'cli', lambda *args, **kwargs: actions.append(args) or '{}')
    monkeypatch.setattr(watcher, 'write_event', lambda *args, **kwargs: actions.append(args[1]))
    watcher.stop_instance(tmp_path / 'watch.jsonl', 'pipeline_failed')
    watcher.stop_instance(tmp_path / 'watch.jsonl', 'supervisor_missing')
    watcher.stop_instance(tmp_path / 'watch.jsonl', 'model_server_unhealthy')
    watcher.stop_instance(tmp_path / 'watch.jsonl', 'pipeline_never_started')
    assert actions == ['automatic_stop_suppressed'] * 4


def test_completed_pipeline_stops_only_with_score_receipt(tmp_path, monkeypatch):
    config = {'instance_id': watcher.INSTANCE_ID, 'prefix': 'integration-main-v20-1154'}
    source = tmp_path / 'watch.json'
    source.write_text(json.dumps(config))
    (tmp_path / 'integration-main-v20-1154-pipeline.json').write_text(json.dumps({
        'status': 'complete', 'phase': 'complete', 'config': config,
        'score_sha256': 'a' * 64}))
    (tmp_path / 'integration-main-v20-1154-shared-score.json').write_text('{}')
    monkeypatch.setattr(watcher, 'RESULTS', tmp_path)
    monkeypatch.setattr(watcher, 'write_event', lambda *a, **k: None)
    verified = []
    monkeypatch.setattr(watcher, 'drive_file_verified',
                        lambda local, remote: verified.append((local.name, remote)) or True)
    actions = []
    monkeypatch.setattr(watcher, 'stop_instance', lambda log, reason: actions.append(reason))
    watcher.watch(source)
    assert actions == ['pipeline_complete']
    assert [name for name, _ in verified] == [
        'integration-main-v20-1154-shared-score.json',
        'integration-main-v20-1154-pipeline.json']


class _StopLoop(Exception):
    pass


def _failed_pipeline(tmp_path, monkeypatch):
    config = {'instance_id': watcher.INSTANCE_ID, 'prefix': 'integration-main-v22-1154',
              'project': '/workspace/frozen-v22', 'port_base': 17791}
    source = tmp_path / 'watch.json'
    source.write_text(json.dumps(config))
    (tmp_path / 'integration-main-v22-1154-pipeline.json').write_text(
        json.dumps({'status': 'failed', 'phase': 'post-off', 'config': config}))
    monkeypatch.setattr(watcher, 'RESULTS', tmp_path)
    monkeypatch.setattr(watcher, 'write_event', lambda log, kind, **details: None)
    actions = []
    monkeypatch.setattr(watcher, 'backup_on_failure', lambda c, phase, log: actions.append(phase))
    monkeypatch.setattr(watcher, 'stop_instance', lambda log, reason: actions.append(reason))
    return source, actions


def test_stale_failed_status_is_ignored_while_a_supervisor_restarts(tmp_path, monkeypatch):
    source, actions = _failed_pipeline(tmp_path, monkeypatch)
    monkeypatch.setattr(watcher, 'FAILED_GRACE_SECONDS', 0)
    monkeypatch.setattr(watcher, 'supervisor_running', lambda: True)
    monkeypatch.setattr(watcher, 'quiesce_orphan_workers',
                        lambda log: (_ for _ in ()).throw(AssertionError('must not quiesce')))
    sleeps = []

    def stop_after_three(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 3:
            raise _StopLoop

    monkeypatch.setattr(watcher.time, 'sleep', stop_after_three)
    try:
        watcher.watch(source, poll_seconds=0)
    except _StopLoop:
        pass
    assert actions == []


def test_failed_status_waits_for_the_grace_period(tmp_path, monkeypatch):
    source, actions = _failed_pipeline(tmp_path, monkeypatch)
    monkeypatch.setattr(watcher, 'FAILED_GRACE_SECONDS', 3600)
    monkeypatch.setattr(watcher, 'supervisor_running', lambda: False)
    monkeypatch.setattr(watcher, 'quiesce_orphan_workers', lambda log: True)
    calls = []

    def stop_after_two(seconds):
        calls.append(seconds)
        if len(calls) >= 2:
            raise _StopLoop

    monkeypatch.setattr(watcher.time, 'sleep', stop_after_two)
    try:
        watcher.watch(source, poll_seconds=0)
    except _StopLoop:
        pass
    assert actions == []


def test_orphans_are_signalled_through_their_real_process_group(monkeypatch):
    seen = iter([[111], [], []])
    monkeypatch.setattr(watcher, 'active_simulation_pids', lambda: next(seen, []))
    monkeypatch.setattr(watcher, 'write_event', lambda log, kind, **details: None)
    monkeypatch.setattr(watcher.os, 'getpgrp', raising=False, value=lambda: 1)
    monkeypatch.setattr(watcher.os, 'getpgid', raising=False, value=lambda pid: 222)
    signalled = []
    monkeypatch.setattr(watcher.os, 'killpg', raising=False, value=lambda group, sig: signalled.append(group))
    monkeypatch.setattr(watcher.time, 'sleep', lambda s: None)
    assert watcher.quiesce_orphan_workers('log') is True
    assert signalled == [222]


def test_backup_is_skipped_when_a_simulator_survives(monkeypatch):
    events = []
    monkeypatch.setattr(watcher, 'write_event', lambda log, kind, **details: events.append(kind))
    monkeypatch.setattr(watcher, 'active_simulation_pids', lambda: [111])
    monkeypatch.setattr(watcher.os, 'getpgrp', raising=False, value=lambda: 1)
    monkeypatch.setattr(watcher.os, 'getpgid', raising=False, value=lambda pid: 222)
    monkeypatch.setattr(watcher.os, 'killpg', raising=False, value=lambda group, sig: None)
    ticks = iter(range(0, 10000, 50))
    monkeypatch.setattr(watcher.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(watcher.time, 'sleep', lambda s: None)
    monkeypatch.setattr(watcher, 'backup_on_failure',
                        lambda *a: (_ for _ in ()).throw(AssertionError('must not back up')))
    watcher.quiesce_then_backup({}, 'post-off', 'log')
    assert 'backup_skipped_live_simulator' in events

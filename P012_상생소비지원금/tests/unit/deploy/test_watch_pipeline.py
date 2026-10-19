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
    monkeypatch.setattr(watcher, 'quiesce_orphan_workers', lambda log: None)
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

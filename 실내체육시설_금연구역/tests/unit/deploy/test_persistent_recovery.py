import importlib.util
import json
from pathlib import Path


def controller():
    path = Path(__file__).resolve().parents[3] / 'deploy/vast/runtime_hotfix_v22_night_progress/resume_runtime.py'
    spec = importlib.util.spec_from_file_location('recovery_controller_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_controller_never_overlaps_work_or_backup():
    c = controller()
    for status in ['failed', 'running', 'complete']:
        for supervisor, simulator, backup in [(1,None,None),(None,2,None),(None,None,3)]:
            assert c.decide_action({'status':status}, supervisor, simulator, backup) != 'resume'
    assert c.decide_action({'status':'failed'},None,None,None) == 'resume'
    assert c.decide_action({'status':'running'},None,None,None) == 'resume'
    assert c.decide_action({'status':'complete'},None,None,None) == 'verify_completion'


def test_failed_process_is_restarted_without_stop_or_database_restore(tmp_path, monkeypatch):
    c = controller()
    config = {'prefix': c.PREFIX}
    (tmp_path/'config.json').write_text(json.dumps(config))
    state = {'status':'failed','phase':'shared-pre','config':config,'error':'night failure'}
    (tmp_path/'pipeline.json').write_text(json.dumps(state))
    monkeypatch.setattr(c,'CONFIG',tmp_path/'config.json')
    monkeypatch.setattr(c,'PIPELINE',tmp_path/'pipeline.json')
    monkeypatch.setattr(c,'RESULTS',tmp_path)
    monkeypatch.setattr(c,'current_processes',lambda:{'supervisor':None,'simulator':None,'backup':None})
    events=[]
    monkeypatch.setattr(c,'validate',lambda:events.append('validate'))
    monkeypatch.setattr(c,'validate_extension',lambda:events.append('extension'))
    monkeypatch.setattr(c,'event',lambda event,**kwargs:events.append(event))
    monkeypatch.setattr(c,'start_model',lambda:events.append('model'))
    monkeypatch.setattr(c,'start_supervisor',lambda:events.append('resume') or 10)
    monkeypatch.setattr(c,'verify_and_stop_completed',lambda *_: (_ for _ in ()).throw(AssertionError('must not stop')))
    assert c.recover_once() == 'resume'
    assert events[-3:] == ['model','resume','automatic_recovery_started']
    saved=next((tmp_path/f'{c.PREFIX}-recovery-events').glob('*.json'))
    assert json.loads(saved.read_text()) == state

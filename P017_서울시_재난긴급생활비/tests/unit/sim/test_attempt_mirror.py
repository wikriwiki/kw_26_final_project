import io
import json
import tarfile

import pytest

from mirror_validation_run import copy_attempt_archive


NAME='a'*64+'_deliberation_request.json'


def archive(path, members):
    with tarfile.open(path,'w') as tf:
        for name,data,kind in members:
            entry=tarfile.TarInfo(name);entry.type=kind;entry.size=len(data)
            tf.addfile(entry,io.BytesIO(data))
    return path


def test_request_survives_before_response_and_repeated_copy_is_exact(tmp_path):
    path=archive(tmp_path/'a.tar',[('attempts/'+NAME,b'{"request":"already submitted"}',tarfile.REGTYPE)])
    result=copy_attempt_archive(path,tmp_path/'mirror')
    assert result['files']==1
    assert json.loads((tmp_path/'mirror'/NAME).read_bytes())['request']=='already submitted'
    assert copy_attempt_archive(path,tmp_path/'mirror')==result


def test_changed_history_preserves_original(tmp_path):
    root=tmp_path/'mirror';root.mkdir();(root/NAME).write_bytes(b'{"original":true}')
    path=archive(tmp_path/'a.tar',[('attempts/'+NAME,b'{}',tarfile.REGTYPE)])
    with pytest.raises(ValueError,match='changed'):copy_attempt_archive(path,root)
    assert (root/NAME).read_bytes()==b'{"original":true}'


@pytest.mark.parametrize('name,data,kind',[
    ('attempts/../../escape.json',b'{}',tarfile.REGTYPE),
    ('attempts/'+NAME,b'',tarfile.SYMTYPE),
    ('attempts/'+NAME,b'{partial',tarfile.REGTYPE),
])
def test_untrusted_or_partial_archive_not_committed(tmp_path,name,data,kind):
    path=archive(tmp_path/'a.tar',[(name,data,kind)])
    with pytest.raises(ValueError):copy_attempt_archive(path,tmp_path/'mirror')
    assert not (tmp_path/'mirror').exists()

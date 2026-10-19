import io
import json
import tarfile
from pathlib import Path
import pytest
from deploy.vast.restore_checkpoint import unpack_run


@pytest.mark.parametrize('name', ['../outside', 'run/../../outside', '/run/outside'])
def test_restore_rejects_archive_path_escape(tmp_path, name):
    archive=tmp_path/'a.tar.gz'
    with tarfile.open(archive,'w:gz') as out:
        item=tarfile.TarInfo(name)
        item.size=4
        out.addfile(item,io.BytesIO(b'data'))
    with pytest.raises(ValueError):
        unpack_run(archive,tmp_path/'restore')


def test_restore_rejects_links_and_restores_incremental_day_files(tmp_path):
    root=tmp_path/'restore'
    for day in ('2017-11-19','2017-11-20'):
        archive=tmp_path/(day+'.tar.gz')
        payload=json.dumps({'day':day}).encode()
        with tarfile.open(archive,'w:gz') as out:
            item=tarfile.TarInfo('run/metrics/day_'+day+'.jsonl')
            item.size=len(payload)
            out.addfile(item,io.BytesIO(payload))
        unpack_run(archive,root)
    assert len(list((root/'metrics').glob('*.jsonl')))==2
    archive=tmp_path/'link.tar.gz'
    with tarfile.open(archive,'w:gz') as out:
        item=tarfile.TarInfo('run/link')
        item.type=tarfile.SYMTYPE
        item.linkname='../outside'
        out.addfile(item)
    with pytest.raises(ValueError):
        unpack_run(archive,root)

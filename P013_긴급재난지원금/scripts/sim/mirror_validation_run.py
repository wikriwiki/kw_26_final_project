"""Copy completed JSONL records off-server while a finite validation run executes."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import re
import subprocess
import tarfile
import time


def complete_prefix(raw,previous=b''):
    boundary=raw.rfind(b'\n')+1
    prefix=raw[:boundary]
    rows=[json.loads(line) for line in prefix.splitlines() if line.strip()]
    if not prefix.startswith(previous): raise ValueError('Remote history differs from existing backup; refusing replacement')
    return prefix,rows,len(raw)-boundary


def copy_attempt_archive(archive, folder):
    """Validate all immutable JSON members before copying; never extract paths."""
    folder=Path(folder); members={}
    with tarfile.open(archive, mode='r:') as tf:
        for member in tf:
            if member.isdir() and member.name.rstrip('/')=='attempts':continue
            if not member.isfile() or not re.fullmatch(r'attempts/[0-9a-f]{64}_[a-z_]+\.json',member.name):
                raise ValueError('Unexpected attempt archive member')
            name=member.name.split('/')[1]
            if name in members:raise ValueError('Duplicate attempt member')
            data=tf.extractfile(member).read();json.loads(data)
            target=folder/name
            if target.exists() and target.read_bytes()!=data:
                raise ValueError('Previously mirrored attempt changed; refusing replacement')
            members[name]=data
    folder.mkdir(parents=True,exist_ok=True)
    for name,data in members.items():
        target=folder/name
        if target.exists():continue
        temporary=target.with_suffix('.json.mirror-pending')
        with temporary.open('wb') as fp:fp.write(data);fp.flush();os.fsync(fp.fileno())
        os.replace(temporary,target)
    return {'files':len(members),'sha256':{name:hashlib.sha256(data).hexdigest() for name,data in members.items()}}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--host',required=True);ap.add_argument('--port',required=True)
    ap.add_argument('--key',required=True);ap.add_argument('--remote',required=True);ap.add_argument('--local',required=True)
    ap.add_argument('--interval',type=int,default=180);ap.add_argument('--hours',type=float,default=8)
    ap.add_argument('--attempts',action='store_true',help='Also copy already-written immutable request, reasoning and answer JSON files')
    args=ap.parse_args()
    if not re.fullmatch(r'/data/validation_[A-Za-z0-9_]+/[A-Za-z0-9_]+',args.remote): raise ValueError('Unexpected remote run path')
    folder=Path(args.local).resolve();folder.mkdir(parents=True,exist_ok=True)
    scratch=folder/'.mirror';scratch.mkdir(exist_ok=True)
    common=['scp','-q','-i',str(Path(args.key).resolve()),'-P',args.port,'-o','BatchMode=yes','-o','ConnectTimeout=15']
    def fetch(name,optional=False):
        tmp=scratch/name
        result=subprocess.run(common+[args.host+':'+args.remote+'/'+name,str(tmp)],capture_output=True,timeout=90,
                              creationflags=0x08000000 if os.name=='nt' else 0)
        if result.returncode:
            if optional:return None
            raise RuntimeError(f'{name}: scp exit {result.returncode}')
        return tmp
    deadline=time.monotonic()+args.hours*3600
    while time.monotonic()<deadline:
        try:
            for name in ['manifest.json','frozen_inputs.json']:
                if not (folder/name).exists():
                    tmp=fetch(name);json.loads(tmp.read_bytes());os.replace(tmp,folder/name)
            final=fetch('summary.json',optional=True)
            tmp=fetch('responses.jsonl')
            old=(folder/'responses.jsonl').read_bytes() if (folder/'responses.jsonl').exists() else b''
            prefix,rows,partial=complete_prefix(tmp.read_bytes(),old)
            with tmp.open('wb') as fp:fp.write(prefix);fp.flush();os.fsync(fp.fileno())
            os.replace(tmp,folder/'responses.jsonl')
            attempt_status=None
            if args.attempts:
                archive=scratch/'attempts.tar'
                command=['ssh','-i',str(Path(args.key).resolve()),'-p',args.port,'-o','BatchMode=yes','-o','ConnectTimeout=15',args.host,
                         'tar --exclude=*.tmp -cf - -C '+args.remote+' attempts']
                with archive.open('wb') as fp:
                    result=subprocess.run(command,stdout=fp,stderr=subprocess.PIPE,timeout=120,
                                          creationflags=0x08000000 if os.name=='nt' else 0)
                # Tar exit1 may report a directory growing while immutable files
                # are copied. Validate the entire archive and capture missing new
                # files at the next pass; other statuses are transport/read errors.
                if result.returncode not in {0,1}:raise RuntimeError('Attempt archive transport failed')
                attempt_status=copy_attempt_archive(archive,folder/'attempts')
                attempt_status['tar_returncode']=result.returncode
                attempt_status['transport_diagnostic']=result.stderr.decode('utf-8',errors='replace')
            completed=False
            if final:
                summary=json.loads(final.read_bytes())
                expected=sum(v['responses'] for v in summary['variants'].values())
                if len(rows)!=expected or partial:raise ValueError('Final summary and mirrored responses differ')
                os.replace(final,folder/'summary.json');completed=True
            status={'copied_at':datetime.now(timezone.utc).isoformat(),'complete_records':len(rows),'trailing_incomplete_bytes_not_copied':partial,
                    'sha256':hashlib.sha256(prefix).hexdigest(),'run_completed':completed,
                    'scope':'Completed responses and frozen inputs; optional already-written attempt files. Data since the latest successful copy and currently generating responses can still be lost; transport failures can extend this window. Cloud Drive sync is not certified.'}
            if attempt_status is not None:
                status['attempt_files_copied']=attempt_status['files']
                with (scratch/'attempt_manifest.json').open('w',encoding='utf-8') as fp:
                    json.dump(attempt_status,fp,indent=2);fp.write('\n');fp.flush();os.fsync(fp.fileno())
                os.replace(scratch/'attempt_manifest.json',folder/'attempt_manifest.json')
            (scratch/'status.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8');os.replace(scratch/'status.json',folder/'mirror_status.json')
            print(json.dumps(status),flush=True)
            if completed:return
        except Exception as exc:
            print(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'error':str(exc)}),flush=True)
        time.sleep(args.interval)
    print('Mirror time limit reached; existing backups preserved.',flush=True)


if __name__=='__main__':main()

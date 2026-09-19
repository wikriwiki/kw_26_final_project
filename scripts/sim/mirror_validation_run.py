"""Copy completed JSONL records off-server while a finite validation run executes."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import re
import subprocess
import time


def complete_prefix(raw,previous=b''):
    boundary=raw.rfind(b'\n')+1
    prefix=raw[:boundary]
    rows=[json.loads(line) for line in prefix.splitlines() if line.strip()]
    if not prefix.startswith(previous): raise ValueError('Remote history differs from existing backup; refusing replacement')
    return prefix,rows,len(raw)-boundary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--host',required=True);ap.add_argument('--port',required=True)
    ap.add_argument('--key',required=True);ap.add_argument('--remote',required=True);ap.add_argument('--local',required=True)
    ap.add_argument('--interval',type=int,default=180);ap.add_argument('--hours',type=float,default=8)
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
            completed=False
            if final:
                summary=json.loads(final.read_bytes())
                expected=sum(v['responses'] for v in summary['variants'].values())
                if len(rows)!=expected or partial:raise ValueError('Final summary and mirrored responses differ')
                os.replace(final,folder/'summary.json');completed=True
            status={'copied_at':datetime.now(timezone.utc).isoformat(),'complete_records':len(rows),'trailing_incomplete_bytes_not_copied':partial,
                    'sha256':hashlib.sha256(prefix).hexdigest(),'run_completed':completed,
                    'scope':'Completed response records and frozen inputs only. Cloud Drive sync and in-flight calls are not certified.'}
            (scratch/'status.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8');os.replace(scratch/'status.json',folder/'mirror_status.json')
            print(json.dumps(status),flush=True)
            if completed:return
        except Exception as exc:
            print(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'error':str(exc)}),flush=True)
        time.sleep(args.interval)
    print('Mirror time limit reached; existing backups preserved.',flush=True)


if __name__=='__main__':main()

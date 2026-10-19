"""보존된 본런 zip 하나에서 관찰 시험에 필요한 것만 뽑아 pickle 로 저장한다.

읽기 전용: zip 은 열기만 하고 고치지 않는다.
뽑는 것 (갈래 on/off 각각, 사람마다):
  - profile: 페르소나 속성 (dossier.profile)
  - plans:   날짜 -> [(장소 유형 category, 계기 trigger, 실제 지출, 시각)]
  - mems:    [(날짜, 기억 종류, 업종, 계기, 지출)]
  - states:  날짜 -> (기분, 피로, 에너지, 잔액)
  - policy:  날짜 -> 정책 상태(받은 누적, 잔액 / 상생은 등록 여부·적립액)
"""
import json
import pickle
import sys
import zipfile


def extract(zip_path, out_path):
    z = zipfile.ZipFile(zip_path)
    names = set(z.namelist())
    root = next(n for n in names if n.endswith('run_manifest.json')).rsplit('/', 1)[0] + '/'
    manifest = json.loads(z.read(root + 'run_manifest.json'))

    policy = {}
    for arm in ('on', 'off'):
        with z.open(root + f'score/{arm}.sector.ledger.jsonl') as f:
            for line in f:
                r = json.loads(line)
                policy[(arm, r['aid'], r['day'])] = {
                    'received': float(r.get('grant_received_cumulative') or 0),
                    'remaining': float(r.get('grant_remaining') or 0),
                }
        cb = root + f'score/{arm}.cashback.ledger.jsonl'
        if cb in names:
            with z.open(cb) as f:
                for line in f:
                    r = json.loads(line)
                    d = policy.setdefault((arm, r['aid'], r['day']), {})
                    d['enrolled'] = bool(r.get('enrolled'))
                    d['cashback'] = float(r.get('cashback_accrued_won') or 0)

    agents, data = {}, {}
    for arm in ('on', 'off'):
        with z.open(root + f'{arm}/dossier.jsonl') as f:
            for line in f:
                r = json.loads(line)
                aid = r['aid']
                agents.setdefault(aid, r['profile'])
                plans = {}
                for p in r['plans']:
                    plans[p['day']] = [(it.get('category') or '?', it.get('trigger') or 'none',
                                        float(it.get('actual_spent') or 0), (it.get('time') or '')[:5])
                                       for it in p['items']]
                mems = [(m['day'], m.get('type'), m.get('category'), m.get('trigger'), float(m.get('spent') or 0))
                        for m in r['memories']]
                states = {s['day']: (s.get('mood'), s.get('fatigue'), s.get('energy'), s.get('balance'))
                          for s in r['states']}
                data[(arm, aid)] = {'plans': plans, 'mems': mems, 'states': states}

    out = {'manifest': manifest, 'agents': agents, 'data': data, 'policy': policy}
    with open(out_path, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(out_path, 'agents', len(agents), 'arm-agents', len(data), 'policy rows', len(policy))


if __name__ == '__main__':
    extract(sys.argv[1], sys.argv[2])

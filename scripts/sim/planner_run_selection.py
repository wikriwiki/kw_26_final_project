"""Select a declared repeat without losing provenance or hiding upstream failure."""


def select_replicate(rows,cells,config,replicate):
    candidates=config['candidates'];seeds=config['seeds']
    if len(candidates)!=1 or len(set(seeds))!=len(seeds) or replicate not in seeds:
        raise ValueError('Select a registered replicate from one frozen candidate')
    variant=candidates[0]['id']
    key=lambda r:(r['replicate'],r['variant'],r['aid'],r['case'],r['arm'])
    expected={(s,variant,c['aid'],c['case'],c['arm']) for s in seeds for c in cells}
    if len(rows)!=len(expected) or {key(r) for r in rows}!=expected:
        raise ValueError('Complete original planner matrix required before selecting a repeat')
    if not all(r.get('eligible') is True for r in rows):
        raise ValueError('Upstream failure must not be hidden by repeat selection')
    return [r for r in rows if r['replicate']==replicate]

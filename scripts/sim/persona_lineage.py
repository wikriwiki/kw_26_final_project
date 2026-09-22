"""Offline persona provenance checks and an explicit baseline-narrative overlay.

This does not update Neo4j, silently choose an ID offset, or certify behavioral
truth. Original structured values stay fixed; incompatible narrative generations
are preserved in an audit and excluded from this particular experiment input.
"""
import copy
import hashlib
import json
from pathlib import Path


def read_complete(path, *, jsonl=False):
    path = Path(path)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if len(raw) != before.st_size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Incomplete or changing source file: {path}')
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()] if jsonl else json.loads(raw)
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected nonempty persona records')
    indexed = {}
    for row in rows:
        aid = row.get('agent_id')
        if not isinstance(aid, str) or not aid or aid in indexed:
            raise ValueError('Missing or duplicate source citizen ID')
        indexed[aid] = row
    return indexed, {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'rows': len(rows)}


def audit_persona(persona, graph, narrative_source, *, source_id):
    """Compare an explicitly mapped source, including numeric narrative anchors."""
    if graph['id'] != persona['id']:
        raise ValueError('Graph/persona identity mismatch')
    result = {'id': persona['id'], 'source_id': source_id,
              'graph_lifestyle_matches_frozen': graph.get('lifestyle') == persona.get('lifestyle'),
              'rich_and_lifestyle_uuid_conflict': bool(graph.get('persona_uuid') and graph.get('nvidia_uuid_v2')
                  and graph['persona_uuid'] != graph['nvidia_uuid_v2'])}
    row = narrative_source.get(source_id)
    if row is None:
        result['source_status'] = 'unresolved'
        return result
    result.update(source_status='mapped',
        source_uuid=row.get('_match', {}).get('nvidia_uuid'),
        source_job=row['personal']['job'], frozen_job=persona['job'],
        lifestyle_exact_match=row['personality']['lifestyle'] == persona['lifestyle'],
        lifestyle_uuid_exact_match=row.get('_match', {}).get('nvidia_uuid') == graph.get('nvidia_uuid_v2'),
        job_exact_match=row['personal']['job'] == persona['job'],
        source_daily_wd=row['spending']['daily_spending_weekday'], frozen_daily_wd=persona['daily_wd'],
        daily_wd_exact_match=row['spending']['daily_spending_weekday'] == persona['daily_wd'])
    return result


def restore_baseline(personas, baseline, *, allow_mapped_residence=False):
    """Restore qualitative source only after identity + job checks, all or fail.

Numeric values can have subsequent calibration, so they are retained verbatim.
This is a declared synthetic prior, not an observed individual biography.
"""
    restored, audit = [], []
    seen = set()
    for original in personas:
        aid = original['id']
        if aid in seen:
            raise ValueError('Duplicate frozen citizen ID')
        seen.add(aid)
        if aid not in baseline:
            raise ValueError(f'Missing baseline citizen: {aid}')
        base = baseline[aid]
        for field, value in [('job', base['personal']['job']), ('gender', base['personal']['gender']),
                             ('age_group', base['personal']['age_group'])]:
            if original.get(field) != value:
                raise ValueError(f'Baseline identity/occupation mismatch: {aid}/{field}')
        residence_remapped = original.get('home_dong_code') != str(base['residence']['dong_code'])
        if residence_remapped and not allow_mapped_residence:
            raise ValueError(f'Baseline residence mismatch: {aid}')
        lifestyle = base['personality']['lifestyle']
        if not isinstance(lifestyle, str) or not lifestyle.strip():
            raise ValueError(f'Empty baseline narrative: {aid}')
        result = copy.deepcopy(original)
        removed = {k: copy.deepcopy(v) for k, v in original.items() if k.startswith('nv_') or k == 'lifestyle'}
        for key in result:
            if key.startswith('nv_'):
                result[key] = None
        result['lifestyle'] = lifestyle
        restored.append(result)
        audit.append({'id': aid, 'quarantined': removed, 'restored_lifestyle': lifestyle,
                      'baseline_record': copy.deepcopy(base),
                      'residence_remapped': residence_remapped,
                      'retained_mapped_home_dong_code': original.get('home_dong_code'),
                      'changed_fields': sorted(k for k in result if result[k] != original[k])})
    return restored, audit


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--frozen', type=Path, action='append', required=True)
    ap.add_argument('--graph', type=Path, required=True)
    ap.add_argument('--narrative', type=Path, required=True)
    ap.add_argument('--baseline', type=Path)
    ap.add_argument('--source-seq-offset', type=int, required=True,
                    help='Explicit hypothesis to audit; never inferred or applied to repair')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError('Refusing overwrite')
    narrative, meta = read_complete(args.narrative, jsonl=True)
    baseline, baseline_meta = read_complete(args.baseline) if args.baseline else ({}, None)
    graph_raw = args.graph.read_bytes()
    graph_rows = json.loads(graph_raw)
    graph = {r['id']: r for r in graph_rows}
    if len(graph) != len(graph_rows):
        raise ValueError('Duplicate graph citizens')
    people, sources = {}, {}
    for path in args.frozen:
        raw = path.read_bytes()
        if len(raw) != path.stat().st_size:
            raise ValueError('Incomplete frozen source')
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        for persona in json.loads(raw)['personas']:
            if persona['id'] in people and persona != people[persona['id']]:
                raise ValueError('Inconsistent frozen persona across sources')
            people[persona['id']] = persona
    rows = []
    for aid, persona in sorted(people.items()):
        prefix, seq = aid.rsplit('_', 1)
        source_id = f'{prefix}_{int(seq) + args.source_seq_offset:03d}'
        row = audit_persona(persona, graph[aid], narrative, source_id=source_id)
        if baseline:
            base = baseline[aid]
            row['baseline_job_exact_match'] = base['personal']['job'] == persona['job']
            row['baseline_lifestyle_exact_match'] = base['personality']['lifestyle'] == persona['lifestyle']
        rows.append(row)
    result = {'scope': 'Read-only selected-citizen provenance audit. Exact job string mismatch is not itself a semantic contradiction.',
              'frozen_sources': sources, 'graph_sha256': hashlib.sha256(graph_raw).hexdigest(),
              'narrative_file': meta, 'source_seq_offset': args.source_seq_offset,
              'baseline_file': baseline_meta,
              'people': len(rows), 'mapped': sum(r['source_status'] == 'mapped' for r in rows),
              'lifestyle_and_uuid_exact_match': sum(r.get('lifestyle_exact_match', False) and r.get('lifestyle_uuid_exact_match', False) for r in rows),
              'job_string_difference': sum(r.get('job_exact_match') is False for r in rows),
              'daily_wd_difference': sum(r.get('daily_wd_exact_match') is False for r in rows),
              'rich_uuid_conflict': sum(r['rich_and_lifestyle_uuid_conflict'] for r in rows), 'rows': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k in ['people', 'mapped', 'lifestyle_and_uuid_exact_match', 'job_string_difference', 'daily_wd_difference', 'rich_uuid_conflict']}))


if __name__ == '__main__':
    main()

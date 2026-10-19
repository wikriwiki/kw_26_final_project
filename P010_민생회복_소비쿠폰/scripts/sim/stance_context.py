"""Budgeted interview context selected by recorded circumstances, never stance.

Full packets stay intact. Original experience items remain verbatim; projections
copy explicit fields and retain parent pointers. Selection is an evidence window,
not a claim that the selected experiences represent an agent's entire history.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from datetime import date
import math

try:
    from .evidence_integrity import EvidenceError, canonical, digest, iso_day, seal, verify
except ImportError:
    from evidence_integrity import EvidenceError, canonical, digest, iso_day, seal, verify


MAX_ITEM_EVALUATIONS = 128
META_KINDS = {'model_call_reference', 'fallback_diagnostic'}
EXPERIENCE_KINDS = {'executed_receipt', 'executed_event', 'stated_rationale', 'social_interaction'}
PERSONA_GROUPS = {
    'identity_and_smoking': ('id','age','age_group','gender','sex','life_stage','smoking_status'),
    'resources_and_routine': ('income','spend_decile','daily_wd','daily_we','we_wd_ratio',
                             'cat_ratio_wd','cat_ratio_we','delivery_days','home_h_wd','home_h_we','mobility'),
    'work_and_location': ('job','work_dong','work_dong_code','work_poi','commute_min',
                          'home_dong','home_dong_code','home_poi','home_gu'),
    'lifestyle': ('lifestyle','tendency','nv_hobbies'),
    'recorded_background': ('nv_cultural','nv_education','nv_marital','nv_family'),
    'recorded_career_and_skills': ('nv_career','nv_skills'),
    'assigned_narrative': ('nv_summary',),
}
STATE_FIELDS = ('balance','month_spent','energy','mood','fatigue','yest_sat','grant_remaining')
DISTRICTS = {'11350','11650','11710'}
FACILITIES = {'billiard','indoor_golf','screen_golf','당구장','당구','골프연습장','실내골프연습장','스크린골프장','스크린골프'}


def _context(item):
    value = item.get('value')
    if not isinstance(value,dict):
        return None, ''
    if isinstance(value.get('context'),dict):
        return value['context'], '/context'
    return value, ''


def _is_known(value):
    if value is None:
        return False
    if isinstance(value,str):
        return value.strip().casefold() not in {'','unknown','미상','정보 없음','알 수 없음'}
    return True


def _projection(packet, parent, kind, group, values, fields, pointer, *, day=None, source_kind):
    present = {key:values[key] for key in fields if key in values}
    payload = {'group':group,'values':present,
               'known_fields':[key for key in fields if key in present and _is_known(present[key])],
               'unknown_fields':[key for key in fields if key not in present or not _is_known(present[key])],
               'source_kind':source_kind}
    return {'evidence_id':'VIEW_'+digest([parent['evidence_id'],pointer,group,payload]),
            'agent_id':packet['agent_id'],'day':day or parent['day'],'kind':kind,
            'text':canonical(payload),'value':payload,
            'source_ref':{'parent_evidence_id':parent['evidence_id'],
                          'parent_packet_sha256':packet['integrity_sha256'],
                          'parent_pointer':pointer,'projected_fields':list(fields),
                          'derivation':'explicit_recorded_field_projection'}}


def _value(item):
    value=item.get('value')
    return value if isinstance(value,dict) else {}


def _facts(item):
    value=_value(item)
    return value.get('values',value)


def _relevance(item):
    """No text sentiment matching; leisure alone never proves policy exposure."""
    value=_facts(item)
    registry_hash=value.get('poi_registry_sha256')
    registry_reference=(isinstance(registry_hash,str) and len(registry_hash)==64
                        and all(char in '0123456789abcdef' for char in registry_hash))
    if registry_reference and (value.get('is_evaluation_poi') is True or value.get('policy_target') is True):
        return 'explicit_target'
    facility = value.get('facility_type') or value.get('sub_category') or value.get('category')
    if isinstance(facility,str) and facility in FACILITIES:
        # A generic golf driving range can be outdoors. A district code does
        # not resolve that missing indoor-facility fact.
        return ('explicit_target' if facility!='골프연습장' and str(value.get('district_code')) in DISTRICTS
                else 'facility_topic_location_unknown')
    return 'unknown'


def _rank(item):
    """Chronological/event ordering without random evidence-ID or stance text."""
    value=_facts(item)
    order=value.get('event_order',value.get('order',0))
    order=float(order) if type(order) in (int,float) and math.isfinite(order) else 0.0
    return (item['day'], order, str(value.get('scheduled_time',value.get('time',''))),
            str(value.get('poi_id',value.get('poi_name',''))), item['kind'])


def _period(day, first, last):
    if date.fromisoformat(day)<first:
        return 'before_window'
    span=(last-first).days+1
    return ('early','middle','late')[min(2,3*(date.fromisoformat(day)-first).days//span)]


def _family(item):
    return {'executed_receipt':'executed','executed_event':'executed',
            'stated_rationale':'public_statement','social_interaction':'social'}.get(item['kind'],item['kind'])


def _candidates(packet):
    source=packet['evidence_items']
    contexts=[item for item in source if item['kind']=='context']
    decoded=[(item,*_context(item)) for item in contexts]
    decoded=[row for row in decoded if row[1] is not None]
    candidates=[]
    personal=[row for row in decoded if isinstance(row[1].get('persona'),dict)]
    if personal:
        # The latest recorded persona is explicit, never assembled from stale
        # non-null fields after the current snapshot says a field is unknown.
        latest=max(row[0]['day'] for row in personal)
        options=[row for row in personal if row[0]['day']==latest]
        parent, context, prefix=max(options,key=lambda row:(
            sum(key in row[1]['persona'] for fields in PERSONA_GROUPS.values() for key in fields),
            row[0].get('source_ref',{}).get('field')=='initial_context'))
        for group, fields in PERSONA_GROUPS.items():
            candidates.append(_projection(packet,parent,'persona_snapshot',group,context['persona'],fields,
                prefix+'/persona',source_kind='assigned_simulated_profile_not_observed_attitude'))
    # Keep at most one equivalent state per day and one copy of each memory
    # value. Repeated prompt contexts must not crowd out other evidence channels.
    state_seen, memory_seen = set(), set()
    for parent, context, prefix in sorted(decoded,key=lambda row:row[0]['day'],reverse=True):
        state=context.get('state')
        if isinstance(state,dict):
            key=(parent['day'],digest({k:state[k] for k in STATE_FIELDS if k in state}))
            if key not in state_seen:
                candidates.append(_projection(packet,parent,'personal_state','prior_state_at_dawn',state,STATE_FIELDS,
                    prefix+'/state',source_kind='modeled_prior_state_not_end_of_day_or_emotion_test'))
                state_seen.add(key)
        memories=context.get('memory')
        if not isinstance(memories,list):
            continue
        for index,memory in enumerate(memories):
            if not isinstance(memory,dict) or not memory.get('day'):
                continue  # Undated summaries cannot be safely used as past experience.
            day=iso_day(memory['day'])
            if day>=parent['day'] or day>packet['through_day']:
                raise EvidenceError('remembered experience is not earlier than its recorded context')
            key=memory.get('id') or digest({key:value for key,value in memory.items() if key not in {'score','days_ago'}})
            if key in memory_seen:
                continue
            memory_seen.add(key)
            kind='remembered_experience' if memory.get('type')=='visited' else 'reported_information'
            fields=tuple(memory)
            projected=_projection(packet,parent,kind,'recorded_memory',memory,fields,
                prefix+f'/memory/{index}',
                source_kind='stored_memory_summary_not_a_new_receipt' if kind=='remembered_experience'
                            else 'heard_or_stored_information_not_verified_event')
            # item.day is when this memory was available in committed context;
            # value.values.day and source_occurrence_day retain its event date.
            projected['source_occurrence_day']=day
            candidates.append(projected)
    candidates.extend(item for item in source if item['kind'] in EXPERIENCE_KINDS)
    return candidates


def _ordered(candidates, first, last):
    """Prioritize circumstances, then balanced time/channel round-robin.

    Numeric low AND high satisfaction and incomplete AND completed purchases
    get symmetric slots. These are recorded outcomes, never inferred opinions.
    """
    priority=[]
    persona=[item for item in candidates if item['kind']=='persona_snapshot']
    groups=list(PERSONA_GROUPS)
    priority.extend(sorted(persona,key=lambda item:groups.index(_value(item)['group'])))
    # One prior-state observation in each temporal third, chosen near its center.
    for period in ('early','middle','late'):
        states=[item for item in candidates if item['kind']=='personal_state'
                and _period(item['day'],first,last)==period]
        if states:
            states.sort(key=_rank)
            priority.append(states[len(states)//2])
    actual=[item for item in candidates if item['kind'] in {'executed_receipt','executed_event'}]
    # First cover a neutral actual record in every third before emphasizing any
    # outcome extreme. Relevant facility facts take precedence when available.
    for period in ('early','middle','late'):
        rows=sorted((item for item in actual if _period(item['day'],first,last)==period),key=_rank)
        relevant=[item for item in rows if _relevance(item)!='unknown']
        rows=relevant or rows
        if rows:
            priority.append(rows[len(rows)//2])
    for period in ('early','middle','late'):
        rows=[item for item in actual if _period(item['day'],first,last)==period]
        targets=sorted((item for item in rows if _relevance(item)!='unknown'),key=_rank)
        if targets:
            priority.append(targets[len(targets)//2])
        satisfactions=[(item,_facts(item).get('actual_satisfaction')) for item in rows]
        satisfactions=[(item,value) for item,value in satisfactions
                       if type(value) in (int,float) and math.isfinite(value)]
        if satisfactions:
            priority.extend([min(satisfactions,key=lambda pair:(pair[1],_rank(pair[0])))[0],
                             max(satisfactions,key=lambda pair:(pair[1],_rank(pair[0])))[0]])
        for statuses in ({'not_purchased','reduced'},{'purchased'}):
            purchase=[item for item in rows if _facts(item).get('purchase_status') in statuses]
            if purchase:
                purchase.sort(key=_rank)
                priority.append(purchase[len(purchase)//2])
    queues=defaultdict(list)
    for item in candidates:
        queues[(_period(item['day'],first,last),_family(item),_relevance(item))].append(item)
    queues={key:deque(sorted(rows,key=_rank)) for key,rows in queues.items()}
    seen=set()
    for item in priority:
        if item['evidence_id'] not in seen:
            seen.add(item['evidence_id'])
            yield item
    # Sorted families within each period provide reproducibility without a
    # recency or arbitrary descending record-ID preference.
    keys=sorted(queues,key=lambda key:(('early','middle','late','before_window').index(key[0]),key[1],key[2]))
    while any(queues.values()):
        for key in keys:
            if queues[key]:
                item=queues[key].popleft()
                if item['evidence_id'] not in seen:
                    seen.add(item['evidence_id'])
                    yield item


def select_stance_context(packet, tokenizer, input_budget, prompt_tokens):
    """Return a sealed bounded packet; original source packet is never mutated.

    At most 128 single-item token estimates plus an exact final-fit check are
    performed. Large items may be skipped, but never silently truncated.
    """
    verify(packet)
    if packet.get('kind')!='grounded_interview_packet' or packet.get('schema_version')!=1:
        raise EvidenceError('stance selection requires a grounded interview packet')
    if type(input_budget) is not int or input_budget<=0:
        raise EvidenceError('input budget must be a positive integer')
    through=iso_day(packet['through_day'])
    source=packet.get('evidence_items')
    if not isinstance(source,list):
        raise EvidenceError('packet evidence_items must be a list')
    days=packet.get('days') or [item['day'] for item in source] or [through]
    if any(iso_day(day)>through for day in days):
        raise EvidenceError('packet day coverage includes the future')
    ids=set()
    for item in source:
        if (not isinstance(item,dict) or not isinstance(item.get('evidence_id'),str)
                or item['evidence_id'] in ids or iso_day(item.get('day'))>through
                or item['day'] not in days
                or item.get('agent_id',packet['agent_id'])!=packet['agent_id']
                or any(key in item and item[key]!=packet.get(key) for key in ('run_id','arm'))
                or not isinstance(item.get('kind'),str) or not isinstance(item.get('text'),str)):
            raise EvidenceError('duplicate, foreign, future, or malformed stance evidence')
        ids.add(item['evidence_id'])
        if item['kind'] in {'executed_receipt','executed_event'}:
            value=_value(item)
            if any(key in value and value[key]!=packet.get(key) for key in ('run_id','arm','agent_id')):
                raise EvidenceError('foreign executed experience in stance context')
    first,last=date.fromisoformat(min(days)),date.fromisoformat(through)
    candidates=_candidates(packet)
    out={key:value for key,value in packet.items() if key not in {'integrity_sha256','evidence_items','selection'}}
    selection={'method':'circumstance_time_channel_balanced_v1','parent_packet_sha256':packet['integrity_sha256'],
        'total_source_items':len(source),'candidate_items':len(candidates),'selected_items':0,
        'omitted_items':len(source),'projected_source_items':0,'source_items_without_selected_representation':len(source),
        'excluded_metadata_items':sum(item['kind'] in META_KINDS for item in source),
        'excluded_other_source_items':sum(item['kind'] not in EXPERIENCE_KINDS|META_KINDS|{'context'} for item in source),
        'input_budget_tokens':input_budget,'candidate_evaluation_limit':MAX_ITEM_EVALUATIONS,
        'evaluated_candidates':0,'not_evaluated_due_to_work_limit':0,'skipped_for_budget':0,
        'source_kind_counts':dict(sorted(Counter(item['kind'] for item in source).items())),
        'source_day_counts':dict(sorted(Counter(item['day'] for item in source).items())),
        'omitted_source_kind_counts':{},'omitted_source_day_counts':{},
        'selected_kind_counts':{},'selected_day_counts':{},'candidate_period_counts':dict(Counter(_period(item['day'],first,last) for item in candidates)),
        'selected_period_counts':{},'candidate_relevance_counts':dict(Counter(_relevance(item) for item in candidates)),
        'selected_relevance_counts':{},'known_unknown_persona_source':'available' if any(item['kind']=='persona_snapshot' for item in candidates) else 'unavailable',
        'personal_context_selected':False,'selected_evidence_available':False,
        'limitations':['Selection does not infer stance, motives, private life, or policy causation.',
                      'Missing fields stay unknown; assigned profile and modeled state are not human observations.',
                      'Leisure without a recorded facility/district link does not prove policy-target exposure.',
                      'Omitted source items and unselected periods are not evidence of absence.',
                      'Whole source items remain verbatim; projections copy only recorded fields.']}
    out.update(evidence_items=[],selection=selection)
    baseline=prompt_tokens(tokenizer,out)
    if baseline>input_budget:
        raise EvidenceError('question and selection metadata exceed the input budget')
    ordered=list(_ordered(candidates,first,last))
    selected=[]
    estimated=baseline
    evaluated=0
    skipped=0
    for item in ordered[:MAX_ITEM_EVALUATIONS]:
        evaluated+=1
        # Estimate each item independently: no repeated tokenization of an
        # ever-growing history for every candidate.
        marginal=max(1,prompt_tokens(tokenizer,{**out,'evidence_items':[item]})-baseline)
        if estimated+marginal<=input_budget:
            selected.append(item)
            estimated+=marginal
        else:
            skipped+=1
    def finalize():
        out['evidence_items']=selected
        originals={item['evidence_id'] for item in selected if item['evidence_id'] in ids}
        parents={item.get('source_ref',{}).get('parent_evidence_id') for item in selected}- {None}
        selection.update(selected_items=len(selected),omitted_items=len(source)-len(originals),
            projected_source_items=len(parents),source_items_without_selected_representation=len(ids-originals-parents),
            evaluated_candidates=evaluated,not_evaluated_due_to_work_limit=max(0,len(ordered)-evaluated),
            skipped_for_budget=skipped,selected_kind_counts=dict(sorted(Counter(item['kind'] for item in selected).items())),
            selected_day_counts=dict(sorted(Counter(item['day'] for item in selected).items())),
            selected_period_counts=dict(Counter(_period(item['day'],first,last) for item in selected)),
            selected_relevance_counts=dict(Counter(_relevance(item) for item in selected)),
            omitted_source_kind_counts=dict(sorted(Counter(item['kind'] for item in source if item['evidence_id'] not in originals).items())),
            omitted_source_day_counts=dict(sorted(Counter(item['day'] for item in source if item['evidence_id'] not in originals).items())),
            personal_context_selected=any(item['kind']=='persona_snapshot' for item in selected),
            selected_evidence_available=bool(selected))
    finalize()
    # Exact rendering is authoritative; additive BPE estimates and changing
    # metadata lengths can differ. Drop only complete lowest-priority items.
    if prompt_tokens(tokenizer,out)>input_budget:
        proposed=list(selected)
        skipped_before=skipped
        low,high=0,len(proposed)
        while low<high:
            middle=(low+high+1)//2
            selected=proposed[:middle]
            skipped=skipped_before+len(proposed)-middle
            finalize()
            if prompt_tokens(tokenizer,out)<=input_budget:
                low=middle
            else:
                high=middle-1
        selected=proposed[:low]
        skipped=skipped_before+len(proposed)-low
        finalize()
        if prompt_tokens(tokenizer,out)>input_budget:
            raise EvidenceError('final selection metadata exceeds the input budget')
    return seal(out)

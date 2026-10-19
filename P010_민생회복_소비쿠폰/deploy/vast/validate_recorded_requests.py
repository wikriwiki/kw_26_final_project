"""Bounded real-model replays of recorded difficult requests; no graph writes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/sim'))
sys.path.insert(0, str(ROOT/'scripts'))
from grounded_schema import stage1_format, stage2_format, night_format
from stage1_intent import Stage1Output, _extract_json, _number_evidence_lines, _evidence_lines
from stage2_poi import Stage2Pick
from night_intent_llm import IntentOutput
from no_smoking_prompts import SYSTEM_STAGE2, SYSTEM_NIGHT
from prompts.no_smoking_v1 import SYSTEM_PROMPT
from prompt_grounding import validate_stated_reason
from prompt_budget import check_request_budget
from openai import OpenAI


def candidate_map(request):
    user=request['messages'][-1]['content']
    schema=request['response_format']['json_schema']['schema']['properties']['picks']['items']
    pids=schema['properties']['poi_id']['enum']
    orders=schema['properties']['order']['enum']
    matches=list(re.finditer(r'^### 이벤트 (\d+) \|',user,re.M))
    mapping={}
    for index,match in enumerate(matches):
        order=int(match.group(1))
        if order not in orders:
            continue
        block=user[match.end():matches[index+1].start() if index+1<len(matches) else len(user)]
        mapping[order]=[{'poi_id':pid} for pid in pids if re.search(r'(?<!\w)'+re.escape(pid)+r'(?!\w)',block)]
    if set(mapping)!=set(orders) or not all(mapping.values()):
        raise ValueError('Recorded candidate mapping is incomplete')
    return mapping


def probe(case, client, out):
    request=copy.deepcopy(case['request'])
    stage=case['stage']
    request['messages'][0]['content']={'stage1':SYSTEM_PROMPT,'stage2':SYSTEM_STAGE2,'night_intent':SYSTEM_NIGHT}[stage]
    user=request['messages'][-1]['content']
    if stage=='stage1':
        user=_number_evidence_lines(re.sub(r'^\[E\d{4}\] ', '', user, flags=re.M))
    refs=_evidence_lines(user)
    collected={}
    candidates=candidate_map(request) if stage=='stage2' else None
    previous=None
    calls=[]
    start=time.monotonic()
    for attempt in range(6):
        now=copy.deepcopy(request)
        now['messages'][-1]['content']=user
        now['temperature']=0
        now['seed']=int(request.get('seed',42))+attempt
        if stage=='stage1':
            now['response_format']=stage1_format(refs)
        elif stage=='stage2':
            remaining=[order for order in candidates if order not in collected]
            now['response_format']=stage2_format(request['response_format'],remaining,candidates,refs)
            if collected:
                now['messages'][-1]['content']+='\n이미 검증된 선택은 반복하지 말고 누락된 order '+', '.join(map(str,remaining))+'만 출력하세요.'
        else:
            now['response_format']=night_format(refs,tuple(case['pair']))
        if previous:
            now['messages'][-1]['content']+='\n직전 검증 오류: '+previous+'. 올바른 입력의 근거와 필드를 사용하세요.'
        try:
            budget=check_request_budget(now)
            response=client.chat.completions.create(**now)
            raw=response.choices[0].message.content
            calls.append({'attempt':attempt,'budget':budget,'finish':response.choices[0].finish_reason,
                'usage':response.usage.model_dump(),'raw':raw})
            data=json.loads(_extract_json(raw))
            values=data.get('events',[]) if stage=='stage1' else data.get('picks',[]) if stage=='stage2' else [data]
            for value in values:
                ref=value.get('evidence_ref')
                if ref not in refs:
                    raise ValueError('Unknown evidence reference')
                value['evidence_quote']=refs[ref]
                validate_stated_reason(value,user,reason_key='pick_reason' if stage=='stage2' else 'reasoning')
            if stage=='stage1':
                if not values or values[0]['anchor']!='residence' or values[-1]['anchor']!='residence':
                    raise ValueError('First and last events must use residence')
                times=[int(v['time'][:2])*60+int(v['time'][3:]) for v in values]
                if any(b-a<20 for a,b in zip(times,times[1:])):
                    raise ValueError('Event times must increase by at least 20 minutes')
                Stage1Output.model_validate(data)
            elif stage=='stage2':
                seen=set()
                for value in values:
                    pick=Stage2Pick.model_validate(value)
                    if pick.order not in remaining or pick.order in seen:
                        raise ValueError('Duplicate or unexpected order')
                    if pick.poi_id not in {c['poi_id'] for c in candidates[pick.order]}:
                        raise ValueError('Cross-order POI')
                    seen.add(pick.order)
                collected.update((v['order'],v) for v in values)
                if set(collected)!=set(candidates):
                    raise ValueError('Missing orders: '+str(sorted(set(candidates)-set(collected))))
            else:
                parsed=IntentOutput.model_validate(data)
                if (parsed.initiator_id,parsed.recipient_id)!=tuple(case['pair']):
                    raise ValueError('Participants changed')
                if parsed.plan_signal.should_inject and parsed.intent!='약속':
                    raise ValueError('Only appointments inject plans')
            result={'id':case['id'],'stage':stage,'status':'passed','calls':calls,
                    'elapsed':round(time.monotonic()-start,3)}
            break
        except Exception as exc:
            error=f'{type(exc).__name__}: {str(exc)[:200]}'
            if error==previous:
                result={'id':case['id'],'stage':stage,'status':'failed','error':error,'calls':calls}
                break
            previous=error
    else:
        result={'id':case['id'],'stage':stage,'status':'failed','error':previous,'calls':calls}
    (out/(case['id']+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    os.environ.update(SIM_PROMPT_TOKEN_GUARD='required',SIM_MODEL_CONTEXT_LENGTH='16384',
        SIM_TOKENIZER_PATH=str(ROOT/'output/experiments/no_smoking_zone/runtime/tokenizer'))
    cases=[json.loads(line) for line in args.cases.read_text(encoding='utf-8').splitlines()]
    client=OpenAI(base_url='http://127.0.0.1:8000/v1',api_key='EMPTY',timeout=180,max_retries=0)
    results=[]
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures=[pool.submit(probe,case,client,args.out) for case in cases]
        for future in as_completed(futures):
            result=future.result()
            results.append(result)
            print(json.dumps({'finished':len(results),'total':len(cases),'status':result['status'],'stage':result['stage']}),flush=True)
    summary={'status':'passed' if all(r['status']=='passed' for r in results) else 'failed',
             'cases':len(cases),'passed':sum(r['status']=='passed' for r in results),
             'elapsed_seconds':round(time.monotonic()-started,3),'workers':16,'context_length':16384,
             'failures':[{'id':r['id'],'stage':r['stage'],'error':r['error']} for r in results if r['status']!='passed']}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)
    raise SystemExit(0 if summary['status']=='passed' else 1)

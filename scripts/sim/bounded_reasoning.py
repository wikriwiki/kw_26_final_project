"""Explicit two-call bounded deliberation protocol for an existing SGLang server.

The first call has a registered deliberation budget; the second continues with a
closed reasoning segment and constrained JSON. This is not a first-response-only
protocol or an unreported repair. No custom server code execution is enabled.
"""
import json
from urllib.request import Request,urlopen


def post(base,payload,timeout):
    req=Request(base+'/generate',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urlopen(req,timeout=timeout) as response: return json.load(response)


def run(*,prefix,schema,base,seed,thinking_tokens,answer_tokens,sampling,timeout,on_deliberation):
    params=dict(sampling,sampling_seed=seed,max_new_tokens=thinking_tokens,stop=['</think>'],no_stop_trim=False)
    request={'text':prefix,'sampling_params':params,'require_reasoning':False,'stream':False}
    deliberation=post(base,request,timeout)
    text=deliberation['text']
    finish=deliberation['meta_info'].get('finish_reason') or {}
    kind=finish.get('type')
    record={'request':request,'response':deliberation,'forced_reasoning_boundary':kind=='length'}
    on_deliberation(record)
    # Matched </think> or the registered budget are expected; EOS/errors are not.
    if kind not in {'stop','length'}:
        raise ValueError(f'Unexpected deliberation finish: {finish}')
    if kind=='stop' and finish.get('matched')!='</think>':
        raise ValueError(f'Unexpected deliberation stop: {finish}')
    # Some tokenizers surface special stop text even with no_stop_trim=False.
    if text.endswith('</think>'): text=text[:-len('</think>')]
    answer_request={'text':prefix+text+'\n</think>\n\n','sampling_params':dict(sampling,
        sampling_seed=seed,max_new_tokens=answer_tokens,json_schema=json.dumps(schema)),
        'require_reasoning':False,'stream':False}
    answer=post(base,answer_request,timeout)
    return record,{'request':answer_request,'response':answer}

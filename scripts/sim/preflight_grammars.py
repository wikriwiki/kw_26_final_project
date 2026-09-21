"""Compile every cell's decoding grammar before the run, so a bad one cannot kill the server.

2026-09-21: v20 crashed the SGLang server twice within minutes of starting. The cause was
a single cell whose EBNF aborted xgrammar in C++:

    terminate called after throwing an instance of 'xgrammar::LogFatalError'
      what(): /project/cpp/earley_parser.cc:203: The element type is not supported! The type is: 5

Three attempts out of 728 used such a grammar. Each one took the whole server down, and
the planner kept going - recording every subsequent timeout as `eligible=False`. That is
worse than an empty cell: it is a wrong value entering the data. Thirty minutes of the
round had to be discarded.

So this compiles each distinct grammar first, in a forked child, because a C++ abort
cannot be caught in-process. What it produces is a list of cell keys whose grammar the
server cannot accept. The run excludes those cells and records why, the same way the
resource gate records a plan it proved impossible.

It does not try to repair the grammar. Two of 452 distinct grammars failed and they were
neither the largest (a 1.38MB grammar compiles) nor structurally malformed (no empty
alternations, no undefined references, nesting depth 1). Finding the xgrammar bug is
upstream work; refusing to send it is this repository's job.

    python scripts/sim/preflight_grammars.py --source action_source.json \\
        --config data/experiments/validation_v21_cohort60_v36.json \\
        --tokenizer <path> --out preflight_grammars.json
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temporal_choice_grammar import build


def server_tokenizer_info(tokenizer_path):
    """The TokenizerInfo the server builds, not a convenient approximation.

    The first version of this check passed a grammar that then killed the server anyway.
    It compiled with `TokenizerInfo.from_huggingface(tokenizer)`; SGLang compiles with the
    model's own vocab_size and stop tokens (xgrammar_backend.XGrammarGrammarBackend), and
    the automaton those produce is not the same one. A check that compiles a different
    grammar than the server does is not a check.
    """
    import json as _json
    import xgrammar
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    config = _json.loads((Path(tokenizer_path) / 'config.json').read_text(encoding='utf-8'))
    vocab_size = config.get('vocab_size') or len(tokenizer)
    stop = config.get('eos_token_id')
    if stop is None:
        gen = Path(tokenizer_path) / 'generation_config.json'
        if gen.exists():
            stop = _json.loads(gen.read_text(encoding='utf-8')).get('eos_token_id')
    if isinstance(stop, int):
        stop = [stop]
    return xgrammar.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=vocab_size, stop_token_ids=stop), vocab_size, stop


def compiles(ebnf, tokenizer_info):
    """True if xgrammar can compile this grammar. Forked, because an abort is not catchable.

    Compiled the way the server compiles it: a raw EBNF string, and a GrammarCompiler with
    its default thread count. Pinning max_threads=1 hid a failure that only appears when
    the compile is threaded.
    """
    import xgrammar
    pid = os.fork()
    if pid == 0:
        try:
            xgrammar.GrammarCompiler(tokenizer_info=tokenizer_info).compile_grammar(ebnf)
            os._exit(0)
        except BaseException:
            os._exit(3)
    _, status = os.waitpid(pid, 0)
    return status == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    source = json.loads(Path(args.source).read_text(encoding='utf-8'))
    step = config.get('temporal_clock_step')
    if step is None:
        print('이 설정은 시간 문법을 쓰지 않는다 — 검사할 것이 없다')
        io.open(args.out, 'w', encoding='utf-8', newline='\n').write(
            json.dumps({'checked': 0, 'rejected': [], 'grammar_free': True}, ensure_ascii=False))
        return 0

    info, vocab_size, stop = server_tokenizer_info(args.tokenizer)
    print('토크나이저: vocab_size=%s · stop_token_ids=%s' % (vocab_size, stop))

    # build() reads cell['has_work'] through the activity catalog, and the planner sets it
    # from the persona just before building. Without it every cell raises and the check
    # reports 960 build failures instead of the two grammars that actually matter.
    people = {p['id']: p for p in source['personas']}

    verdict = {}          # grammar sha256 -> bool
    rejected, unbuildable = [], []
    for cell in source['cells']:
        cell['has_work'] = bool(people[cell['aid']].get('work_poi_id'))
        key = {k: cell[k] for k in ('aid', 'case', 'arm')}
        try:
            ebnf, audit = build(cell, clock_step=step,
                                last_start_not_before=config.get('last_start_not_before'),
                                allow_zone_commitments=config.get('allow_zone_commitments', False))
        except Exception as exc:
            unbuildable.append(dict(key, reason='build failed: %s' % exc))
            continue
        digest = hashlib.sha256(ebnf.encode('utf-8')).hexdigest()
        if digest not in verdict:
            verdict[digest] = compiles(ebnf, info)
        if not verdict[digest]:
            rejected.append(dict(key, grammar_sha256=digest, grammar_bytes=len(ebnf),
                                 productions=audit.get('productions')))

    result = {
        'checked_cells': len(source['cells']),
        'distinct_grammars': len(verdict),
        'uncompilable_grammars': sum(1 for ok in verdict.values() if not ok),
        'rejected': rejected,
        'unbuildable': unbuildable,
        'tokenizer': {'vocab_size': vocab_size, 'stop_token_ids': stop,
                      'note': '서버(xgrammar_backend)와 같은 값으로 컴파일했다.'},
        'reason': ('xgrammar 가 컴파일하지 못하는 문법을 서버에 보내면 C++ abort 로 '
                   '서버 전체가 죽고, 그 뒤 모든 칸이 timeout 을 값으로 기록한다.'),
        'note': '제외된 칸은 행렬을 불완전하게 만든다. 라운드 문서에 그대로 적을 것.',
    }
    io.open(args.out, 'w', encoding='utf-8', newline='\n').write(
        json.dumps(result, ensure_ascii=False, indent=1))
    print('wrote', args.out)
    print('  칸 %d · 서로 다른 문법 %d · 컴파일 불가 %d'
          % (result['checked_cells'], result['distinct_grammars'],
             result['uncompilable_grammars']))
    print('  제외할 칸 %d · 빌드 실패 %d' % (len(rejected), len(unbuildable)))
    for row in rejected[:8]:
        print('     %s · %s · %s  (%d자)'
              % (row['aid'], row['case'], row['arm'], row['grammar_bytes']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Force xgrammar to compile grammars on one thread inside SGLang.

2026-09-21: v20 could not finish. The server died four times with

    terminate called after throwing an instance of 'xgrammar::LogFatalError'
      what(): /project/cpp/earley_parser.cc:203: The element type is not supported! The type is: 5

A C++ abort, not a Python exception, so SGLang's `except RuntimeError` in
`dispatch_ebnf` never sees it and the whole server goes down. The planner then kept
sending to a dead server and recorded every timeout as `eligible=False` - a wrong value
rather than a missing one.

What actually triggers it, measured on this machine with 233 grammars that all compile
fine one at a time:

    shared GrammarCompiler, 8 threads   abort at ~100 grammars
    shared GrammarCompiler, sequential  abort at ~100 grammars
    shared GrammarCompiler, max_threads=1   all 233 pass

`compile_grammar` parallelises internally, so "sequential" still used eight threads.
Pinning max_threads=1 removes the crash. It also changes what a genuinely malformed
grammar does: single-threaded it raises `RuntimeError`, which `dispatch_ebnf` catches
and turns into an InvalidGrammarObject, so one bad cell no longer takes down the run.

This only changes how many threads compile a grammar. Sampling, the grammar itself and
the tokens it admits are untouched, so runs before and after stay comparable.

    python scripts/sim/patch_sglang_grammar_threads.py            # apply
    python scripts/sim/patch_sglang_grammar_threads.py --check    # report only
"""
from __future__ import annotations

import argparse
import importlib.util
import io
from pathlib import Path

OLD = 'self.grammar_compiler = GrammarCompiler(tokenizer_info=tokenizer_info)'
NEW = ('        # 2026-09-21: 여러 스레드로 컴파일하면 xgrammar 가 earley_parser.cc:203 에서\n'
       '        # C++ abort 를 내고 서버 전체가 죽는다. 한 스레드로 묶으면 같은 문법이\n'
       '        # 잡을 수 있는 RuntimeError 가 되어 아래 dispatch_ebnf 가 처리한다.\n'
       '        # 생성 의미는 바뀌지 않는다 — 컴파일 병렬도만 바뀐다.\n'
       '        self.grammar_compiler = GrammarCompiler(tokenizer_info=tokenizer_info,\n'
       '                                                max_threads=1)')


def target():
    spec = importlib.util.find_spec('sglang.srt.constrained.xgrammar_backend')
    if spec is None or not spec.origin:
        raise SystemExit('sglang 을 찾을 수 없다 — 서버의 venv 로 실행할 것')
    return Path(spec.origin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()

    path = target()
    text = io.open(path, encoding='utf-8').read()
    if 'max_threads=1' in text:
        print('이미 적용됨:', path)
        return 0
    if OLD not in text:
        print('대상 줄을 찾지 못했다 — sglang 판이 다르다:', path)
        return 1
    if args.check:
        print('적용 가능:', path)
        return 0
    backup = path.with_suffix('.py.orig')
    if not backup.exists():
        io.open(backup, 'w', encoding='utf-8', newline='\n').write(text)
    io.open(path, 'w', encoding='utf-8', newline='\n').write(
        text.replace('        ' + OLD, NEW))
    print('적용:', path)
    print('원본 보관:', backup)
    print('서버를 다시 띄워야 반영된다')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

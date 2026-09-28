"""Exact text-only LG tokenizer checks before any required simulation dispatch."""
from functools import lru_cache
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import threading
try:
    from .execution_errors import PromptBudgetError
except ImportError:
    from execution_errors import PromptBudgetError

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'data/experiments/no_smoking_zone/tokenizer_manifest.json'
_LOCK = threading.Lock()


def tokenizer_manifest():
    return json.loads(MANIFEST.read_text(encoding='utf-8'))


def verify_tokenizer_files(folder):
    folder = Path(folder)
    manifest = tokenizer_manifest()
    for name, expected in manifest['files'].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Tokenizer differs from frozen LG revision: {name}')
    return manifest


@lru_cache(maxsize=2)
def _load(folder):
    verify_tokenizer_files(folder)
    from transformers import PreTrainedTokenizerFast
    folder = Path(folder)
    config = json.loads((folder / 'tokenizer_config.json').read_text(encoding='utf-8'))
    config.pop('tokenizer_class', None)
    config.pop('auto_map', None)
    result = PreTrainedTokenizerFast(tokenizer_file=str(folder / 'tokenizer.json'), **config)
    result.chat_template = (folder / 'chat_template.jinja').read_text(encoding='utf-8')
    return result


def load_tokenizer(folder):
    with _LOCK:
        return _load(str(Path(folder).resolve()))


def check_request_budget(request):
    if os.environ.get('SIM_PROMPT_TOKEN_GUARD') != 'required':
        return None
    folder = os.environ.get('SIM_TOKENIZER_PATH')
    if not folder:
        raise ValueError('SIM_TOKENIZER_PATH is required for verified prompt dispatch')
    manifest = tokenizer_manifest()
    if request['model'] != manifest['model']:
        raise ValueError('Tokenizer/model identity mismatch')
    tokenizer = load_tokenizer(folder)
    token_ids = tokenizer.apply_chat_template(
        request['messages'], tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=False,
    )
    # Some transformers releases return a BatchEncoding with two *fields*
    # (input_ids and attention_mask). len(BatchEncoding) is 2, not the number
    # of tokens, so never use len() until the actual IDs have been selected.
    if isinstance(token_ids, Mapping):
        token_ids = token_ids.get('input_ids')
    if (not isinstance(token_ids, (list, tuple)) or not token_ids
            or any(type(token_id) is not int for token_id in token_ids)):
        raise ValueError('Tokenizer did not return one exact token ID sequence')
    count = len(token_ids)
    output = request['max_tokens']
    limit = int(os.environ.get('SIM_MODEL_CONTEXT_LENGTH', '8192'))
    if type(output) is not int or output <= 0 or limit <= 128:
        raise ValueError('Invalid output/context token budget')
    if count + output + 128 > limit:
        raise PromptBudgetError(f'Prompt exceeds frozen context budget: input={count}, output={output}, limit={limit}')
    return {'input_tokens': count, 'reserved_output_tokens': output,
            'margin_tokens': 128, 'context_limit': limit}


def download_tokenizer(folder):
    """Download only the three pinned public tokenizer files, never model weights."""
    from huggingface_hub import hf_hub_download
    manifest = tokenizer_manifest()
    for name in manifest['files']:
        hf_hub_download(manifest['model'], filename=name, revision=manifest['revision'], local_dir=folder)
    verify_tokenizer_files(folder)
    return Path(folder)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', type=Path, required=True)
    args = parser.parse_args()
    download_tokenizer(args.download)
    print('Pinned LG text tokenizer downloaded and hashes verified.')

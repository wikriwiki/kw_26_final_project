"""A newly registered candidate can reuse contexts without altering old text."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.sim.prepare_reused_prompt_pilot import prepare
from scripts.sim.validate_prompt_v3 import digest
from scripts.sim.prompts import v51, v53


def test_new_candidate_requires_explicit_flag_and_preserves_contexts(tmp_path):
    old = json.loads((Path(__file__).resolve().parents[3]
                      / 'data/experiments/validation_v51_prodtemp_20260926.json').read_text(
                          encoding='utf-8'))
    source = tmp_path / 'source'
    source.mkdir()
    frozen = {'cells': [{'id': 'unchanged-cell'}],
              'systems': {'v51': v51.SYSTEM_PROMPT}}
    manifest = {'config': old, 'config_sha256': digest(old),
                'inputs_sha256': digest(frozen)}
    (source / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    (source / 'frozen_inputs.json').write_text(json.dumps(frozen), encoding='utf-8')
    new = copy.deepcopy(old)
    new['id'] = 'new-candidate'
    new['candidates'] = ['v51', 'v53']
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(new), encoding='utf-8')
    destination = tmp_path / 'out'
    with pytest.raises(ValueError, match='explicit'):
        prepare(source, config, destination)
    result = prepare(source, config, destination, allow_new_candidates=True)
    inputs = json.loads((destination / 'frozen_inputs.json').read_text(encoding='utf-8'))
    assert result['contexts'] == 1
    assert inputs['cells'] == frozen['cells']
    assert inputs['systems'] == {'v51': v51.SYSTEM_PROMPT, 'v53': v53.SYSTEM_PROMPT}

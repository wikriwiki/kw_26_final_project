"""Experimental verbatim evidence field: no generated factual justification.

Selecting an input span does not prove it supports the action. Semantic review
and the ordinary schedule checks remain necessary. This is constrained decoding,
not a claim that the unconstrained model independently avoided hallucinations.
"""
import re

ROUTINE = '오늘의 일상 선택'


def evidence_atoms(user):
    result = [ROUTINE]
    for line in user.splitlines():
        line = line.strip()
        if not line or line.startswith('##') or line.startswith('외출 anchor 허용값:'):
            continue
        if line.startswith('오늘 하루 계획을') or line.startswith('→'):
            continue
        for atom in re.split(r'(?<=[.!?])\s+', line):
            if atom and atom not in result:
                result.append(atom)
    return result


def constrain_evidence(schema, atoms):
    from copy import deepcopy
    result = deepcopy(schema)
    def visit(node):
        if isinstance(node, dict):
            if 'reasoning' in node.get('properties', {}):
                node['properties']['reasoning'] = {'type': 'string', 'enum': list(atoms)}
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(result)
    return result

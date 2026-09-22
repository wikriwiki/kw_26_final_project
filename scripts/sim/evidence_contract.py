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


def constrain_field(schema, field, atoms):
    from copy import deepcopy
    result = deepcopy(schema)
    def visit(node):
        if isinstance(node, dict):
            if field in node.get('properties', {}):
                node['properties'][field] = {'type': 'string', 'enum': list(atoms)}
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(result)
    return result


def constrain_evidence(schema, atoms):
    return constrain_field(schema, 'reasoning', atoms)


def constrain_trigger_evidence(schema, pools):
    """A causal label may cite only its explicitly provided factual channel.

    This enforces provenance, not semantic entailment or a policy-effect target.
    """
    from copy import deepcopy
    special = {'policy', 'appointment', 'rumor'}
    def visit(node):
        if isinstance(node, dict):
            if {'reasoning', 'trigger'} <= set(node.get('properties', {})):
                allowed = node['properties']['trigger']['enum']
                atoms = set(node['properties']['reasoning']['enum'])
                branches = []
                ordinary = [t for t in allowed if t not in special]
                if ordinary:
                    branch = deepcopy(node)
                    branch['properties']['trigger']['enum'] = ordinary
                    branches.append(branch)
                for trigger in sorted(set(allowed) & special):
                    evidence = pools.get(trigger)
                    if not evidence or not set(evidence) <= atoms:
                        raise ValueError('Missing or non-verbatim causal evidence pool: ' + trigger)
                    branch = deepcopy(node)
                    branch['properties']['trigger']['enum'] = [trigger]
                    branch['properties']['reasoning']['enum'] = list(evidence)
                    branches.append(branch)
                return {'anyOf': branches}
            return {key: visit(value) for key, value in node.items()}
        if isinstance(node, list):
            return [visit(value) for value in node]
        return node
    return visit(schema)

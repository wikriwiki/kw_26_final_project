import re

def canonical_evidence_ref(value, evidence_lines):
    """Repair only a zero-padded spelling of an existing, single input ref.

    The source line must exist. Multiple refs and invented numbers stay invalid;
    choosing one of them would change the model's evidential claim.
    """
    if not isinstance(value, str):
        return value
    candidate = value.strip()
    if candidate.startswith('[') and candidate.endswith(']'):
        candidate = candidate[1:-1].strip()
    if not re.fullmatch(r'E[0-9]{1,8}', candidate):
        return value
    if candidate in evidence_lines:
        return candidate
    canonical = f'E{int(candidate[1:]):04d}'
    return canonical if canonical in evidence_lines else value

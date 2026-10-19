"""Validate public decision explanations against text actually shown to the model.

A matching quotation proves access to a source span, not that an explanation is
true or that the span logically supports every claim. Preserve that distinction
in interviews and evaluation; never treat this as a hidden-thought transcript.
"""


def validate_stated_reason(record, user_text, *, reason_key='reasoning'):
    reason = record.get(reason_key)
    quote = record.get('evidence_quote')
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
        raise ValueError(f'{reason_key} must be a brief, nonempty stated explanation')
    if not isinstance(quote, str) or len(quote.strip()) < 4 or len(quote) > 1000:
        raise ValueError('evidence_quote must contain 4..1000 characters from the input')
    if quote not in user_text:
        raise ValueError('evidence_quote does not occur in the context shown for this decision')
    return {'quote_span_verified': True, 'semantic_support_verified': False,
            'explanation_kind': 'model_stated_rationale'}

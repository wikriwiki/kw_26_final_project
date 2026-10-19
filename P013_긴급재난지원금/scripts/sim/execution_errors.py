"""Separate permanent request errors from genuinely transient failures."""


class PromptBudgetError(ValueError):
    pass


def fatal_dispatch_error(exc):
    if isinstance(exc, PromptBudgetError):
        return True
    status = getattr(exc, 'status_code', None)
    return isinstance(status, int) and 400 <= status < 500 and status not in (408, 409, 429)


def transient_execution_error(exc):
    if fatal_dispatch_error(exc):
        return False
    retryable = getattr(exc, 'is_retryable', None)
    if callable(retryable) and retryable():
        return True
    return type(exc).__name__ in {'ServiceUnavailable', 'SessionExpired',
                                 'APIConnectionError', 'APITimeoutError', 'RateLimitError'}

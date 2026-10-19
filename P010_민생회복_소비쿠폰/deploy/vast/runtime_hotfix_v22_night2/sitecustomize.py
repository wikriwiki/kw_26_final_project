"""Keep the v22 State repair and allow bounded Night2 classification retries.

This module is loaded only by the resumed simulator through PYTHONPATH. It
does not alter the frozen v22 project or previously sealed agent results.
"""

import hashlib
import importlib.util
import os
from pathlib import Path


if os.environ.get("NO_SMOKING_V22_SKIP_PERSISTENCE_HOTFIX") == "1":
    previous = Path("/workspace/no-smoking-runtime-hotfix-v22-skip/sitecustomize.py")
    expected = "e6b2212e007c49889e13a117e22fcb8017d26231b75fba8b810e4afc1521a486"
    if hashlib.sha256(previous.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Existing skipped-State repair changed")
    spec = importlib.util.spec_from_file_location("_v22_state_repair", previous)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import night_intent_llm

    original_classify = night_intent_llm.classify_intent

    def classify_intent_with_retries(pair_key, data, max_retry=2):
        # A pair is written only after a validated response. This increases
        # attempts for rare malformed responses without changing the schema.
        return original_classify(pair_key, data, max_retry=max(5, max_retry))

    night_intent_llm.classify_intent = classify_intent_with_retries

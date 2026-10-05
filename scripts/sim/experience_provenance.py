"""Bounded source provenance and atomic local artifacts. Never reads secrets."""
import os
import hashlib
from functools import lru_cache
from pathlib import Path
import tempfile

from evidence_integrity import canonical, digest


@lru_cache(maxsize=1)
def source_fingerprint():
    root = Path(__file__).resolve().parent
    # Include simulator adapters and prompt modules, not only the entry point.
    # Input graph/model weights still require a separate deployment manifest.
    paths = sorted(root.rglob('*.py'))
    return digest({p.relative_to(root).as_posix(): p.read_text(encoding='utf-8') for p in paths})


def _settings():
    # All EXP_* settings can alter a citizen's state or decisions. A fixed
    # shortlist silently omitted income, the cashback base ratio and the
    # plan-to-total switch; a resumed run could then mix two experiments.
    # [2026-10-06] 결정에 닿지 않는 운영 설정은 뺀다: 시도 예산(EXP_AGENT_DAY_MAX_ATTEMPTS), 모델 서버 주소(LLM_BASE_URL).
    # 본런 도중 예산을 늘리거나 A100 직접↔GPU 풀 중계를 바꾸면 재개가 거부됐다. 같은 모델인지는 중계가 확인한다.
    settings = {k: v for k, v in os.environ.items()
                if k.startswith('EXP_') and k != 'EXP_AGENT_DAY_MAX_ATTEMPTS'}
    for key in ('LLM_MODE', 'SIM_ENVIRONMENT',
                'SIM_PROMPT_VARIANT', 'CONSUMPTION_MODEL',
                'POLICY_BACKTEST_DETERMINISTIC', 'POLICY_POI_SORT_BOOST',
                'SIM_ALLOW_STAGE2_FALLBACK', 'PYTHONHASHSEED',
                # doinggyu 엔진이 지문에 넣던 설정(인터뷰 근거·토큰 상한·문맥 길이)
                'SIM_INTERVIEW_EVIDENCE', 'SIM_PROMPT_TOKEN_GUARD', 'SIM_MODEL_CONTEXT_LENGTH'):
        settings[key] = os.environ.get(key)
    return settings


def execution_fingerprint():
    result = {'source': source_fingerprint(), 'settings': _settings()}
    from no_smoking_context import configured_context
    smoking_runtime = configured_context()
    if smoking_runtime:
        from prompt_budget import MANIFEST
        result['no_smoking'] = {'arm': smoking_runtime.arm,
                               'manifest_sha256': smoking_runtime.manifest_sha256,
                               'tokenizer_manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest()}
    return digest(result)


def paired_environment_fingerprint():
    """Pair two runs whose sole intended setting difference is environment ID."""
    settings = _settings()
    settings.pop('SIM_ENVIRONMENT', None)
    return digest({'source': source_fingerprint(), 'settings': settings})


def atomic_json(path, value):
    atomic_text(path, canonical(value))


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)

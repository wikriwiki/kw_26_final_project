#!/usr/bin/env bash
# One phase only. Post branches must restore separate copies of the verified Dec-2 dump.
set -euo pipefail
umask 077
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv-no-smoking/bin/python}"
: "${BUNDLE_DIR:?Set BUNDLE_DIR to a validated experiment bundle}"
: "${ARM:?Set ARM to off or on}"
: "${OUTPUT_DIR:?Set a new OUTPUT_DIR for this arm}"
[[ "$ARM" == off || "$ARM" == on ]] || { printf '%s\n' 'ARM must be off or on' >&2; exit 2; }
START_DATE="${START_DATE:-2017-11-19}"
DAYS="${DAYS:-28}"
WORKERS="${WORKERS:-4}"
PHASE="${PHASE:-standalone}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-3600}"
[[ "$PHASE" == standalone || "$PHASE" == shared_pre || "$PHASE" == post_branch ]] || { printf '%s\n' 'PHASE must be standalone, shared_pre or post_branch' >&2; exit 2; }
if [[ "$PHASE" == post_branch ]]; then
  : "${BRANCH_MANIFEST:?Set BRANCH_MANIFEST to the verified Dec-2 branch record}"
fi
for number in "$DAYS" "$WORKERS" "$RUN_TIMEOUT_SECONDS"; do
  [[ "$number" =~ ^[1-9][0-9]*$ ]] || { printf '%s\n' 'days/workers/timeout must be positive integers' >&2; exit 2; }
done
LLM_MODE="${LLM_MODE:-exaone_4_5}"
MODEL="${MODEL:-LGAI-EXAONE/EXAONE-4.5-33B-AWQ}"
[[ "$MODEL" == LGAI-EXAONE/EXAONE-4.5-* ]] || { printf '%s\n' 'Only LG EXAONE 4.5 models are allowed' >&2; exit 2; }
export LLM_MODE MODEL
export SGLANG_BASE_URL="http://127.0.0.1:${PORT:-8000}/v1"
export NO_SMOKING_SERVER_CONFIG="$ROOT/output/experiments/no_smoking_zone/runtime/server-config.json"
command=("$PYTHON" scripts/experiments/no_smoking_zone.py run
  --bundle "$BUNDLE_DIR" --arm "$ARM" --start "$START_DATE"
  --days "$DAYS" --out "$OUTPUT_DIR" --workers "$WORKERS" --phase "$PHASE")
if [[ "$PHASE" == post_branch ]]; then
  command+=(--branch-manifest "$BRANCH_MANIFEST")
fi
if [[ "${1:-}" == --dry-run ]]; then
  printf '%q ' timeout --signal=TERM --kill-after=30 "$RUN_TIMEOUT_SECONDS" "${command[@]}"
  printf '\n'
  exit 0
fi
: "${NEO4J_USER:?Set NEO4J_USER}"
: "${NO_SMOKING_OFF_NEO4J_URI:?Set the isolated policy-off database URI}"
: "${NO_SMOKING_ON_NEO4J_URI:?Set the isolated policy-on database URI}"
: "${NO_SMOKING_SNAPSHOT_SHA256:?Set the SHA256 of the verified clean Day 0 snapshot}"
# The runner requires distinct endpoints and validates the selected arm's graph
# and snapshot marker. The other arm is inspected when that arm is launched.
# Passwords may be NEO4J_PASSWORD or the arm-specific *_PASSWORD variables.
[[ ! -e "$OUTPUT_DIR" ]] || { printf '%s\n' 'OUTPUT_DIR exists; choose a new directory' >&2; exit 2; }
"$PYTHON" scripts/experiments/no_smoking_zone.py preflight --bundle "$BUNDLE_DIR"
if [[ "$PHASE" != standalone ]]; then
  : "${SIM_POST_DAY_BACKUP_HOOK:?Set the verified post-day backup hook path}"
  "$PYTHON" "$SIM_POST_DAY_BACKUP_HOOK" --probe
fi
# Health + actual tiny inference; a models response alone does not prove generation works.
"$PYTHON" - <<'PY'
import json, os, urllib.request
base = os.environ['SGLANG_BASE_URL']
with urllib.request.urlopen(base.removesuffix('/v1') + '/health', timeout=10):
    pass
with urllib.request.urlopen(base + '/models', timeout=10) as response:
    models = json.load(response)
model = os.environ['MODEL']
from scripts.sim.llm_client import get_spec
assert get_spec(os.environ['LLM_MODE']).hf_id == model, 'LLM_MODE and MODEL disagree'
assert model in {item['id'] for item in models['data']}, 'served model does not match MODEL'
from openai import OpenAI
client = OpenAI(base_url=base, api_key='EMPTY', timeout=60, max_retries=0)
answer = client.chat.completions.create(model=model, messages=[{'role':'user','content':'Reply OK.'}],
    max_tokens=16, temperature=0, extra_body={'chat_template_kwargs':{'enable_thinking':False}})
assert answer.choices and answer.choices[0].message.content, 'empty generation'
print('Model health and generation passed.')
PY
exec timeout --signal=TERM --kill-after=30 "$RUN_TIMEOUT_SECONDS" "${command[@]}"

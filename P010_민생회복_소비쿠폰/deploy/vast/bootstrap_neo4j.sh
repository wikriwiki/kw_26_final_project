#!/usr/bin/env bash
# Restore one verified clean baseline into two NEW local Neo4j homes.
# Never run with bash -x: credentials are inherited environment variables.
set +x
set -euo pipefail
umask 077
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv-no-smoking/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3
exec "$PYTHON" "$ROOT/deploy/vast/neo4j_pair.py" "$@"

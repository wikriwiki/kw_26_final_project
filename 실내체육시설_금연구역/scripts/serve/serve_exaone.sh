#!/usr/bin/env bash
# LG EXAONE entry using the existing SGLang environment.
# Override TP/PORT/VENV explicitly for a different GPU or server layout.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/serve_exaone45_sglang_a100x2.sh" "$@"

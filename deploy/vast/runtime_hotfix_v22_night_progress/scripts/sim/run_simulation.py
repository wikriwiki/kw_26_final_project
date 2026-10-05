"""Execute the original simulator with the reviewed terminal-night function."""
import sys
from pathlib import Path

PROJECT = Path('/workspace/no-smoking-project-v22-perf-final')
sys.path.insert(0, str(PROJECT / 'scripts/sim'))
sys.path.insert(0, str(PROJECT / 'scripts'))
import run_simulation as original
from _night_progress_runtime import replace_functions

replace_functions(Path(__file__).resolve().parents[2] / 'functions/run_simulation.py', original)
original.main()

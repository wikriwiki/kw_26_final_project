cd /data/repo_ab3w_20261011 && source /data/venv/bin/activate
python -m pytest -q tests/unit/report/test_score_p016_20261011.py 2>&1 | tail -2
B=/data/ab3w/p016_t150a
python scripts/report/score_p016.py --on-dossier $B/on/dossier.jsonl --off-dossier $B/off/dossier.jsonl --policy data/experiments/P016_ab3w_policy_20261011.json --start 2020-07-30 --end 2020-08-01 --json-out $B/score/p016_score_m3.json | grep -E "M3|판정"

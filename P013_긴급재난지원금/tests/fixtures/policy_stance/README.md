# Synthetic stance regression fixtures

These are hand-authored fictional responses, not GPU/model outputs, real human
attitudes, or labels inferred from the policy study. Every record and design has
`synthetic_fixture=true`.

`synthetic_demo_inputs.json` contains four OFF/pre training responses, two
distinct ON/post held-out responses, the frozen design, and literal independent
test assertions for the held-out responses. `synthetic_demo_expected.json`
records compact regression expectations and the input digest.

The deliberately low-confidence opposition response remains insufficient in the
analysis and counts as an error against the substantive held-out assertion.
The resulting 0.5 scores are software-test expectations, not empirical accuracy.
The two lexical topic groups test genuine unsupervised assignment without using
declared stance fields or demographic metadata columns.

Run `python -m pytest tests/unit/sim/test_policy_stance_analysis.py -q`.

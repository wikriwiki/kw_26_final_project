The tool frozen at `b7d3ac7` exited on its first canonical metric row with
`KeyError: run_id`, before reading or joining the recovered catalog to outcomes.
The sector exporter records run ID in `prompt_provenance.run_id`, not a
top-level `run_id`. The adapter now reads that documented field and requires
`provenance.experience_run_id == [run_id]`; receipt and metric checks are
unchanged. The plan, source gates, cohort, groups, formula, confidence interval,
and interpretation rules remain unchanged. No recovered-catalog group result
had been calculated when this correction was made.

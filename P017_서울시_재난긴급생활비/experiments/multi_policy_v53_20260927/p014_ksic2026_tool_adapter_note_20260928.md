The tool frozen at `b7d3ac7` exited on its first canonical metric row with
`KeyError: run_id`, before reading or joining the recovered catalog to outcomes.
The sector exporter records run ID in `prompt_provenance.run_id`, not a
top-level `run_id`. The adapter now reads that documented field and requires
`provenance.experience_run_id == [run_id]`; receipt and metric checks are
unchanged. The plan, source gates, cohort, groups, formula, confidence interval,
and interpretation rules remain unchanged. No recovered-catalog group result
had been calculated when this correction was made.

The next call passed the byte/SHA/NUL source gate but stopped at a catalog row
whose industry code was blank. A source-only full scan then found 534,978 rows,
zero column errors, zero blank or duplicate merchant IDs, and 199 blank industry
codes (0.0372%). This is classification missingness in the intact source, not
the corruption of the separate G: copy. Neither failed call produced a group
outcome.

The original plan did not require every source row to have an industry code.
The adapter now marks blank or malformed codes unresolved and counts any
corresponding receipt as **unmatched** for both count and won coverage. The
previous minimum 99% thresholds remain unchanged. A normal non-target sector
code such as `I56111` is resolved outside the two target groups; its prefix is
not stripped or renamed. Target normalization still strips only one `G` from
exact `G` + five digits, or accepts exactly five digits unchanged. The catalog
SHA, bytes, row count, zero duplicate IDs, cohort, groups, formula, bootstrap,
and sparse/interpretation gates remain unchanged. This correction was frozen
before the first recovered-source group outcome calculation.

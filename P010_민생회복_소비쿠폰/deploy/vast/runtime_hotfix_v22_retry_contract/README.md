# v22 retry contract, effective 2017-11-24 (Day6)

This extension keeps the frozen v22 project and existing run/graph identities.
It loads the previous grounding layer, then chooses old or corrected functions
by **simulation date**. Existing Day1–Day5 decisions are not recomputed. A new
Python process is needed once to load the day gate; the day transition itself
needs no process replacement. The Vast instance and model stay running.

## Corrected behavior

- Stage1 no longer breaks after an identical second response and reports six
  calls. All allowed calls have distinct numbered retry feedback; failures and
  timings reflect actual calls.
- Retry feedback contains one error explanation and the event index/time or
  pick order. A rejected middle item remains visible in a truncated quotation.
- Stage2's system/user instructions and output schema request the same remaining
  orders. An exactly identical replay of a saved pick is ignored and counted;
  conflicting repeats, duplicate orders and invented candidates stay errors.
- Stage1 checks time syntax and appointment peer membership in Python, including
  when the server enforces JSON syntax only. Fifteen-minute gaps remain allowed.
- Metrics distinguish requested temperature from effective temperature. The
  existing deterministic temperature, model, workers and grammar are retained.

`manifest.json` pins the function copies and runtime module. Its SHA256 is
`7f0c6f2d4ed7b956b1371898b3736db511150625ae71db1f6044819520166b99`.
New sealed metrics include `retry_contract_sha256`; successful decision timings
also carry this field. The original source fingerprint is preserved separately.

## Validation and limits

Offline tests cover call accounting, feedback location, remaining-order scope,
conflicting/exact replay, peer/time validation, temperature metadata and the day
gate. `verify_runtime.py` exercises the actual installed patch stack with stubs.
`real_model_probe.py` sends one real request after a synthetic partial pick. It
writes no graph/experiment results and is not a production error-rate benchmark.

Stage1 reuse across whole-agent retries is **not enabled**: its model I/O ledger
must retain correct provenance before such a cache can be used. Model sampling,
grammar backend and concurrency optimizations are likewise not part of this fix.

## Recorded one-time handoff

Do not rerun `switch_once.py` or `continue_handoff.py` during routine monitoring.
They were prepared for the specific Day5 handoff and have process/date guards.
The planned maintenance route verifies all sealed file/graph outboxes, preserves
completed days, makes a fresh full-graph/full-run Drive backup, independently
checks remote hashes, then invokes the new launcher. Only missing file copies
of already committed DB outboxes may be recovered; no graph baseline is loaded.

The first handoff exposed a process-group coupling: Neo4j last started by the
backup hook shared the simulator's process group. Terminating that group also
initiated a clean DB shutdown. The fallback tried to restart before shutdown
finished. `ensure_database` now waits for it to finish and starts the **same
store** in a separate session before checking receipts. The continuation used
that store. The initial maintenance helper also needed the project's `scripts`
directory on `sys.path`; that import-path error was fixed before backup began.
Both failures are retained in the handoff journal, not erased as successes.

The Drive uploader then encountered the shared OAuth project's quota error.
`resume_external_snapshot.py` completed this specific handoff after the fresh
graph and full run archive had instead been copied off the instance to the
user's computer and fully SHA256-verified. It left only the existing network
upload running and did not mark Drive verified. Day5 resumed from 918 sealed
agent-days at 2026-09-28 17:28:47 UTC. The uploader finished at 17:28:51 UTC and
an independent Drive hash/commit check passed at 17:34:54 UTC. The quota issue
remains intermittent; these are separate pieces of evidence, not a permanent
credential fix. The external-snapshot helper is also a one-time incident tool,
not a routine monitor command.

Use `docs/HANDOFF_NO_SMOKING_ZONE.md` and the ignored
`deploy/vast/local/ACTIVE_DEPLOYMENT.json` for live PIDs, backup timestamps and
current status. A deployed day gate is distinct from observed Day6 model output.

# Paired baseline restoration amendment — registered before OFF completes

At approximately 2026-09-27 17:27 KST, while the OFF arm was still running and before any ON result existed, source review found that `scripts/neo4j_load/97_reset_run_artifacts.py` resets `KNOWS_POI` visit attributes but retains relationships newly created by `plan_writer.py` during the preceding arm. An ON arm started after only that reset could see OFF-learned places. `Agent.execution_lock` also persists after reset, although it is not prompt content.

The experiment's policy, cohort, income map, prompt, dates, outcomes and gates remain as preregistered. The stricter pairing procedure is:

1. Let OFF finish all five days and export the full audited daily ledger. Pause the shell runner during export so it cannot begin ON automatically.
2. Verify and copy the OFF ledger, manifest, cohort, metrics, and run summary from A100 to both G: and C: by SHA256 before modifying the graph.
3. Stop Neo4j, restore the pre-pilot `neo4j.dump` whose hash was verified on server/G:/C:, restart Neo4j, and run the same reset plus Day0 seed. Load P013 only after reset and verify policy DB wiring.
4. Run ON using the same frozen code checkout, prompt, model, 80 citizens, income map and calendar. Audit and export it independently, then compute the paired effect.

This change prevents prior-arm graph contamination. It does not justify external effect-size comparison; the 3-day post-policy proxy remains a technical pilot. If the dump restore or audit fails, do not score the pair.

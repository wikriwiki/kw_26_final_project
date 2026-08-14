# GPU server reproducibility snapshot (2026-08-14)

This directory contains a `kw_26_final_project`-only snapshot collected from
`/home/ubuntu/data` before the temporary GPU server was reclaimed. No files
from the separate `baram_a100` project are included.

## Contents

- `archives/kw_repro_20260814.tar.gz`: exact compressed copy of the selected
  server files (source code, run scripts, experiment summaries, failure logs,
  statistics, and lightweight result metadata).
- `archives/kw_env_20260814.tar.gz`: exact environment-information archive.
- `server_data/exp001_repo/scripts/`: exact server-side source snapshot.
- `server_data/exp001/`: run scripts and lightweight experiment records.
- `server_data/exp_R*/`: compact experiment records. Repetitive per-agent
  metrics and timing files remain recoverable from the archive but are ignored
  in the expanded Git view.
- `environment/`: Python package freezes and hardware/OS/runtime information.

## Archive integrity

Verify the downloaded archives from this directory:

```bash
sha256sum -c CHECKSUMS.sha256
```

Extract the reproducibility archive elsewhere:

```bash
mkdir kw_gpu_snapshot
tar -xzf archives/kw_repro_20260814.tar.gz -C kw_gpu_snapshot
```

## Selection and exclusions

The archive intentionally excludes replaceable or GitHub-incompatible runtime
material: Python virtual environments, the 22 GB model cache, the Neo4j 5.26.0
distribution, Neo4j database dumps, large event streams, and files at least
50 MiB. Their server-side paths and sizes are recorded in
`environment/large-files.tsv` and `environment/directory-sizes.txt`.

Credential-bearing service configuration, Jupyter configuration, SSH keys,
and shell profiles were not collected. The server source tree itself had no
`.git` directory, so this snapshot preserves the deployed files independently
of Git metadata.

Captured on 2026-08-14 (Asia/Seoul) from the two-A100 GPU server.

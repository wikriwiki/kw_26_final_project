#!/usr/bin/env bash
# Keep the Neo4j restart's tmux process group alive after the ON dump.
set -u
OUT=/data/p013_v53_pilot_20260927
bash "$OUT/dump_on_graph.sh" > "$OUT/on_graph_dump.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$OUT/on_graph_dump.exitcode"
sleep infinity

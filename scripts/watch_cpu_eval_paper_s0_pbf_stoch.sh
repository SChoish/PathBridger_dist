#!/usr/bin/env bash
# Thin wrapper: same CPU NT watcher as pathwt, parameterized by RUN_GROUP.
# Separate script name so pathwt keepalive does not TERM this process.
set +e
set -uo pipefail
REPO="${REPO:-/home/ext_csv/PathBridger}"
# Caller must set RUN_GROUP (paper_s0_pbf_lrg / paper_s0_pbf_jflow).
exec bash "$REPO/scripts/watch_cpu_eval_paper_s0_pbf_pathwt.sh" "$@"

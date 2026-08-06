#!/usr/bin/env bash
# Thin wrapper: GPU large-N watcher (N>=8) with RUN_GROUP from env.
# Separate argv from pathwt so process lists stay distinct.
set +e
set -uo pipefail
REPO="${REPO:-/home/ext_csv/PathBridger_dist}"
exec bash "$REPO/scripts/watch_gpu_eval_large_n_paper_s0_pbf_pathwt.sh" "$@"

#!/usr/bin/env bash
# Wait until the jflow GPU M-sweep is fully idle, then start the CPU M-sweep
# for any remaining Best cells (M=1/8/16). Shares DONE/LOCK markers.
set +e
set -uo pipefail

REPO="${REPO:-/home/ext_csv/PathBridger_dist}"
GPU_LOG="${GPU_LOG:-$REPO/nohup_logs/paper_s0_pbf_jflow_m_sweep_gpu}"
CPU_LOG="${CPU_LOG:-$REPO/nohup_logs/paper_s0_pbf_jflow_m_sweep}"
POLL_SEC="${POLL_SEC:-30}"
PREFIX_MS="${PREFIX_MS:-1 8 16}"
MAX="${MAX:-3}"
CPU_BASE="${CPU_BASE:-64}"
EVAL_GRID="${EVAL_GRID:-best}"
EXPECTED_RUNS="${EXPECTED_RUNS:-8}"

mkdir -p "$CPU_LOG" "$GPU_LOG"
cd "$REPO" || exit 1

ts() { TZ=Asia/Seoul date '+%F %T %Z'; }
log() { printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$CPU_LOG/handoff.log"; }

gpu_busy() {
  # live gpu eval workers or live gpu sweep master
  pgrep -af 'scripts/gpu_eval_pbf_best_m.py' >/dev/null 2>&1 && return 0
  if [[ -f "$GPU_LOG/sweep.pid" ]]; then
    local pid
    pid="$(cat "$GPU_LOG/sweep.pid" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  fi
  pgrep -af 'scripts/sweep_jflow_prefix_m_gpu_eval.sh' >/dev/null 2>&1 && return 0
  return 1
}

pending_count() {
  /home/ext_csv/miniconda3/envs/offrl/bin/python - <<'PY'
from pathlib import Path
root = Path('/home/ext_csv/PathBridger_dist/exp/pathbridger/paper_s0_pbf_jflow')
n = 0
for d in root.iterdir() if root.is_dir() else []:
    if not d.is_dir() or d.name.startswith('_'):
        continue
    for st in (800000, 900000, 1000000):
        if not (d / 'checkpoints' / f'params_{st}.pkl').exists():
            continue
        for m in (1, 8, 16):
            if not (d / 'cpu_eval' / f'step_{st}_m{m}.DONE').exists():
                n += 1
print(n)
PY
}

log "WAIT GPU→CPU handoff start (POLL=${POLL_SEC}s)"
echo $$ >"$CPU_LOG/handoff.pid"

while gpu_busy; do
  if [[ -f "$CPU_LOG/HANDOFF_STOP" ]]; then
    log "HANDOFF_STOP — abort wait"
    rm -f "$CPU_LOG/handoff.pid"
    exit 0
  fi
  log "gpu still busy; pending=$(pending_count)"
  sleep "$POLL_SEC"
done

pend="$(pending_count)"
log "GPU idle; pending=${pend}"
if [[ "${pend}" == "0" ]]; then
  log "nothing left for CPU — handoff complete"
  rm -f "$CPU_LOG/handoff.pid"
  exit 0
fi

# clear stale CPU STOP / dead pid
rm -f "$CPU_LOG/STOP"
if [[ -f "$CPU_LOG/sweep.pid" ]]; then
  opid="$(cat "$CPU_LOG/sweep.pid" 2>/dev/null || true)"
  if [[ -n "${opid:-}" ]] && kill -0 "$opid" 2>/dev/null; then
    log "CPU sweep already live pid=${opid} — not double-launching"
    rm -f "$CPU_LOG/handoff.pid"
    exit 0
  fi
  rm -f "$CPU_LOG/sweep.pid"
fi

# clear dead LOCKs
find "$REPO/exp/pathbridger/paper_s0_pbf_jflow" -name 'step_*_m*.LOCK' -type f 2>/dev/null | while read -r lock; do
  pid="$(cat "$lock" 2>/dev/null || true)"
  if [[ -n "${pid:-}" ]] && ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$lock"
  fi
done

log "LAUNCH CPU M-sweep PREFIX_MS=${PREFIX_MS} MAX=${MAX} CPU_BASE=${CPU_BASE}"
PREFIX_MS="${PREFIX_MS}" MAX="${MAX}" CPU_BASE="${CPU_BASE}" \
  EVAL_GRID="${EVAL_GRID}" EXPECTED_RUNS="${EXPECTED_RUNS}" \
  nohup bash scripts/sweep_jflow_prefix_m_eval.sh >>"$CPU_LOG/nohup_sweep.out" 2>&1 &
echo $! >"$CPU_LOG/boot.pid"
log "CPU sweep boot pid=$!"
rm -f "$CPU_LOG/handoff.pid"
log "HANDOFF_DONE"

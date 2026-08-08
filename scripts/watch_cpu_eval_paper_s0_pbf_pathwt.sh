#!/usr/bin/env bash
# CPU eval watcher for paper_s0_pbf_pathwt / stoch groups (RUN_GROUP).
# Default EVAL_GRID=best: paper Best-(N,T) only @800/900/1M. Set EVAL_GRID=full
# for the legacy 19-cell NT sweep. MAX concurrent CPU jobs.
#
# Hardened: never `set -e`; skip `_quarantine_*` / missing ckpts; per-step LOCK
# so a restarted watcher does not double-launch. GPU is never touched
# (CUDA_VISIBLE_DEVICES= + JAX_PLATFORMS=cpu).
set +e
set -uo pipefail

REPO="${REPO:-/home/ext_csv/PathBridger_dist}"
PY="${PY:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
RUN_GROUP="${RUN_GROUP:-paper_s0_pbf_pathwt}"
EXP_ROOT="${EXP_ROOT:-$REPO/exp/pathbridger/${RUN_GROUP}}"
LOGROOT="${LOGROOT:-$REPO/nohup_logs/${RUN_GROUP}_cpu_eval}"
STEPS=(800000 900000 1000000)
MAX="${MAX:-4}"
EPISODES="${EPISODES:-50}"
POLL_SEC="${POLL_SEC:-30}"
CPU_BASE="${CPU_BASE:-32}"
CPU_STRIDE="${CPU_STRIDE:-8}"
AUTO_EXIT="${AUTO_EXIT:-0}"
EXPECTED_RUNS="${EXPECTED_RUNS:-8}"
# best = paper Best-(N,T) only; full = 19-cell NT sweep
EVAL_GRID="${EVAL_GRID:-best}"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export EIGEN_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false ${XLA_FLAGS:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=
export JAX_PLATFORMS=cpu
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOGROOT"
cd "$REPO" || exit 1

ts() { TZ=Asia/Seoul date '+%F %T %Z'; }
log() { printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$LOGROOT/watcher.log"; }

declare -a PIDS=()
declare -a KEYS=()
declare -a CPU_SLOTS=()

reap_one() {
  local i pid key rc lock
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      key="${KEYS[$i]}"
      wait "$pid" 2>/dev/null && rc=0 || rc=$?
      log "DONE key=${key} pid=${pid} rc=${rc}"
      if [[ "$key" == *_step* ]]; then
        local rname="${key%_step*}"
        local step="${key##*_step}"
        lock="${EXP_ROOT}/${rname}/cpu_eval/step_${step}.LOCK"
        if [[ -f "$lock" ]] && [[ "$(cat "$lock" 2>/dev/null)" == "$pid" ]]; then
          rm -f "$lock"
        fi
      fi
      unset 'PIDS[i]' 'KEYS[i]' 'CPU_SLOTS[i]'
      PIDS=("${PIDS[@]}")
      KEYS=("${KEYS[@]}")
      CPU_SLOTS=("${CPU_SLOTS[@]}")
      return 0
    fi
  done
  return 1
}

next_cpu_slot() {
  local used s
  for ((s=0; s<MAX; s++)); do
    used=0
    for u in "${CPU_SLOTS[@]:-}"; do
      [[ "$u" == "$s" ]] && used=1 && break
    done
    if (( used == 0 )); then
      echo "$s"
      return 0
    fi
  done
  return 1
}

lock_held() {
  local lock="$1"
  local pid
  [[ -f "$lock" ]] || return 1
  pid="$(cat "$lock" 2>/dev/null || true)"
  [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null
}

launch_eval() {
  local run_dir="$1" step="$2"
  local name out donef lock slot cpu0 cpu1
  name="$(basename "$run_dir")"
  if [[ "$name" == _* || ! -d "$run_dir" || ! -f "$run_dir/checkpoints/params_${step}.pkl" ]]; then
    log "SKIP missing/quarantined ${name} step=${step}"
    return 0
  fi
  mkdir -p "${run_dir}/cpu_eval"
  donef="${run_dir}/cpu_eval/step_${step}.DONE"
  lock="${run_dir}/cpu_eval/step_${step}.LOCK"
  if [[ -f "$donef" ]]; then
    log "SKIP already done ${name} step=${step}"
    return 0
  fi
  if lock_held "$lock"; then
    log "SKIP lock held ${name} step=${step} pid=$(cat "$lock")"
    return 0
  fi
  rm -f "$lock" 2>/dev/null || true

  while (( ${#PIDS[@]} >= MAX )); do
    if ! reap_one; then sleep 5; fi
  done
  slot="$(next_cpu_slot)" || slot=0
  cpu0=$((CPU_BASE + slot * CPU_STRIDE))
  cpu1=$((cpu0 + CPU_STRIDE - 1))
  out="$LOGROOT/${name}_step${step}.out"
  log "LAUNCH ${name} step=${step} grid=${EVAL_GRID} cpus=${cpu0}-${cpu1} episodes=${EPISODES}"
  (
    set +e
    taskset -c "${cpu0}-${cpu1}" \
      "$PY" -u scripts/cpu_eval_pbf_pathwt_ckpt.py \
        --run-dir="${run_dir}" \
        --checkpoint-step="${step}" \
        --episodes="${EPISODES}" \
        --seed=0 \
        --grid="${EVAL_GRID}" \
      >"$out" 2>&1
    exit $?
  ) &
  local child=$!
  echo "$child" >"$lock"
  PIDS+=("$child")
  KEYS+=("${name}_step${step}")
  CPU_SLOTS+=("$slot")
}

list_pending() {
  "$PY" - "$EXP_ROOT" <<'PY' || true
from pathlib import Path
import os, sys
root = Path(sys.argv[1])
steps = (800000, 900000, 1000000)
if not root.is_dir():
    raise SystemExit
for d in sorted(root.iterdir()):
    if not d.is_dir() or d.name.startswith('_') or not (d / 'flags.json').exists():
        continue
    if not (d / 'checkpoints' / 'params_800000.pkl').is_file():
        continue
    ck = d / 'checkpoints'
    for step in steps:
        if not (ck / f'params_{step}.pkl').is_file():
            continue
        if (d / 'cpu_eval' / f'step_{step}.DONE').is_file():
            continue
        lock = d / 'cpu_eval' / f'step_{step}.LOCK'
        if lock.is_file():
            try:
                pid = int(lock.read_text().strip())
            except Exception:
                pid = -1
            if pid > 0:
                try:
                    os.kill(pid, 0)
                    continue  # live worker
                except OSError:
                    pass
        print(f'{d}|{step}')
PY
}

log "START watcher EXP_ROOT=${EXP_ROOT} MAX=${MAX} EPISODES=${EPISODES} grid=${EVAL_GRID} AUTO_EXIT=${AUTO_EXIT}"

idle_rounds=0
while true; do
  while reap_one; do :; done
  mapfile -t PENDING < <(list_pending)
  if (( ${#PENDING[@]} == 0 )); then
    idle_rounds=$((idle_rounds + 1))
    if (( idle_rounds % 20 == 1 )); then
      log "idle pending=0 live=${#PIDS[@]} (poll ${POLL_SEC}s)"
    fi
  else
    idle_rounds=0
    for spec in "${PENDING[@]}"; do
      [[ -z "${spec:-}" ]] && continue
      name_step="$(basename "${spec%%|*}")_step${spec##*|}"
      skip=0
      for k in "${KEYS[@]:-}"; do
        if [[ "$k" == "$name_step" ]]; then skip=1; break; fi
      done
      if (( skip )); then continue; fi
      IFS='|' read -r run_dir step <<<"$spec"
      launch_eval "$run_dir" "$step" || log "WARN launch_eval failed ${spec}"
    done
  fi

  if [[ "${AUTO_EXIT}" == "1" ]]; then
    remaining="$("$PY" - "$EXP_ROOT" "$EXPECTED_RUNS" <<'PY' || echo '0/0|0'
from pathlib import Path
import sys
root = Path(sys.argv[1])
expected = int(sys.argv[2])
steps = (800000, 900000, 1000000)
need = expected * len(steps)
have = 0
runs = [
    d for d in root.iterdir()
    if d.is_dir() and not d.name.startswith('_') and (d / 'flags.json').exists()
] if root.is_dir() else []
for d in runs:
    for step in steps:
        if (d / 'cpu_eval' / f'step_{step}.DONE').is_file():
            have += 1
print(f'{have}/{need}|{len(runs)}')
PY
)"
    IFS='|' read -r prog nruns <<<"${remaining:-0/0|0}"
    nruns="${nruns:-0}"
    if [[ "${nruns}" =~ ^[0-9]+$ ]] \
      && (( nruns >= EXPECTED_RUNS )) \
      && [[ "${prog}" == "$((EXPECTED_RUNS * 3))/$((EXPECTED_RUNS * 3))" ]] \
      && (( ${#PIDS[@]} == 0 )) \
      && [[ -f "${REPO}/nohup_logs/${RUN_GROUP}/QUEUE_DONE" ]]; then
      log "ALL_EVAL_COMPLETE ${prog} runs=${nruns}"
      touch "$LOGROOT/EVAL_QUEUE_DONE"
      break
    fi
  fi
  sleep "$POLL_SEC" || sleep 30
done

log "WATCHER_EXIT"

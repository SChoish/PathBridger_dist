#!/usr/bin/env bash
# GPU sidecar for large-N NT cells (N>=8) of paper_s0_pbf_pathwt.
# Fills N in {8,16,32} x T whenever those cells are missing (parallel with CPU).
# CPU still evaluates low-N first within each of its jobs; GPU does not wait
# for small-N completion so all GPU slots stay busy.
# Both write the same eval_results/*.json; existing files are skipped.
#
# Default: 2 GPUs × 2 slots = 4 concurrent large-N step jobs.
# Prefer launching AFTER train finishes so GPUs are free (see
# wait_train_then_gpu_eval_large_n_paper_s0_pbf_pathwt.sh).
set +e
set -uo pipefail

REPO="${REPO:-/home/ext_csv/PathBridger}"
PY="${PY:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
RUN_GROUP="${RUN_GROUP:-paper_s0_pbf_pathwt}"
EXP_ROOT="${EXP_ROOT:-$REPO/exp/pathbridger/${RUN_GROUP}}"
LOGROOT="${LOGROOT:-$REPO/nohup_logs/${RUN_GROUP}_gpu_eval_large_n}"
MIN_N="${MIN_N:-8}"
read -r -a GPUS <<< "${GPU_LIST:-0 1}"
MAX_PER_GPU="${MAX_PER_GPU:-2}"
EPISODES="${EPISODES:-50}"
POLL_SEC="${POLL_SEC:-20}"
CPU_BASE="${CPU_BASE:-96}"
CPU_STRIDE="${CPU_STRIDE:-8}"
AUTO_EXIT="${AUTO_EXIT:-1}"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export EIGEN_NUM_THREADS=1
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1 --xla_gpu_autotune_level=0 ${XLA_FLAGS:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOGROOT"
cd "$REPO" || exit 1

ts() { TZ=Asia/Seoul date '+%F %T %Z'; }
log() { printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$LOGROOT/watcher.log"; }

total_slots=$((${#GPUS[@]} * MAX_PER_GPU))
declare -a PIDS=()
declare -a KEYS=()
declare -a SLOT_USED=()

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
        lock="${EXP_ROOT}/${rname}/gpu_eval/large_n_step_${step}.LOCK"
        if [[ -f "$lock" ]] && [[ "$(cat "$lock" 2>/dev/null)" == "$pid" ]]; then
          rm -f "$lock"
        fi
      fi
      unset 'PIDS[i]' 'KEYS[i]' 'SLOT_USED[i]'
      PIDS=("${PIDS[@]}")
      KEYS=("${KEYS[@]}")
      SLOT_USED=("${SLOT_USED[@]}")
      return 0
    fi
  done
  return 1
}

next_slot() {
  local s used
  for ((s=0; s<total_slots; s++)); do
    used=0
    for u in "${SLOT_USED[@]:-}"; do
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
  local lock="$1" pid
  [[ -f "$lock" ]] || return 1
  pid="$(cat "$lock" 2>/dev/null || true)"
  [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null
}

launch_eval() {
  local run_dir="$1" step="$2"
  local name out donef lock slot gpu_i gpu cpu0 cpu1
  name="$(basename "$run_dir")"
  if [[ "$name" == _* || ! -d "$run_dir" || ! -f "$run_dir/checkpoints/params_${step}.pkl" ]]; then
    log "SKIP missing/quarantined ${name} step=${step}"
    return 0
  fi
  mkdir -p "${run_dir}/gpu_eval" "${run_dir}/cpu_eval"
  donef="${run_dir}/cpu_eval/step_${step}.DONE"
  lock="${run_dir}/gpu_eval/large_n_step_${step}.LOCK"
  if [[ -f "$donef" ]]; then
    log "SKIP already done ${name} step=${step}"
    return 0
  fi
  if lock_held "$lock"; then
    log "SKIP gpu lock held ${name} step=${step} pid=$(cat "$lock")"
    return 0
  fi
  if "$PY" - "$run_dir" "$step" "$MIN_N" <<'PY'
import sys
from pathlib import Path
run, step, min_n = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
er = run / 'eval_results'
# Match gpu_eval_pbf_pathwt_large_n.FLOW_NT_GRID filtered by min_n.
grid = [(1, 0.0)] + [
    (n, t) for n in (1, 2, 4, 8, 16, 32) for t in (0.25, 0.5, 1.0)
]
grid = [(n, t) for n, t in grid if int(n) >= int(min_n)]
def tok(t):
    return str(int(t)) if float(t) == int(t) else str(t).replace('.', 'p')
missing = [1 for n, t in grid if not (er / f'epoch{step}_t{tok(t)}_n{n}.json').is_file()]
raise SystemExit(0 if missing else 1)
PY
  then
    :
  else
    log "SKIP NT>=${MIN_N} complete ${name} step=${step}"
    return 0
  fi

  while (( ${#PIDS[@]} >= total_slots )); do
    if ! reap_one; then sleep 5; fi
  done
  slot="$(next_slot)" || slot=0
  gpu_i=$((slot / MAX_PER_GPU))
  gpu="${GPUS[$gpu_i]}"
  cpu0=$((CPU_BASE + slot * CPU_STRIDE))
  cpu1=$((cpu0 + CPU_STRIDE - 1))
  out="$LOGROOT/${name}_step${step}.out"
  rm -f "$lock" 2>/dev/null || true
  log "LAUNCH ${name} step=${step} NT>=${MIN_N} gpu=${gpu} cpus=${cpu0}-${cpu1}"
  # Launch python directly (no subshell) so LOCK pid == live eval worker.
  CUDA_VISIBLE_DEVICES="${gpu}" \
    nohup taskset -c "${cpu0}-${cpu1}" \
    "$PY" -u scripts/gpu_eval_pbf_pathwt_large_n.py \
      --run-dir="${run_dir}" \
      --checkpoint-step="${step}" \
      --episodes="${EPISODES}" \
      --seed=0 \
      --gpu="${gpu}" \
      --min-n="${MIN_N}" \
    >"$out" 2>&1 &
  local child=$!
  echo "$child" >"$lock"
  PIDS+=("$child")
  KEYS+=("${name}_step${step}")
  SLOT_USED+=("$slot")
}

list_pending() {
  "$PY" - "$EXP_ROOT" "$MIN_N" <<'PY' || true
from pathlib import Path
import os, sys
root = Path(sys.argv[1])
min_n = int(sys.argv[2])
steps = (800000, 900000, 1000000)
grid = [(1, 0.0)] + [
    (n, t) for n in (1, 2, 4, 8, 16, 32) for t in (0.25, 0.5, 1.0)
]
grid = [(n, t) for n, t in grid if int(n) >= int(min_n)]

def tok(t):
    return str(int(t)) if float(t) == int(t) else str(t).replace('.', 'p')

if not root.is_dir():
    raise SystemExit
for d in sorted(root.iterdir()):
    if not d.is_dir() or d.name.startswith('_') or not (d / 'flags.json').exists():
        continue
    ck = d / 'checkpoints'
    er = d / 'eval_results'
    for step in steps:
        if not (ck / f'params_{step}.pkl').is_file():
            continue
        if (d / 'cpu_eval' / f'step_{step}.DONE').is_file():
            continue
        lock = d / 'gpu_eval' / f'large_n_step_{step}.LOCK'
        if lock.is_file():
            try:
                pid = int(lock.read_text().strip())
            except Exception:
                pid = -1
            if pid > 0:
                try:
                    os.kill(pid, 0)
                    # Only skip if lock points at a live gpu_eval worker.
                    # Stale locks from watcher bash PIDs must not block the queue.
                    cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\\0', b' ').decode('utf-8', 'ignore')
                    if 'gpu_eval_pbf_pathwt_large_n' in cmdline:
                        continue
                except OSError:
                    pass
        missing = [
            1 for n, t in grid
            if not (er / f'epoch{step}_t{tok(t)}_n{n}.json').is_file()
        ]
        if missing:
            print(f'{d}|{step}')
PY
}

echo "$$" >"$LOGROOT/watcher.pid"
log "START gpu large-N watcher EXP_ROOT=${EXP_ROOT} GPUS=${GPUS[*]} MAX_PER_GPU=${MAX_PER_GPU} MIN_N=${MIN_N} AUTO_EXIT=${AUTO_EXIT}"

idle_rounds=0
while true; do
  while reap_one; do :; done
  mapfile -t PENDING < <(list_pending)
  if (( ${#PENDING[@]} == 0 )); then
    idle_rounds=$((idle_rounds + 1))
    if (( idle_rounds % 15 == 1 )); then
      log "idle pending=0 live=${#PIDS[@]} (poll ${POLL_SEC}s)"
    fi
    if [[ "${AUTO_EXIT}" == "1" ]] && (( ${#PIDS[@]} == 0 )) && (( idle_rounds >= 3 )); then
      log "AUTO_EXIT: no pending large-N work"
      break
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
  sleep "$POLL_SEC"
done

log "EXIT gpu large-N watcher"
touch "$LOGROOT/GPU_LARGE_N_DONE"

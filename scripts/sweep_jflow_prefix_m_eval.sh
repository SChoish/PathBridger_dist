#!/usr/bin/env bash
# Best-NT CPU eval sweep over prefix-sample counts M for paper_s0_pbf_jflow.
# Writes epoch*_m{M}.json (does not overwrite legacy M=4 cells without _m).
# Default M list: 1 8 16. Re-polls until EXPECTED_RUNS have 800/900/1M ckpts
# or STOP is present.
set +e
set -uo pipefail

REPO="${REPO:-/home/ext_csv/PathBridger}"
PY="${PY:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
RUN_GROUP="${RUN_GROUP:-paper_s0_pbf_jflow}"
EXP_ROOT="${EXP_ROOT:-$REPO/exp/pathbridger/${RUN_GROUP}}"
LOGROOT="${LOGROOT:-$REPO/nohup_logs/${RUN_GROUP}_m_sweep}"
STEPS=(800000 900000 1000000)
# shellcheck disable=SC2206
MS=(${PREFIX_MS:-1 8 16})
MAX="${MAX:-3}"
EPISODES="${EPISODES:-50}"
POLL_SEC="${POLL_SEC:-45}"
CPU_BASE="${CPU_BASE:-64}"
CPU_STRIDE="${CPU_STRIDE:-8}"
EXPECTED_RUNS="${EXPECTED_RUNS:-8}"
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
log() { printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$LOGROOT/sweep.log"; }

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
      if [[ "$key" == *_m* ]]; then
        local rname="${key%%_step*}"
        local rest="${key#*_step}"
        local step="${rest%%_m*}"
        local m="${rest##*_m}"
        lock="${EXP_ROOT}/${rname}/cpu_eval/step_${step}_m${m}.LOCK"
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
  local lock="$1" pid
  [[ -f "$lock" ]] || return 1
  pid="$(cat "$lock" 2>/dev/null || true)"
  [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null
}

launch_one() {
  local run_dir="$1" step="$2" m="$3"
  local name out donef lock slot cpu0 cpu1
  name="$(basename "$run_dir")"
  if [[ "$name" == _* || ! -f "$run_dir/checkpoints/params_${step}.pkl" ]]; then
    return 0
  fi
  mkdir -p "${run_dir}/cpu_eval"
  donef="${run_dir}/cpu_eval/step_${step}_m${m}.DONE"
  lock="${run_dir}/cpu_eval/step_${step}_m${m}.LOCK"
  if [[ -f "$donef" ]]; then
    return 0
  fi
  if lock_held "$lock"; then
    return 0
  fi
  rm -f "$lock" 2>/dev/null || true

  while (( ${#PIDS[@]} >= MAX )); do
    if ! reap_one; then sleep 5; fi
  done
  slot="$(next_cpu_slot)" || slot=0
  cpu0=$((CPU_BASE + slot * CPU_STRIDE))
  cpu1=$((cpu0 + CPU_STRIDE - 1))
  out="$LOGROOT/${name}_step${step}_m${m}.out"
  log "LAUNCH ${name} step=${step} m=${m} cpus=${cpu0}-${cpu1}"
  (
    set +e
    taskset -c "${cpu0}-${cpu1}" \
      "$PY" -u scripts/cpu_eval_pbf_pathwt_ckpt.py \
        --run-dir="${run_dir}" \
        --checkpoint-step="${step}" \
        --episodes="${EPISODES}" \
        --seed=0 \
        --grid="${EVAL_GRID}" \
        --prefix-samples="${m}" \
      >"$out" 2>&1
    exit $?
  ) &
  local child=$!
  echo "$child" >"$lock"
  PIDS+=("$child")
  KEYS+=("${name}_step${step}_m${m}")
  CPU_SLOTS+=("$slot")
}

list_pending() {
  "$PY" - "$EXP_ROOT" "${MS[*]}" <<'PY' || true
from pathlib import Path
import os, sys
root = Path(sys.argv[1])
ms = [int(x) for x in sys.argv[2].split()]
steps = (800000, 900000, 1000000)
if not root.is_dir():
    raise SystemExit
for d in sorted(root.iterdir()):
    if not d.is_dir() or d.name.startswith('_') or not (d / 'flags.json').exists():
        continue
    ck = d / 'checkpoints'
    for step in steps:
        if not (ck / f'params_{step}.pkl').is_file():
            continue
        for m in ms:
            done = d / 'cpu_eval' / f'step_{step}_m{m}.DONE'
            if done.is_file():
                continue
            lock = d / 'cpu_eval' / f'step_{step}_m{m}.LOCK'
            if lock.is_file():
                try:
                    pid = int(lock.read_text().strip())
                except Exception:
                    pid = -1
                if pid > 0:
                    try:
                        os.kill(pid, 0)
                        continue
                    except OSError:
                        pass
            print(f'{d}|{step}|{m}')
PY
}

log "START M-sweep group=${RUN_GROUP} Ms=${MS[*]} MAX=${MAX} CPU_BASE=${CPU_BASE} grid=${EVAL_GRID}"
echo $$ >"$LOGROOT/sweep.pid"

idle_rounds=0
while true; do
  if [[ -f "$LOGROOT/STOP" ]]; then
    log "STOP requested"
    break
  fi
  while reap_one; do :; done
  mapfile -t pending < <(list_pending)
  if (( ${#pending[@]} > 0 )); then
    idle_rounds=0
    for item in "${pending[@]}"; do
      [[ -z "$item" ]] && continue
      IFS='|' read -r rd st m <<<"$item"
      launch_one "$rd" "$st" "$m"
    done
  else
    n_complete=0
    if [[ -d "$EXP_ROOT" ]]; then
      for d in "$EXP_ROOT"/*/; do
        [[ -f "${d}checkpoints/params_1000000.pkl" ]] || continue
        ok=1
        for st in "${STEPS[@]}"; do
          for m in "${MS[@]}"; do
            [[ -f "${d}cpu_eval/step_${st}_m${m}.DONE" ]] || ok=0
          done
        done
        (( ok )) && n_complete=$((n_complete + 1))
      done
    fi
    if (( n_complete >= EXPECTED_RUNS )) && (( ${#PIDS[@]} == 0 )); then
      log "SWEEP_COMPLETE runs=${n_complete} Ms=${MS[*]}"
      break
    fi
    idle_rounds=$((idle_rounds + 1))
    if (( idle_rounds % 20 == 1 )); then
      log "idle pending=0 live=${#PIDS[@]} complete_runs=${n_complete}/${EXPECTED_RUNS}"
    fi
  fi
  sleep "$POLL_SEC"
done

while (( ${#PIDS[@]} > 0 )); do
  if ! reap_one; then sleep 5; fi
done
log "EXIT"
rm -f "$LOGROOT/sweep.pid"

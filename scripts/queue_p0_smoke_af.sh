#!/usr/bin/env bash
# PathBridger_AF p0_smoke queue — action-free algos only, offline defaults (1M/3M/50k),
# online 50k, seed0, envs: amm / cube-double / scene. 2 GPUs × 1 slot.
set +e
set -uo pipefail

REPO="${REPO:-/home/ext_csv/PathBridger_AF}"
PY="${PY:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
LOGROOT="${LOGROOT:-$REPO/nohup_logs/p0_smoke_af}"
# shellcheck disable=SC2206
GPU_LIST=(${GPU_LIST:-0 1})
MAX="${MAX:-${#GPU_LIST[@]}}"
RUN_GROUP="${RUN_GROUP:-p0_smoke_50k}"
ONLINE_STEPS="${ONLINE_STEPS:-50000}"
EVAL_STEPS="${EVAL_STEPS:-0,10000,25000,50000}"
SEED="${SEED:-0}"
POLL_SEC="${POLL_SEC:-30}"

# Error 21 thread pins
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export EIGEN_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_gpu_force_compilation_parallelism=1 --xla_gpu_autotune_level=0 ${XLA_FLAGS:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOGROOT"
cd "$REPO" || exit 1

ts() { TZ=Asia/Seoul date '+%F %T %Z'; }
log() { printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$LOGROOT/orchestrator.log"; }

# action-free only (no gc_sac / gc_td3 / gc_sac_50_50)
ALGOS=(
  pbf_online_idm
  gc_mscp_style
  hiql_endpoint_online
  gc_af_guide
  gc_oso_decqn_factorized
)
ENVS=(
  antmaze-medium-navigate-v0:antmaze_medium
  cube-double-play-v0:cube_double
  scene-play-v0:scene
)

declare -a QUEUE=()
for algo in "${ALGOS[@]}"; do
  for pair in "${ENVS[@]}"; do
    env="${pair%%:*}"
    cfg="${pair##*:}"
    QUEUE+=("${algo}|${env}|${cfg}")
  done
done

declare -a PIDS=()
declare -a KEYS=()
declare -a GPU_SLOTS=()
declare -a CPU_BASES=(0 16)  # 16 cores each for 2 slots on 224-core host

job_tag() {
  local algo="$1" env="$2"
  echo "${algo}_${env}_s${SEED}" | tr '-' '_'
}

is_done() {
  local algo="$1" env="$2"
  local tag donef
  tag="$(job_tag "$algo" "$env")"
  donef="${LOGROOT}/${tag}.DONE"
  [[ -f "$donef" ]]
}

reap_one() {
  local i pid key rc
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      key="${KEYS[$i]}"
      wait "$pid" 2>/dev/null && rc=0 || rc=$?
      log "DONE key=${key} pid=${pid} rc=${rc}"
      if (( rc == 0 )); then
        touch "${LOGROOT}/${key}.DONE"
      else
        echo "rc=${rc}" >"${LOGROOT}/${key}.FAIL"
      fi
      rm -f "${LOGROOT}/${key}.pid" "${LOGROOT}/${key}.LOCK" 2>/dev/null || true
      unset 'PIDS[i]' 'KEYS[i]' 'GPU_SLOTS[i]'
      PIDS=("${PIDS[@]}")
      KEYS=("${KEYS[@]}")
      GPU_SLOTS=("${GPU_SLOTS[@]}")
      return 0
    fi
  done
  return 1
}

next_slot() {
  local used s
  for ((s=0; s<MAX; s++)); do
    used=0
    for u in "${GPU_SLOTS[@]:-}"; do
      [[ "$u" == "$s" ]] && used=1 && break
    done
    if (( used == 0 )); then
      echo "$s"
      return 0
    fi
  done
  return 1
}

launch_one() {
  local algo="$1" env="$2" cfg="$3"
  local tag out slot gpu cpu0 cpu1 cmd
  tag="$(job_tag "$algo" "$env")"
  if is_done "$algo" "$env"; then
    log "SKIP done ${tag}"
    return 0
  fi
  if [[ -f "${LOGROOT}/${tag}.LOCK" ]]; then
    local lpid
    lpid="$(cat "${LOGROOT}/${tag}.LOCK" 2>/dev/null || true)"
    if [[ -n "${lpid:-}" ]] && kill -0 "$lpid" 2>/dev/null; then
      log "SKIP lock ${tag} pid=${lpid}"
      return 0
    fi
    rm -f "${LOGROOT}/${tag}.LOCK"
  fi

  while (( ${#PIDS[@]} >= MAX )); do
    if ! reap_one; then sleep 5; fi
  done
  slot="$(next_slot)" || slot=0
  gpu="${GPU_LIST[$slot]}"
  cpu0=$((CPU_BASES[slot]))
  cpu1=$((cpu0 + 15))
  out="${LOGROOT}/${tag}.out"

  cmd=(
    "$PY" -u train_af.py
    --algorithm="$algo"
    --seed="$SEED"
    --run_group="$RUN_GROUP"
    --protocol_suite=p0_smoke
    --online_steps="$ONLINE_STEPS"
    --random_steps=10000
    --replay_capacity=1000000
    --eval_steps="$EVAL_STEPS"
    --eval_episodes=10
    --nouse_tqdm
    --save_dir=exp/
  )
  if [[ "$algo" == "pbf_online_idm" ]]; then
    cmd+=(--pbf="configs/pbf_af/${cfg}.py")
  else
    cmd+=(--env_name="$env")
  fi

  log "LAUNCH gpu=${gpu} cpus=${cpu0}-${cpu1} ${tag}"
  (
    set +e
    export CUDA_VISIBLE_DEVICES="${gpu}"
    taskset -c "${cpu0}-${cpu1}" "${cmd[@]}" >"$out" 2>&1
    exit $?
  ) &
  local child=$!
  echo "$child" >"${LOGROOT}/${tag}.pid"
  echo "$child" >"${LOGROOT}/${tag}.LOCK"
  PIDS+=("$child")
  KEYS+=("$tag")
  GPU_SLOTS+=("$slot")
}

log "START p0_smoke AF-only jobs=${#QUEUE[@]} GPUs=${GPU_LIST[*]} MAX=${MAX} offline=defaults online=${ONLINE_STEPS}"
echo $$ >"$LOGROOT/orchestrator.pid"

idx=0
while (( idx < ${#QUEUE[@]} )) || (( ${#PIDS[@]} > 0 )); do
  if [[ -f "$LOGROOT/STOP" ]]; then
    log "STOP requested — waiting for live jobs (no kill -9)"
    while (( ${#PIDS[@]} > 0 )); do
      if ! reap_one; then sleep 5; fi
    done
    break
  fi
  while reap_one; do :; done
  while (( idx < ${#QUEUE[@]} )) && (( ${#PIDS[@]} < MAX )); do
    IFS='|' read -r algo env cfg <<<"${QUEUE[$idx]}"
    idx=$((idx + 1))
    launch_one "$algo" "$env" "$cfg"
  done
  if (( idx >= ${#QUEUE[@]} )) && (( ${#PIDS[@]} == 0 )); then
    log "QUEUE_COMPLETE"
    break
  fi
  sleep "$POLL_SEC"
done

log "EXIT"
rm -f "$LOGROOT/orchestrator.pid"

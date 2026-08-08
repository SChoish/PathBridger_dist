#!/usr/bin/env bash
# PathBridger_AF_pixel_lapo pilot — 6 algos × 4 visual envs × seeds {0,1,2} = 72.
# Online 250k. Soft-stop + resume_interval=50k. RAID visual datasets required.
set +e
set -uo pipefail

REPO="${REPO:-/home/ext_csv/PathBridger_AF_pixel_lapo}"
PY="${PY:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
LOGROOT="${LOGROOT:-$REPO/nohup_logs/pixel_pilot}"
DATASET_DIR="${DATASET_DIR:-/raid/ext_csv/datasets/ogbench_visual}"
# shellcheck disable=SC2206
GPU_LIST=(${GPU_LIST:-0 1})
SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"
MAX="${MAX:-$((${#GPU_LIST[@]} * SLOTS_PER_GPU))}"
RUN_GROUP="${RUN_GROUP:-pixel_pilot_250k}"
ONLINE_STEPS="${ONLINE_STEPS:-250000}"
EVAL_STEPS="${EVAL_STEPS:-0,10000,25000,50000,100000,250000}"
# shellcheck disable=SC2206
SEEDS=(${SEEDS:-0 1 2})
POLL_SEC="${POLL_SEC:-30}"
CPU_STRIDE="${CPU_STRIDE:-16}"
CPU_BASE0="${CPU_BASE0:-0}"
LAUNCH_STAGGER_SEC="${LAUNCH_STAGGER_SEC:-20}"
AUTO_RESUME="${AUTO_RESUME:-1}"
SAVE_REPLAY="${SAVE_REPLAY:-1}"
RESUME_KEEP="${RESUME_KEEP:-1}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.40}"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export EIGEN_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_gpu_force_compilation_parallelism=1 ${XLA_FLAGS:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOGROOT"
cd "$REPO" || exit 1

ts() { TZ=Asia/Seoul date '+%F %T %Z'; }
log() { printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$LOGROOT/orchestrator.log"; }

if [[ ! -d "$DATASET_DIR" ]]; then
  log "FATAL dataset_dir missing: $DATASET_DIR"
  exit 1
fi
if [[ -e "$LOGROOT/STOP" ]]; then
  log "FATAL stale STOP marker: $LOGROOT/STOP (remove it explicitly before launch)"
  exit 1
fi

ALGOS=(
  gc_pixel_lapo_decoder
  gc_pixel_drqv2
  vip_style_frozen_gc_drqv2
  vip_style_finetuned_gc_drqv2
  gc_pixel_apv_style_drq
)
ENVS=(
  visual-antmaze-large-navigate-v0
  visual-cube-double-play-v0
  visual-puzzle-4x4-play-v0
  visual-scene-play-v0
)

declare -a QUEUE=()
for algo in "${ALGOS[@]}"; do
  for env in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      QUEUE+=("${algo}|${env}|${seed}")
    done
  done
done

declare -a PIDS=()
declare -a KEYS=()
declare -a GPU_SLOTS=()
declare -a SLOT_GPUS=()
declare -a CPU_BASES=()

slot_i=0
for gpu in "${GPU_LIST[@]}"; do
  for ((s = 0; s < SLOTS_PER_GPU; s++)); do
    SLOT_GPUS+=("$gpu")
    CPU_BASES+=($((CPU_BASE0 + slot_i * CPU_STRIDE)))
    slot_i=$((slot_i + 1))
  done
done
MAX=${#SLOT_GPUS[@]}

job_tag() {
  local algo="$1" env="$2" seed="$3"
  echo "${algo}_${env}_s${seed}" | tr '-' '_'
}

is_done() {
  [[ -f "${LOGROOT}/$(job_tag "$1" "$2" "$3").DONE" ]]
}

reap_one() {
  local i pid key rc
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      key="${KEYS[$i]}"
      wait "$pid" 2>/dev/null && rc=0 || rc=$?
      log "DONE key=${key} pid=${pid} rc=${rc}"
      if ((rc == 0)); then
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

soft_stop_all() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      log "SOFT_STOP pid=${pid}"
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
}

next_slot() {
  local used s
  for ((s = 0; s < MAX; s++)); do
    used=0
    for u in "${GPU_SLOTS[@]:-}"; do
      [[ "$u" == "$s" ]] && used=1 && break
    done
    if ((used == 0)); then
      echo "$s"
      return 0
    fi
  done
  return 1
}

launch_one() {
  local algo="$1" env="$2" seed="$3"
  local tag out slot gpu cpu0 cpu1 restore_run seed_tag
  tag="$(job_tag "$algo" "$env" "$seed")"
  if is_done "$algo" "$env" "$seed"; then
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

  while ((${#PIDS[@]} >= MAX)); do
    if ! reap_one; then sleep 5; fi
  done
  slot="$(next_slot)" || slot=0
  gpu="${SLOT_GPUS[$slot]}"
  cpu0="${CPU_BASES[$slot]}"
  cpu1=$((cpu0 + CPU_STRIDE - 1))
  out="${LOGROOT}/${tag}.out"

  local -a cmd=(
    "$PY" -u train_pixel.py
    --algorithm="$algo"
    --env_name="$env"
    --seed="$seed"
    --run_group="$RUN_GROUP"
    --protocol_suite=pilot
    --dataset_dir="$DATASET_DIR"
    --online_steps="$ONLINE_STEPS"
    --random_steps=10000
    --replay_capacity=50000
    --frame_stack=3
    --her_probability=0.8
    --eval_steps="$EVAL_STEPS"
    --eval_episodes=10
    --resume_interval=50000
    --resume_keep="$RESUME_KEEP"
    --nouse_tqdm
    --save_dir=exp/
  )
  if [[ "$SAVE_REPLAY" == "1" ]]; then
    cmd+=(--save_replay)
  else
    cmd+=(--nosave_replay)
  fi
  restore_run=""
  if [[ "$AUTO_RESUME" == "1" ]]; then
    seed_tag="$(printf '%03d' "$seed")"
    restore_run="$({
      find "$REPO/exp/pixel_o2o/$RUN_GROUP" \
        -mindepth 1 -maxdepth 1 -type d \
        -name "${algo}_${env}_sd${seed_tag}_*" \
        -printf '%T@ %p\n' 2>/dev/null || true
    } | sort -nr | head -1 | cut -d' ' -f2-)"
    if [[ -n "$restore_run" ]] && compgen -G "$restore_run/checkpoints/*.pkl" >/dev/null; then
      cmd+=(
        --restore_path="$restore_run/checkpoints"
        --restore_step=-1
        --resume_in_place
      )
      log "RESUME tag=${tag} run=${restore_run}"
    fi
  fi

  log "LAUNCH gpu=${gpu} cpus=${cpu0}-${cpu1} mem_frac=${XLA_PYTHON_CLIENT_MEM_FRACTION} ${tag}"
  (
    set +e
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION}"
    exec taskset -c "${cpu0}-${cpu1}" "${cmd[@]}" >"$out" 2>&1
  ) &
  local child=$!
  echo "$child" >"${LOGROOT}/${tag}.pid"
  echo "$child" >"${LOGROOT}/${tag}.LOCK"
  PIDS+=("$child")
  KEYS+=("$tag")
  GPU_SLOTS+=("$slot")
  if ((LAUNCH_STAGGER_SEC > 0)); then
    sleep "$LAUNCH_STAGGER_SEC"
  fi
}

log "START pixel pilot jobs=${#QUEUE[@]} GPUs=${GPU_LIST[*]} slots_per_gpu=${SLOTS_PER_GPU} MAX=${MAX} online=${ONLINE_STEPS} seeds=${SEEDS[*]} dataset=${DATASET_DIR} auto_resume=${AUTO_RESUME} save_replay=${SAVE_REPLAY}"
echo $$ >"$LOGROOT/orchestrator.pid"

idx=0
while ((idx < ${#QUEUE[@]})) || ((${#PIDS[@]} > 0)); do
  if [[ -f "$LOGROOT/STOP" ]]; then
    log "STOP requested — forwarding SIGTERM for emergency checkpoints"
    soft_stop_all
    while ((${#PIDS[@]} > 0)); do
      if ! reap_one; then sleep 5; fi
    done
    break
  fi
  while reap_one; do :; done
  while ((idx < ${#QUEUE[@]})) && ((${#PIDS[@]} < MAX)); do
    IFS='|' read -r algo env seed <<<"${QUEUE[$idx]}"
    idx=$((idx + 1))
    launch_one "$algo" "$env" "$seed"
  done
  if ((idx >= ${#QUEUE[@]})) && ((${#PIDS[@]} == 0)); then
    log "QUEUE_COMPLETE"
    break
  fi
  sleep "$POLL_SEC"
done

log "EXIT"
rm -f "$LOGROOT/orchestrator.pid"

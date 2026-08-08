#!/usr/bin/env bash
set -euo pipefail

baseline_name="${1:?usage: $0 pixel_hiql|pixel_ota ENV_NAME [SAVE_DIR]}"
benchmark_env="${2:?usage: $0 pixel_hiql|pixel_ota ENV_NAME [SAVE_DIR]}"
benchmark_save_dir="${3:-exp}"

case "${baseline_name}" in
  pixel_hiql|pixel_ota) ;;
  *)
    echo "baseline must be pixel_hiql or pixel_ota" >&2
    exit 2
    ;;
esac

python train_pixel.py \
  --algorithm="${baseline_name}" \
  --env_name="${benchmark_env}" \
  --save_dir="${benchmark_save_dir}" \
  --run_group="pixel_hiql_ota_offline" \
  --protocol_suite="official_visual_offline" \
  --offline_steps=500000 \
  --online_steps=0 \
  --random_steps=0 \
  --frame_stack=1 \
  --eval_steps=0 \
  --eval_episodes=50

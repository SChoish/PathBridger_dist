"""Compile and time the production pixel-PBF update path on one GPU."""

from __future__ import annotations

import argparse
import time

import jax
import numpy as np

from agents.pixel_registry import create_pixel_algorithm
from agents.pixel_trl_critic_locks import apply_pixel_pbf_locks
from utils.pixel_data import PixelTrajectoryData


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--timed-steps', type=int, default=3)
    parser.add_argument('--image-size', type=int, default=64)
    args = parser.parse_args()

    devices = jax.devices()
    if not devices or devices[0].platform != 'gpu':
        raise RuntimeError(f'GPU smoke requires a GPU JAX device, got {devices}.')
    if args.batch_size != 256:
        raise ValueError('Production pixel PBF locks --batch-size=256.')
    if args.timed_steps < 1:
        raise ValueError('timed-steps must be positive.')

    rng = np.random.default_rng(20260808)
    frame_count = 512
    frames = rng.integers(
        0,
        256,
        (frame_count, args.image_size, args.image_size, 3),
        dtype=np.uint8,
    )
    terminals = np.zeros(frame_count, dtype=np.float32)
    terminals[-1] = 1.0
    actions = rng.uniform(-1.0, 1.0, (frame_count, 5)).astype(np.float32)
    data = PixelTrajectoryData(
        {'observations': frames, 'terminals': terminals, 'actions': actions},
        seed=7,
        frame_stack=1,
    )
    config = apply_pixel_pbf_locks(
        'visual-cube-double-play-v0',
        {
            'frame_stack': 1,
            'offline_batch_size': args.batch_size,
            'endpoint_value_scale': 5.0,
        },
    )
    agent, config = create_pixel_algorithm(
        'pixel_pbf',
        seed=11,
        example_images=data.example_images,
        action_dim=actions.shape[-1],
        config=config,
    )
    device_frames = jax.device_put(data.observations)
    device_initials = jax.device_put(
        data.episodes.initial_for_state.astype(np.int32, copy=False)
    )

    def update_once(current_agent, *, full_metrics: bool):
        batch = data.sample(
            args.batch_size,
            path_horizon=int(config['path_horizon']),
            endpoint_horizon=int(config['endpoint_horizon']),
            discount=float(config['discount']),
            value_geom_sample=bool(config['value_geom_sample']),
            value_p_curgoal=float(config['value_p_curgoal']),
            value_p_trajgoal=float(config['value_p_trajgoal']),
            value_p_randomgoal=float(config['value_p_randomgoal']),
            pbf_indices_only=True,
        )
        updated, info = current_agent.offline_update_indexed(
            batch,
            device_frames,
            device_initials,
            full_metrics=full_metrics,
        )
        jax.block_until_ready(info['loss/total'])
        return updated, info

    compile_start = time.perf_counter()
    agent, info = update_once(agent, full_metrics=False)
    fast_compile_seconds = time.perf_counter() - compile_start

    timed_start = time.perf_counter()
    for _ in range(args.timed_steps):
        agent, info = update_once(agent, full_metrics=False)
    fast_timed_seconds = time.perf_counter() - timed_start

    diagnostic_compile_start = time.perf_counter()
    agent, info = update_once(agent, full_metrics=True)
    diagnostic_compile_seconds = time.perf_counter() - diagnostic_compile_start
    diagnostic_timed_start = time.perf_counter()
    for _ in range(args.timed_steps):
        agent, info = update_once(agent, full_metrics=True)
    diagnostic_timed_seconds = time.perf_counter() - diagnostic_timed_start

    action_start = time.perf_counter()
    chunks = agent.sample_action_chunks(
        data.example_images[:1],
        data.example_images[1:2],
        seed=jax.random.PRNGKey(13),
        num_candidates=2,
        temperature=0.25,
    )
    chunks = np.asarray(jax.device_get(chunks))
    action_seconds = time.perf_counter() - action_start
    metrics = {key: float(jax.device_get(value)) for key, value in info.items()}

    if not np.all(np.isfinite(chunks)):
        raise RuntimeError('Pixel PBF produced non-finite action chunks.')
    required = (
        'loss/total',
        'repr/cross_sample_std',
        'repr/effective_rank',
        'repr/one_step_distance',
        'idm/action_mse',
    )
    if not all(np.isfinite(metrics[key]) for key in required):
        raise RuntimeError(f'Non-finite smoke metrics: {metrics}.')

    print(f'device={devices[0]}')
    print(f'batch_size={args.batch_size} image_size={args.image_size}')
    fast_mean = fast_timed_seconds / args.timed_steps
    diagnostic_mean = diagnostic_timed_seconds / args.timed_steps
    print(f'fast_compile_and_first_update_seconds={fast_compile_seconds:.3f}')
    print(f'fast_mean_update_seconds={fast_mean:.4f}')
    print(f'fast_updates_per_second={1.0 / fast_mean:.3f}')
    print(
        'diagnostic_compile_and_first_update_seconds='
        f'{diagnostic_compile_seconds:.3f}'
    )
    print(f'diagnostic_mean_update_seconds={diagnostic_mean:.4f}')
    print(f'diagnostic_updates_per_second={1.0 / diagnostic_mean:.3f}')
    print(f'fast_path_speedup={diagnostic_mean / fast_mean:.3f}x')
    print(f'action_compile_and_inference_seconds={action_seconds:.3f}')
    print(f'action_abs_mean={np.mean(np.abs(chunks)):.6f}')
    for key in required:
        print(f'{key}={metrics[key]:.6f}')


if __name__ == '__main__':
    main()

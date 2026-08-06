# Pixel LAPO extension protocol

This LAPO-specific track is separate from the state-vector action-free OGBench table. Its
results, run counts, manifests, checkpoints, and aggregate statistics must not
be pooled with `af_o2o_v3`. The multi-algorithm visual protocol is documented
in `docs/pixel_benchmark_protocol.md`.

## Method status

`gc_pixel_lapo_decoder` is a controlled continuous-control OGBench adaptation of
[Learning to Act without Actions](https://arxiv.org/abs/2312.10812). It is not
an official reproduction of the Procgen experiments in the
[LAPO repository](https://github.com/schmidtdominik/LAPO).

The official method learns a quantized latent action model from videos, clones
a latent policy, and grounds latent actions online. The OGBench adaptation keeps
that causal information flow but makes two declared changes:

1. The latent policy is goal-conditioned on `(current RGB, goal RGB)`.
2. The online grounding model emits bounded continuous OGBench actions.

The original stage-3 PPO policy update is deliberately omitted in this
controlled variant. Only its latent-to-action decoder is updated online; the
visual latent model and latent policy remain frozen and are hash-checked at
every checkpoint. This is an algorithm-specific choice for `gc_pixel_lapo_decoder`,
not a global restriction on the pixel benchmark. Other registered methods may
fine-tune their encoder, policy, critic, or action-conditioned dynamics online
when required by their declared update rule.

## Information boundary and stages

| Stage | Data visible | Trainable module | Objective |
| --- | --- | --- | --- |
| Offline 1 | RGB observation pairs, terminals | VQ latent IDM + world model | next-frame/feature prediction and VQ losses |
| Offline 2 | RGB current/next/future-goal triples, terminals | goal-conditioned latent policy | BC on inferred latent codes |
| Online | newly collected RGB/action/RGB transitions | continuous latent-action decoder only | tanh-Gaussian action NLL |

The visual loader constructs an immutable dataset containing exactly
`observations` and `terminals`. Offline actions, rewards, simulator state, and
privileged metadata are discarded before either offline training stage. Online
action labels enter only through the newly collected replay buffer.

## Benchmark

The environments are the eight official `visual-*` counterparts used by this
repository's state protocol. Evaluation uses official task IDs 1--5, ten
episodes per task, any-step success, and checkpoints at online environment
steps `0, 10k, 25k, 50k, 100k, 250k, 500k, 1M` where applicable.

The isolated manifest generator exposes:

- `p0_smoke`: 3 environments x 1 seed = 3 runs.
- `pilot`: 4 environments x 3 seeds = 12 runs.
- `screening`: 8 environments x 5 seeds = 40 runs.

Generate a manifest without launching it:

```bash
python scripts/make_pixel_lapo_manifest.py --suite=p0_smoke \
  --output=manifests/pixel_lapo_p0.csv
```

No generated pixel manifest is consumed by the existing state queue scripts.

## Defaults and resource notes

The LAPO-aligned defaults are two VQ codebooks, 64 entries per codebook,
16-dimensional codes, 50k stage-1 updates, and 60k stage-2 updates. The unified
`pixel_o2o_v3` runner uses three-frame histories and a 50k transition replay
whose transitions reference shared raw `uint8` frames. This replaces the old
three-full-image-per-transition layout. The official visual datasets are large
and should be provisioned explicitly. Training refuses to download a missing
dataset by default; opt in only with `--allow_dataset_download=true`. The
manifest generator never starts training or downloads data.

Example development smoke after datasets are available:

```bash
python train_pixel_lapo.py \
  --env_name=visual-antmaze-medium-navigate-v0 \
  --lapo_stage1_steps=2 --lapo_stage2_steps=2 --online_steps=10 \
  --random_steps=10 --update_start=2 --replay_capacity=20 \
  --eval_episodes=0 --eval_steps=10 --use_tqdm=false
```

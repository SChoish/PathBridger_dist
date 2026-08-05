# PathBridger-TriangleQ

`pathbridger_triangleQ` keeps PathBridger's PBF endpoint flow,
endpoint-pinned state bridge, and inverse-dynamics model (IDM), while replacing
the Bridger transitive-value critic with PathFlower's triangular chunk-Q
structure.

The method is actor-free and state-based:

```text
(current state, final goal)
  -> PBF endpoint candidates
  -> endpoint-pinned bridge for every candidate
  -> IDM decodes each bridge prefix into a 5-step action chunk
  -> Triangle-Q ranks the chunks against the final goal
  -> execute 5 actions and replan
```

## Critic

The critic contains online/EMA copies of `Q(s, A_5, g)` and `V(s, g)`.
Its objectives match the `triangular_q` mode in PathFlower:

1. **Base Q**

   `Q(s_t, A_t, s_{t+d}) <- gamma^d` for a same-trajectory base goal.

2. **Triangular Q**

   `Q(s_t, A_t, g) <- Qbar(s_t, A_t, s_k) * Qbar(s_k, A_k, g)`.

3. **Expectile V**

   `V(s_t, g)` is fit to the EMA Q of the dataset action chunk.

PBF endpoint flow matching is weighted by the target value improvement from
the current state to the dataset endpoint. Bridge reconstruction remains
aligned with the executed five-step prefix. No rotation-, contact-, or
constraint-specific cost is introduced. The PBF endpoint proposer retains its
standard `phi` achieved-goal condition, while Triangle-Q consumes the full goal
observation.

## Training

```bash
cd /home/ext_csv/pathbridger_triangleQ
pip install -e .
python main.py \
  --agent=configs/pbf/cube_triple.py \
  --seed=0 \
  --run_group=paper
```

Eight PBF configs are included under `configs/pbf/`. Training defaults to one
million updates, batch size 1024, logging every 5,000 steps, and checkpointing
every 100,000 steps. Runs are saved below:

```text
exp/pathbridger_triangleq/<run_group>/<experiment>/
```

## Evaluation

```bash
python evaluate.py \
  --agent=configs/pbf/cube_triple.py \
  --checkpoint_dir=exp/pathbridger_triangleq/paper/<run>/checkpoints \
  --checkpoint_step=1000000
```

Candidate count and PBF temperature are environment-specific. Candidate
ranking always uses the triangular action-chunk critic; the scalar value is a
training signal and endpoint-flow weighting signal.

## Tests

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest -q
```

Checkpoints are not compatible with PathBridger, PathBridger-DQC, or
PathFlower because the joint module and optimizer trees differ.

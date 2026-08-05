"""Optional end-to-end agent smoke test for installations with the ML stack."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")
pytest.importorskip("optax")
pytest.importorskip("ml_collections")

import agents.pathbridger as pathbridger  # noqa: E402


def _small_agent(monkeypatch):
    """Use the real modules and losses with small hidden layers for test speed."""

    for name in (
        "ScalarValueNet",
        "GaussianEndpointProposer",
        "FlowEndpointProposer",
        "BridgeResidual",
        "InverseDynamics",
        "BinaryChunkCritic",
    ):
        module = getattr(pathbridger, name)
        monkeypatch.setattr(
            pathbridger,
            name,
            partial(module, hidden_dims=(8,)),
        )

    config = pathbridger.get_config().to_dict()
    config.update(
        endpoint_distribution="gaussian",
        horizon=5,
        sequence_horizon=5,
        action_chunk_horizon=5,
        eval_num_candidates=2,
        eval_temperature=0.0,
    )
    observations = jnp.asarray(
        [[0.0, 0.0, 0.1, -0.1], [0.1, 0.0, 0.2, -0.1]],
        dtype=jnp.float32,
    )
    actions = jnp.asarray([[0.0, 0.1], [0.1, 0.0]], dtype=jnp.float32)
    agent = pathbridger.PathBridgerAgent.create(
        seed=0,
        ex_observations=observations,
        ex_actions=actions,
        config=config,
    )
    return agent, observations, actions


def _batch(observations, actions):
    batch_size, state_dim = observations.shape
    next_observations = observations + 0.01
    endpoint_targets = observations + 0.05
    times = jnp.linspace(0.0, 1.0, 6, dtype=jnp.float32)
    trajectory = (
        observations[:, None, :]
        + times[None, :, None]
        * (endpoint_targets - observations)[:, None, :]
    )
    return {
        "observations": observations,
        "next_observations": next_observations,
        "actions": actions,
        "trajectory": trajectory,
        "endpoint_goals": observations + 0.1,
        "endpoint_targets": endpoint_targets,
        "value_goals": observations + 0.1,
        "value_offsets": jnp.asarray([5.0, 5.0], dtype=jnp.float32),
        "action_chunk_actions": jnp.tile(actions, (1, 5)),
        "valids": jnp.ones((batch_size, 5), dtype=jnp.float32),
        "trl_base_goals": observations + 0.05,
        "trl_base_offsets": jnp.asarray([5.0, 5.0], dtype=jnp.float32),
        "trl_split_observations": observations + 0.02,
        "trl_split_goals": observations + 0.02,
        "trl_split_action_chunk_actions": jnp.tile(actions, (1, 5)),
        "trl_split_offsets": jnp.asarray([2.0, 2.0], dtype=jnp.float32),
        "trl_valid_mask": jnp.ones((batch_size,), dtype=jnp.float32),
    }


def test_agent_create_update_and_sample_action_chunks(monkeypatch):
    agent, observations, actions = _small_agent(monkeypatch)
    batch = _batch(observations, actions)

    fast_agent, fast_info = agent.update(batch, full_metrics=False)
    updated_agent, info = agent.update(batch, full_metrics=True)
    assert int(updated_agent.network.step) == int(agent.network.step) + 1
    assert "loss/total" in info
    assert "triangle_q/base_loss" in info
    assert "triangle_q/triangle_loss" in info
    assert "triangle_q/value_loss" in info
    assert np.isfinite(np.asarray(jax.device_get(info["loss/total"]))).all()
    assert "triangle_q/base_pred_mean" not in fast_info
    assert "endpoint/weight_mean" not in fast_info
    assert "triangle_q/base_pred_mean" in info
    assert "endpoint/weight_mean" in info
    fast_leaves = jax.tree_util.tree_leaves(fast_agent)
    full_leaves = jax.tree_util.tree_leaves(updated_agent)
    assert len(fast_leaves) == len(full_leaves)
    for fast, full in zip(fast_leaves, full_leaves):
        np.testing.assert_allclose(fast, full, rtol=1e-6, atol=1e-6)

    chunks = updated_agent.sample_action_chunks(
        observations=observations[:1],
        goals=observations[:1] + 0.2,
        seed=jax.random.PRNGKey(3),
        num_candidates=2,
        temperature=0.0,
    )
    chunks = np.asarray(jax.device_get(chunks))
    assert chunks.shape == (1, 5, actions.shape[-1])
    assert np.isfinite(chunks).all()

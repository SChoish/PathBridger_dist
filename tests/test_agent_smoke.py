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
        "ScalarTransitiveValue",
        "GaussianEndpointProposer",
        "FlowEndpointProposer",
        "BridgeResidual",
        "InverseDynamics",
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
        "value_offsets": jnp.asarray([3.0, 6.0], dtype=jnp.float32),
        "base_goals": observations + 0.03,
        "base_offsets": jnp.asarray([3.0, 5.0], dtype=jnp.float32),
        "transitive_subgoals": jnp.stack(
            (observations[0], observations[1] + 0.04),
            axis=0,
        ),
        "transitive_offsets": jnp.asarray([0.0, 3.0], dtype=jnp.float32),
        "transitive_valids": jnp.asarray([0.0, 1.0], dtype=jnp.float32),
    }


def test_agent_create_update_and_sample_action_chunks(monkeypatch):
    agent, observations, actions = _small_agent(monkeypatch)
    batch = _batch(observations, actions)

    updated_agent, info = agent.update(batch)
    assert int(updated_agent.network.step) == int(agent.network.step) + 1
    assert "loss/total" in info
    assert np.isfinite(np.asarray(jax.device_get(info["loss/total"]))).all()

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

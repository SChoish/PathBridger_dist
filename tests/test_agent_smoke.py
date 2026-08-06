"""Optional end-to-end agent smoke test for installations with the ML stack."""

from __future__ import annotations

from functools import partial
import pickle

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")
pytest.importorskip("optax")
pytest.importorskip("ml_collections")

import agents.pathbridger as pathbridger  # noqa: E402
from utils.flax_utils import restore_agent, save_agent  # noqa: E402


def _small_agent(monkeypatch, **config_overrides):
    """Use the real modules and losses with small hidden layers for test speed."""

    for name in (
        "ScalarTransitiveValue",
        "GaussianEndpointProposer",
        "FlowEndpointProposer",
        "BridgeResidual",
        "InverseDynamics",
        "LowRankGaussianPrefix",
        "JointFlowPrefix",
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
    config.update(config_overrides)
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
        "bridge_targets": trajectory[:, 1:, :],
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

    fast_agent, fast_info = agent.update(batch, full_metrics=False)
    updated_agent, info = agent.update(batch, full_metrics=True)
    assert int(updated_agent.network.step) == int(agent.network.step) + 1
    assert "loss/total" in info
    assert np.isfinite(np.asarray(jax.device_get(info["loss/total"]))).all()
    assert "value/self_mean" not in fast_info
    assert "endpoint/weight_mean" not in fast_info
    assert "value/self_mean" in info
    assert "endpoint/weight_mean" in info
    assert "bridge/path_energy_mean" in info
    assert float(info["bridge/path_weight_strength"]) == 0.0
    assert float(info["bridge/path_weight_mean"]) == 1.0
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


def test_temporal_path_weights_handle_progress_stagnation_and_padding():
    distances = jnp.asarray(
        [
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    active = jnp.asarray(
        [[1.0] * 5, [1.0] * 5, [0.0] * 5],
        dtype=jnp.float32,
    )

    weights, energies, defects, deltas = pathbridger._temporal_path_weights(
        distances,
        active,
        jnp.asarray(1.0, dtype=jnp.float32),
        0.1,
    )

    np.testing.assert_allclose(energies, [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(weights, [1.0, np.exp(-1.0), 1.0], atol=1e-6)
    np.testing.assert_allclose(defects[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(defects[2], 0.0, atol=1e-6)
    np.testing.assert_allclose(deltas[1], 0.0, atol=1e-6)


def test_training_prefix_matches_full_bridge(monkeypatch):
    agent, observations, _ = _small_agent(monkeypatch, horizon=8)
    endpoints = observations + 0.2

    full_bridge = agent._construct_bridge(observations, endpoints)
    training_prefix = agent._construct_bridge_at_indices(
        observations,
        endpoints,
        jnp.arange(1, 6),
    )

    np.testing.assert_allclose(
        np.asarray(jax.device_get(training_prefix)),
        np.asarray(jax.device_get(full_bridge[:, 1:6, :])),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("prefix_model", ["low_rank_gaussian", "joint_flow"])
def test_stochastic_backend_update_sample_and_endpoint_pin(
    monkeypatch,
    prefix_model,
):
    agent, observations, actions = _small_agent(
        monkeypatch,
        prefix_model=prefix_model,
        prefix_rank=2,
        prefix_flow_steps=4,
    )
    agent = agent.replace(
        state_scale=jnp.asarray([1.0, 2.0, 0.5, 1.5], dtype=jnp.float32)
    )
    batch = _batch(observations, actions)

    updated_agent, info = agent.update(batch, full_metrics=True)
    assert "prefix/loss" in info
    assert "loss/prefix_weighted" in info
    assert np.isfinite(np.asarray(jax.device_get(info["prefix/loss"]))).all()

    endpoints = batch["endpoint_targets"]
    prefixes = updated_agent.sample_prefixes(
        observations,
        endpoints,
        seed=jax.random.PRNGKey(9),
        num_samples=2,
        temperature=1.0,
        include_deterministic=True,
    )
    repeated_prefixes = updated_agent.sample_prefixes(
        observations,
        endpoints,
        seed=jax.random.PRNGKey(9),
        num_samples=2,
        temperature=1.0,
        include_deterministic=True,
    )
    np.testing.assert_allclose(prefixes, repeated_prefixes)
    prefixes = np.asarray(jax.device_get(prefixes))
    assert prefixes.shape == (2, 3, 6, observations.shape[-1])
    np.testing.assert_allclose(
        prefixes[:, :, 0, :],
        np.broadcast_to(observations[:, None, :], prefixes[:, :, 0, :].shape),
    )
    np.testing.assert_allclose(
        prefixes[:, :, -1, :],
        np.broadcast_to(endpoints[:, None, :], prefixes[:, :, -1, :].shape),
    )
    assert np.isfinite(prefixes).all()

    chunks = updated_agent.sample_action_chunks(
        observations,
        observations + 0.2,
        seed=jax.random.PRNGKey(10),
        num_candidates=2,
        temperature=0.0,
    )
    assert np.asarray(chunks).shape == (2, 5, actions.shape[-1])
    assert np.isfinite(np.asarray(chunks)).all()


def test_h_less_than_k_stochastic_prefix_is_not_endpoint_pinned(monkeypatch):
    agent, observations, _ = _small_agent(
        monkeypatch,
        horizon=8,
        prefix_model="low_rank_gaussian",
        prefix_rank=2,
    )
    endpoints = observations + 0.2
    prefixes = np.asarray(
        agent.sample_prefixes(
            observations,
            endpoints,
            seed=jax.random.PRNGKey(11),
            num_samples=2,
            temperature=1.0,
        )
    )
    assert prefixes.shape == (2, 2, 6, observations.shape[-1])
    assert not np.allclose(prefixes[:, :, -1, :], endpoints[:, None, :])


def test_prefix_rng_does_not_change_endpoint_samples(monkeypatch):
    deterministic, observations, _ = _small_agent(monkeypatch)
    stochastic, _, _ = _small_agent(
        monkeypatch,
        prefix_model="low_rank_gaussian",
        prefix_rank=2,
    )
    goals = observations + 0.2
    seed = jax.random.PRNGKey(12)
    deterministic_candidates = deterministic._sample_endpoint_candidates(
        observations,
        goals,
        seed,
        num_candidates=3,
        temperature=0.5,
    )
    stochastic_candidates = stochastic._sample_endpoint_candidates(
        observations,
        goals,
        seed,
        num_candidates=3,
        temperature=0.5,
    )
    np.testing.assert_allclose(
        deterministic_candidates,
        stochastic_candidates,
        rtol=1e-6,
        atol=1e-6,
    )


def test_transv_chain_includes_exact_deterministic_candidate(monkeypatch):
    agent, observations, actions = _small_agent(
        monkeypatch,
        horizon=8,
        prefix_model="low_rank_gaussian",
        prefix_rank=2,
        eval_prefix_selection="transv_chain",
        eval_num_prefix_samples=2,
        eval_include_deterministic_prefix=True,
    )
    endpoints = observations + 0.2
    candidates = agent._sample_prefixes(
        observations,
        endpoints,
        jax.random.PRNGKey(13),
        num_samples=2,
        temperature=1.0,
        include_deterministic=True,
    )
    deterministic = agent._construct_bridge_at_indices(
        observations,
        endpoints,
        jnp.arange(0, 6),
    )
    np.testing.assert_allclose(candidates[:, 0, :, :], deterministic)
    assert agent._transv_chain_scores(candidates, endpoints).shape == (2, 3)

    chunks = agent.sample_action_chunks(
        observations,
        observations + 0.3,
        seed=jax.random.PRNGKey(14),
        num_candidates=2,
        temperature=0.0,
    )
    assert np.asarray(chunks).shape == (2, 5, actions.shape[-1])
    assert np.isfinite(np.asarray(chunks)).all()


def test_stochastic_checkpoint_metadata_and_legacy_restore(monkeypatch, tmp_path):
    stochastic, observations, _ = _small_agent(
        monkeypatch,
        prefix_model="low_rank_gaussian",
        prefix_rank=2,
    )
    stochastic = stochastic.replace(
        state_scale=jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    )
    path = save_agent(stochastic, tmp_path, 1)
    restored = restore_agent(stochastic, path)
    np.testing.assert_allclose(restored.state_scale, stochastic.state_scale)

    incompatible, _, _ = _small_agent(
        monkeypatch,
        prefix_model="low_rank_gaussian",
        prefix_rank=3,
    )
    incompatible = incompatible.replace(state_scale=stochastic.state_scale)
    with pytest.raises(ValueError, match="architecture mismatch"):
        restore_agent(incompatible, path)

    deterministic, _, _ = _small_agent(monkeypatch)
    legacy_path = save_agent(deterministic, tmp_path, 2)
    with open(legacy_path, "rb") as file:
        payload = pickle.load(file)
    payload.pop("metadata")
    payload["agent"].pop("state_scale")
    with open(legacy_path, "wb") as file:
        pickle.dump(payload, file)
    legacy_restored = restore_agent(deterministic, legacy_path)
    np.testing.assert_allclose(
        legacy_restored.state_scale,
        np.ones(observations.shape[-1], dtype=np.float32),
    )

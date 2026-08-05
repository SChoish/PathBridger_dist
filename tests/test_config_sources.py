"""PBF and Triangle-Q configuration/source invariants."""

import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    'observations', 'next_observations', 'actions', 'trajectory',
    'endpoint_goals', 'endpoint_targets', 'value_goals', 'value_offsets',
    'action_chunk_actions', 'trl_base_goals', 'trl_base_offsets',
    'trl_split_observations', 'trl_split_goals',
    'trl_split_action_chunk_actions', 'trl_split_offsets', 'trl_valid_mask',
)


def _load(path):
    spec = importlib.util.spec_from_file_location(f'cfg_{path.stem}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def test_all_eight_pbf_configs_attach_triangle_q():
    paths = sorted((ROOT / 'configs' / 'pbf').glob('*.py'))
    paths = [path for path in paths if path.name != '__init__.py']
    assert len(paths) == 8
    for path in paths:
        config = _load(path)
        assert config.endpoint_distribution == 'flow'
        assert config.sequence_horizon == config.horizon
        assert config.action_chunk_horizon == 5
        assert config.batch_size == 1024
        assert config.value_geom_sample is True
        assert config.num_qs == 2
        assert config.q_agg == 'mean'
        assert config.tau_q == 0.7
        assert config.tau_v == 0.7


def test_agent_and_sampler_share_triangle_batch_contract():
    source = (ROOT / 'agents' / 'pathbridger.py').read_text()
    tree = ast.parse(source)
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == '_REQUIRED_BATCH_KEYS'
            for target in node.targets
        )
    )
    assert ast.literal_eval(assignment.value) == REQUIRED
    dataset_source = (ROOT / 'utils' / 'datasets.py').read_text()
    for key in REQUIRED:
        assert repr(key) in dataset_source


def test_agent_has_pathflower_critic_modules_and_no_dqc_stack():
    source = (ROOT / 'agents' / 'pathbridger.py').read_text()
    for name in ('action_critic', 'target_action_critic', 'value', 'target_value'):
        assert repr(name) in source
    assert 'triangle_q_loss' in source
    assert "'phi'" in source
    assert 'infer_phi_goal_obs_indices' in source
    assert 'partial_chunk_critic' not in source
    assert 'full_chunk_critic' not in source
    assert 'kappa_d' not in source

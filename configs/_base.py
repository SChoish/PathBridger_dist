"""Build PBF configs with PathFlower's triangular-Q critic."""

from agents.pathbridger import get_config as get_agent_config


def best_config(
    *,
    env_name: str,
    endpoint_distribution: str,
    horizon: int,
    discount: float,
    endpoint_value_scale: float,
    value_distance_weight_power: float,
    eval_num_candidates: int,
    eval_temperature: float,
):
    """Attach an environment's established PBF settings to Triangle-Q."""

    if str(endpoint_distribution).lower() != 'flow':
        raise ValueError('pathbridger_triangleQ currently exposes PBF (flow) only.')
    config = get_agent_config()
    config.env_name = env_name
    config.endpoint_distribution = 'flow'
    config.horizon = int(horizon)
    config.sequence_horizon = int(horizon)
    config.action_chunk_horizon = 5
    config.discount = float(discount)
    config.batch_size = 1024
    config.actor_p = (0.0, 0.0, 1.0, 0.0)
    config.value_geom_sample = True
    config.num_qs = 2
    config.q_agg = 'mean'
    config.tau = 0.005
    config.tau_q = 0.7
    config.tau_v = 0.7
    config.lambda_q_base = 1.0
    config.lambda_q_tri = 1.0
    config.lambda_v = 1.0
    config.value_distance_weight_power = float(value_distance_weight_power)
    config.endpoint_value_scale = float(endpoint_value_scale)
    config.eval_num_candidates = int(eval_num_candidates)
    config.eval_temperature = float(eval_temperature)
    return config

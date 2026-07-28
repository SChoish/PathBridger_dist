"""Build the environment-specific configurations reported in the paper."""

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
    config = get_agent_config()
    config.env_name = env_name
    config.endpoint_distribution = endpoint_distribution
    config.horizon = horizon
    config.discount = discount
    config.endpoint_value_scale = endpoint_value_scale
    config.value_distance_weight_power = value_distance_weight_power
    config.eval_num_candidates = eval_num_candidates
    config.eval_temperature = eval_temperature
    return config


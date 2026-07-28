from configs._base import best_config


def get_config():
    return best_config(
        env_name='antmaze-large-navigate-v0',
        endpoint_distribution='flow',
        horizon=25,
        discount=0.995,
        endpoint_value_scale=10.0,
        value_distance_weight_power=0.0,
        eval_num_candidates=32,
        eval_temperature=0.5,
    )


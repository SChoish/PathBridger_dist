from configs._base import best_config


def get_config():
    return best_config(
        env_name='puzzle-3x3-play-v0',
        endpoint_distribution='flow',
        horizon=25,
        discount=0.99,
        endpoint_value_scale=10.0,
        value_distance_weight_power=0.5,
        eval_num_candidates=32,
        eval_temperature=1.0,
    )


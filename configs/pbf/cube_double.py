from configs._base import best_config


def get_config():
    return best_config(
        env_name='cube-double-play-v0',
        endpoint_distribution='flow',
        horizon=40,
        discount=0.99,
        endpoint_value_scale=10.0,
        value_distance_weight_power=1.0,
        eval_num_candidates=8,
        eval_temperature=0.25,
    )


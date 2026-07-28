from configs._base import best_config


def get_config():
    return best_config(
        env_name='cube-single-play-v0',
        endpoint_distribution='gaussian',
        horizon=25,
        discount=0.99,
        endpoint_value_scale=10.0,
        value_distance_weight_power=0.7,
        eval_num_candidates=1,
        eval_temperature=0.0,
    )


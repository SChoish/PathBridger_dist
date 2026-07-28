from configs._base import best_config


def get_config():
    return best_config(
        env_name='scene-play-v0',
        endpoint_distribution='gaussian',
        horizon=40,
        discount=0.99,
        endpoint_value_scale=5.0,
        value_distance_weight_power=1.0,
        eval_num_candidates=16,
        eval_temperature=0.5,
    )


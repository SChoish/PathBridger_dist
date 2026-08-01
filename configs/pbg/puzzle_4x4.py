from configs._base import best_config


def get_config():
    return best_config(
        env_name='puzzle-4x4-play-v0',
        endpoint_distribution='gaussian',
        horizon=40,
        discount=0.995,
        endpoint_value_scale=10.0,
        value_distance_weight_power=2.0,
        eval_num_candidates=16,
        eval_temperature=0.5,
    )


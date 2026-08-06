from configs.pbf.cube_triple import get_config as _base


def get_config():
    config = _base()
    config.offline_action_free = True
    return config

from os import path, mkdir


def root_path():
    return path.dirname(__file__)


MODELS_DIR = path.join(f"{root_path()}", "models")

if not path.exists(MODELS_DIR):
    mkdir(MODELS_DIR)

WEIGHTS_DIR = path.join(f"{root_path()}", "weights")

if not path.exists(WEIGHTS_DIR):
    mkdir(WEIGHTS_DIR)

PARAMS_DIR = path.join(f"{root_path()}", "params")

if not path.exists(PARAMS_DIR):
    mkdir(PARAMS_DIR)

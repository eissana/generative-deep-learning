from os import path, mkdir
import logging


def root_path():
    return path.dirname(__file__)


def get_logger(mod_name):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s [%(asctime)s] %(message)s')
    log = logging.getLogger(mod_name)
    return log


MODELS_DIR = path.join(f"{root_path()}", "models")

if not path.exists(MODELS_DIR):
    mkdir(MODELS_DIR)

WEIGHTS_DIR = path.join(f"{root_path()}", "weights")

if not path.exists(WEIGHTS_DIR):
    mkdir(WEIGHTS_DIR)

PARAMS_DIR = path.join(f"{root_path()}", "params")

if not path.exists(PARAMS_DIR):
    mkdir(PARAMS_DIR)

DATA_DIR = path.join(f"{root_path()}", "data")

if not path.exists(DATA_DIR):
    mkdir(DATA_DIR)

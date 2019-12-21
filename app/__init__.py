from os import path, mkdir

MODELS_DIR = "models"

if not path.exists(MODELS_DIR):
    mkdir(MODELS_DIR)

MODELS_PLOT_DIR = "viz"

if not path.exists(MODELS_PLOT_DIR):
    mkdir(MODELS_PLOT_DIR)


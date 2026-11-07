import os
import random
from pathlib import Path
import numpy as np
import tensorflow as tf

# Model Architecture
BACKBONE = "EfficientNetB0"
IMG_SIZE = (224, 224)
INPUT_SIZE = (224, 224)
IMG_CHANNELS = 3
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
NUM_CLASSES = 38
DROPOUT_RATE = 0.4

# CBAM Settings
CBAM_REDUCTION_RATIO = 8
CBAM_KERNEL_SIZE = 7
CBAM_SPATIAL_KERNEL = 7
DEFAULT_CBAM_STAGES = [4, 5]

# Paths & Weights
WEIGHTS_FILE = "best.weights.h5"
DATA_DIR = Path("datasets/plantvillage")
PLANTDOC_DIR = Path("datasets/plantdoc")
MODELS_DIR = Path("models")
VIS_DIR = Path("visualizations")

# Data Splits
SPLIT_NAMES = ["train", "val"]

# Dynamically load PlantVillage class names
_pv_path = DATA_DIR
if _pv_path.exists():
    PLANTVILLAGE_CLASSES = sorted([d.name for d in _pv_path.iterdir() if d.is_dir()])
else:
    PLANTVILLAGE_CLASSES = [str(i) for i in range(NUM_CLASSES)]

# Training Hyperparameters
EPOCHS_HEAD = 10
EPOCHS_FINETUNE = 20
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
UNFREEZE_LAYERS = 80
SEED = 42

def set_global_seed(seed=SEED):
    """Sets random seeds for full reproducibility across NumPy, Python, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def ensure_dirs():
    """Ensures that required output directories exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

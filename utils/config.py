# import the necessary packages
from pathlib import Path

import torch

# base path of the dataset
DATASET_PATH = Path("dataset") / "train_03"

# define the path to the images and masks dataset
IMAGE_DATASET_PATH = (DATASET_PATH / "images").as_posix()
MASK_DATASET_PATH = (DATASET_PATH / "masks").as_posix()

# define the explosion-crater-detection split
TEST_SPLIT = 0.15

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 60
BATCH_SIZE = 64

# define the input image dimensions
INPUT_IMAGE_WIDTH = 64
INPUT_IMAGE_HEIGHT = 64

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory
BASE_OUTPUT = Path("output")

# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = (BASE_OUTPUT / "unet.pt").as_posix()
PLOT_PATH = (BASE_OUTPUT / "plot.png").as_posix()
TEST_PATHS = (BASE_OUTPUT / "test_paths.txt").as_posix()

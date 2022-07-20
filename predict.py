# USAGE
# python predict.py

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from osgeo import gdal

# import the necessary packages
from utils import config

logging.basicConfig(
    filename="logs.log",
    filemode='a',
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO
)


def prepare_plot(orig_image, orig_mask, pred_mask):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(orig_image)
    ax[1].imshow(orig_mask)
    ax[2].imshow(pred_mask)

    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")

    # set the layout of the figure and display it
    figure.tight_layout()
    figure.show()


def make_predictions(model, image_path):
    # set model to evaluation mode
    model.eval()

    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image_dataset = gdal.Open(image_path)
        image = image_dataset.ReadAsArray().transpose(1, 2, 0)
        image = (image / 18496).astype(np.float32)

        # resize the image and make a copy of it for visualization
        orig = image.copy()

        # find the filename and generate the path to ground truth
        # mask
        filename = Path(image_path).name
        ground_truth_path = (Path(config.MASK_DATASET_PATH) / filename).as_posix()

        # load the ground-truth segmentation mask in grayscale mode
        # and resize it
        mask_dataset = gdal.Open(ground_truth_path)
        gt_mask = mask_dataset.ReadAsArray()
        gt_mask = (gt_mask * 255).astype(np.uint8)

        # make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(config.DEVICE)

        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        pred_mask = model(image).squeeze()

        pred_mask -= pred_mask.min(1, keepdim=True)[0]
        pred_mask /= pred_mask.max(1, keepdim=True)[0]
        # pred_mask = torch.sigmoid(pred_mask)
        pred_mask = pred_mask.detach().cpu().numpy()

        # filter out the weak predictions and convert them to integers
        pred_mask = (pred_mask > config.THRESHOLD) * 255
        pred_mask = pred_mask.astype(np.uint8)
        # prepare a plot for visualization
        prepare_plot(orig, gt_mask, pred_mask)


# load the image paths in our testing file and randomly select 10
# image paths
logging.info("loading up explosion-crater-detection image paths...")
image_paths = open(config.TEST_PATHS).read().strip().split("\n")
image_paths = np.random.choice(image_paths, size=20)

# load our model from disk and flash it to the current device
logging.info("load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)

# iterate over the randomly selected explosion-crater-detection image paths
for path in image_paths:
    # make predictions and visualize the results
    make_predictions(unet, path)

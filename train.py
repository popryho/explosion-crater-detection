# USAGE
# python train.py
# import the necessary packages
import logging
import os
import time

import matplotlib.pyplot as plt
import torch
from imutils import paths
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils import config
from utils.dataset import SegmentationDataset
from utils.model import UNet

logging.basicConfig(
    filename="logs.log",
    filemode='a',
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO
)

# load the image and mask filepaths in a sorted manner
image_paths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
mask_paths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

# partition the data into training and testing splits using 85% of
# the data for training and the remaining 15% for testing
train_images, test_images, train_masks, test_masks = train_test_split(
    image_paths, mask_paths, test_size=config.TEST_SPLIT, random_state=42)

# write the testing image paths to disk so that we can use then
# when evaluating/testing our model
logging.info("saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(test_images))
f.close()

# define transformations
transforms = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize((config.INPUT_IMAGE_HEIGHT,
    #                    config.INPUT_IMAGE_WIDTH)),
    transforms.ToTensor()
])

# create the train and explosion-crater-detection datasets
train_ds = SegmentationDataset(image_paths=train_images, mask_paths=train_masks,
                               transforms=transforms)
test_ds = SegmentationDataset(image_paths=test_images, mask_paths=test_masks,
                              transforms=transforms)

logging.info(f"found {len(train_ds)} examples in the training set...")
logging.info(f"found {len(test_ds)} examples in the testing set...")

# create the training and explosion-crater-detection data loaders
train_loader = DataLoader(train_ds, shuffle=True,
                          batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                          num_workers=os.cpu_count())
test_loader = DataLoader(test_ds, shuffle=False,
                         batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                         num_workers=os.cpu_count())

# initialize our UNet model
unet = UNet().to(config.DEVICE)

# initialize loss function and optimizer
loss_func = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr=config.INIT_LR)

# calculate steps per epoch for training and explosion-crater-detection set
train_steps = len(train_ds) // config.BATCH_SIZE
test_steps = len(test_ds) // config.BATCH_SIZE

# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}

# loop over epochs
logging.info("training the network...\n\n")
startTime = time.time()

for epoch in range(config.NUM_EPOCHS):
    # set the model in training mode
    unet.train()

    # initialize the total training and validation loss
    total_train_loss = 0
    total_test_loss = 0

    tqdm_data = tqdm(train_loader, desc=f'Training (epoch #{epoch})', total=int(len(train_loader)))
    # loop over the training set
    for (i, (x, y)) in enumerate(tqdm_data):
        # send the input to the device
        (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

        # perform a forward pass and calculate the training loss
        pred = unet(x)
        loss = loss_func(pred, y)

        # first, zero out any previously accumulated gradients, then
        # perform backpropagation, and then update model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()

        # add the loss to the total training loss so far
        total_train_loss += loss
        tqdm_data.set_postfix(loss=(total_train_loss / ((i + 1) * train_loader.batch_size)))

    # switch off autograd
    with torch.no_grad():
        # set the model in evaluation mode
        unet.eval()

        tqdm_data = tqdm(test_loader, desc=f'Validation (epoch #{epoch})', total=int(len(test_loader)))
        # loop over the validation set
        for i, (x, y) in enumerate(tqdm_data):
            # send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

            # make the predictions and calculate the validation loss
            pred = unet(x)
            total_test_loss += loss_func(pred, y)
            tqdm_data.set_postfix(loss=(total_test_loss / ((i + 1) * test_loader.batch_size)))

    # calculate the average training and validation loss
    avg_train_loss = total_train_loss / train_steps
    avg_test_loss = total_test_loss / test_steps

    # update our training history
    H["train_loss"].append(avg_train_loss.cpu().detach().numpy())
    H["test_loss"].append(avg_test_loss.cpu().detach().numpy())

    # print the model training and validation information
    logging.info(f"Train loss: {avg_train_loss:.6f}, Test loss: {avg_test_loss:.4f}")

# display the total time needed to perform the training
end_time = time.time()
logging.info(f"total time taken to train the model: {end_time - startTime:.2f}s")

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)

# serialize the model to disk
torch.save(unet, config.MODEL_PATH)

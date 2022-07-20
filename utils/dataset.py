# import the necessary packages
# import cv2
import numpy as np
from osgeo import gdal
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __getitem__(self, index):
        # grab the image path from the current index
        image_path = self.image_paths[index]
        mask_paths = self.mask_paths[index]

        # load the image from disk, swap its channels from BGR to RGB,
        # and read the associated mask from disk in grayscale mode
        image_dataset = gdal.Open(image_path)
        mask_dataset = gdal.Open(mask_paths)

        # image = (image_dataset.ReadAsArray().transpose(1, 2, 0)).astype(np.uint8)
        image = image_dataset.ReadAsArray().transpose(1, 2, 0)
        image = (image / 18496).astype(np.float32)
        mask = mask_dataset.ReadAsArray()
        mask = (mask * 255).astype(np.uint8)
        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image = self.transforms(image)
            mask = self.transforms(mask)
        # return a tuple of the image and its mask
        return image, mask

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.image_paths)

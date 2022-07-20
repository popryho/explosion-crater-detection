"""
` python utils/image_cutting.py
    --image_path_input data/data_part/S2B_tile_20220702_37UDQ_cut.tif
    --mask_path_input data/data_part/tr_class_cut.tif
    --image_path_output dataset/train_02/images
    --mask_path_output dataset/train_02/masks
    --image_size 64 --step 16
`

"""
import argparse
from pathlib import Path

import numpy as np
from osgeo import gdal, osr


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--image_path_input', type=str, required=True,
                        default='data/data_part/S2B_tile_20220702_37UDQ_cut.tif',
                        help='specify the local path to .tif [Tagged Image Format] file with the image')
    parser.add_argument('--mask_path_input', type=str, required=True,
                        default='data/data_part/tr_class_cut.tif',
                        help='specify the local path to .tif [Tagged Image Format] file with the mask')

    parser.add_argument('--image_path_output', type=str,
                        help='specify the output folder for cropped .tif files for image')
    parser.add_argument('--mask_path_output', type=str,
                        help='specify the output folder for cropped .tif files for mask')

    parser.add_argument('--threshold', type=float, default=1e-1,
                        help='Threshold to prevent unlabeled image saves')
    parser.add_argument('--image_size', type=int, default=64,
                        help='Image size HxW')
    parser.add_argument('--step', type=int, default=64,
                        help='Step when cutting the image')
    return parser


def crop_image(image: gdal.Dataset, output_path: str, i: int, j: int, x_size: int, y_size: int, idx: int):
    Path(output_path).mkdir(parents=True, exist_ok=True)

    output_path = Path(output_path) / f'{idx:05d}.tif'

    driver = gdal.GetDriverByName("GTiff")
    raster_srs = osr.SpatialReference()

    e_type = gdal.GDT_Byte if image.RasterCount == 1 else gdal.GDT_UInt16
    output_dataset = driver.Create(output_path.as_posix(), xsize=x_size, ysize=y_size, bands=image.RasterCount,
                                   eType=e_type)

    raster_srs.ImportFromWkt(image.GetProjectionRef())
    output_dataset.SetProjection(raster_srs.ExportToWkt())
    output_dataset.SetGeoTransform(image.GetGeoTransform())

    for b in range(image.RasterCount):
        band = image.GetRasterBand(b + 1).ReadAsArray(i, j, x_size, y_size)
        output_dataset.GetRasterBand(b + 1).WriteArray(band)


def run(config: argparse.Namespace):
    image = gdal.Open(config.image_path_input)
    mask = gdal.Open(config.mask_path_input)

    xsize = ysize = config.image_size
    step = config.step

    if not config.image_path_output:
        config.image_path_output = \
            (Path(config.image_path_input).parent.parent / Path(config.image_path_input).stem).as_posix()
    if not config.mask_path_output:
        config.mask_path_output = \
            (Path(config.mask_path_input).parent.parent / Path(config.mask_path_input).stem).as_posix()
    idx = 0
    for i in range(0, image.RasterXSize - xsize, step):
        for j in range(0, image.RasterYSize - ysize, step):
            # prevent saving images without any craters
            band = mask.GetRasterBand(1).ReadAsArray(i, j, xsize, ysize)
            if np.sum(band == 1) / (xsize * ysize) < config.threshold:
                continue
            # Create mask
            crop_image(mask, config.mask_path_output, i, j, xsize, ysize, idx)
            # Create image
            crop_image(image, config.image_path_output, i, j, xsize, ysize, idx)
            idx += 1


if __name__ == '__main__':

    args = get_parser().parse_args()
    run(args)
